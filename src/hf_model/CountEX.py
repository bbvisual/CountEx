# coding=utf-8
"""
Negative Grounding DINO Model for Object Detection with Negative Caption Support.
This module extends the original GroundingDinoForObjectDetection to support negative captions
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from transformers.modeling_outputs import ModelOutput
import torch.nn.functional as F
from .modeling_grounding_dino import (
    GroundingDinoForObjectDetection,
    GroundingDinoObjectDetectionOutput,
    GroundingDinoEncoderOutput,
)


# density_fpn_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bilinear(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class DensityFPNHead(nn.Module):
    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        # ---- 1×1 lateral convs (P3–P6) ----
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_channels, mid_channels, 1) for _ in range(4)
        ])

        # ---- smooth convs after add ----
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                norm_layer(mid_channels),
                act_layer(inplace=True),
            ) for _ in range(3)          # P6→P5, P5→P4, P4→P3
        ])

        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                act_layer(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                norm_layer(mid_channels),
                act_layer(inplace=True),
            ) for _ in range(3)          # 167×94 → … → 1336×752
        ])

        # ---- output 3×3 conv -> 1 ----
        self.out_conv = nn.Conv2d(mid_channels, 1, 3, padding=1, bias=False)

    def forward(self, feats):
        assert len(feats) == 4, "Expect feats list = [P3,P4,P5,P6]"

        # lateral 1×1
        lat = [l(f) for l, f in zip(self.lateral, feats)]

        # top-down FPN fusion
        x = lat[-1]                              # P6
        for i in range(3)[::-1]:                 # P5,P4,P3
            x = _bilinear(x, lat[i].shape[-2:])
            x = x + lat[i]
            x = self.smooth[i](x)

        # three-stage upsample + conv
        for up in self.up_blocks:
            h, w = x.shape[-2], x.shape[-1]
            x = _bilinear(x, (h * 2, w * 2))
            x = up(x)

        x = self.out_conv(x)
        return F.relu(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

# -----------------------------------
# 1) CommonFinderSimple
#    learn r "common prototypes", representing the common representative of positive/negative
#    non fancy: only MHA pooling + two light regularizations (shareability + diversity)
# -----------------------------------
class CommonFinderSimple(nn.Module):
    """
    Inputs:
      Q_pos: [B, K, D]
      Q_neg: [B, K, D]
    Returns:
      C_rows: [B, r, D]   # batch copied r common prototypes (unitized)
      loss:   scalar      # small regularization: shareability + diversity
      stats:  dict
    """
    def __init__(self, d_model=256, r=64, nhead=4,
                 share_w=0.02, div_w=0.02, ln_after=False):
        super().__init__()
        self.r = r
        self.share_w = share_w
        self.div_w = div_w

        proto = torch.randn(r, d_model)
        self.proto = nn.Parameter(l2norm(proto, -1))     # r×D learnable "core queries"
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.post  = nn.Linear(d_model, d_model)
        self.ln    = nn.LayerNorm(d_model) if ln_after else nn.Identity()

    def forward(self, Q_pos: torch.Tensor, Q_neg: torch.Tensor):
        B, K, D = Q_pos.shape
        seeds = self.proto[None].expand(B, -1, -1).contiguous()   # [B,r,D]
        X = torch.cat([Q_pos, Q_neg], dim=1)                      # [B,2K,D]

        # use seeds to do one attention pooling on positive and negative sets, get r "common prototypes"
        C, _ = self.attn(query=seeds, key=X, value=X)             # [B,r,D]
        C = l2norm(self.ln(self.post(C)), -1)                     # unitization

        # ---- Simple regularization: encourage C to be close to both Q_pos and Q_neg, and diverse from each other ----
        # Shareability: average of maximum cosine similarity between C and Q_pos/Q_neg
        cos_pos = torch.einsum('brd,bkd->brk', C, l2norm(Q_pos, -1))  # [B,r,K]
        cos_neg = torch.einsum('brd,bkd->brk', C, l2norm(Q_neg, -1))
        share_term = -(cos_pos.amax(dim=-1).mean() + cos_neg.amax(dim=-1).mean())

        # Diversity: cosine between C should not collapse
        C0 = l2norm(self.proto, -1)                       # [r,D]
        gram = C0 @ C0.t()                                # [r,r]
        div_term = (gram - torch.eye(self.r, device=gram.device)).pow(2).mean()

        loss = self.share_w * share_term + self.div_w * div_term
        stats = {
            'share_term': share_term.detach(),
            'div_term':   div_term.detach(),
            'mean_cos_pos': cos_pos.mean().detach(),
            'mean_cos_neg': cos_neg.mean().detach()
        }
        return C, loss, stats


# -----------------------------------
# 2) NegExclusiveSimple
#    Remove "common" information from negative queries: two simple strategies can be used independently or together
#    (A) Soft removal: subtract the projection onto C (residual keeps non-common)
#    (B) Filtering: only keep the Top-M negative samples least similar to C
# -----------------------------------
class NegExclusiveSimple(nn.Module):
    """
    Inputs:
      Q_neg: [B,K,D]
      C_rows: [B,r,D]   # common prototypes
    Args:
      mode: 'residual' | 'filter' | 'both'
      M:    Top-M for 'filter'
      thresh: Filter threshold (max_cos_neg < thresh to keep), None means only use Top-M
    Returns:
      neg_refs: [B, M_or_K, D]  # as negative reference (for next fusion)
      aux: dict
    """
    def __init__(self, mode='residual', M=16, thresh=None):
        super().__init__()
        assert mode in ('residual', 'filter', 'both')
        self.mode = mode
        self.M = M
        self.thresh = thresh

    def forward(self, Q_neg: torch.Tensor, C_rows: torch.Tensor):
        B, K, D = Q_neg.shape
        r = C_rows.size(1)
        Qn = l2norm(Q_neg, -1)
        C  = l2norm(C_rows, -1)

        sim = torch.einsum('bkd,brd->bkr', Qn, C).amax(dim=-1)   # [B,K]

        outputs = {}
        if self.mode in ('residual', 'both'):
            # proj = (Q · C^T) C  -> [B,K,D]; first weight [B,K,r], then multiply C [B,r,D]
            w = torch.einsum('bkd,brd->bkr', Qn, C)              # [B,K,r]
            proj = torch.einsum('bkr,brd->bkd', w, C)            # [B,K,D]
            neg_resid = l2norm(Qn - proj, -1)                    # non-common residual
            outputs['residual'] = neg_resid

        if self.mode in ('filter', 'both'):
            excl_score = 1.0 - sim                               # large = away from common
            if self.thresh is not None:
                mask = (sim < self.thresh).float()
                excl_score = excl_score * mask + (-1e4) * (1 - mask)
            M = min(self.M, K)
            topv, topi = torch.topk(excl_score, k=M, dim=1)      # [B,M]
            neg_top = torch.gather(Qn, 1, topi.unsqueeze(-1).expand(-1, -1, D))
            outputs['filtered'] = neg_top

        if self.mode == 'residual':
            neg_refs = outputs['residual']
        elif self.mode == 'filter':
            neg_refs = outputs['filtered']
        else:
            R = outputs['residual']                  # [B,K,D]
            excl_score = 1.0 - sim
            M = min(self.M, K)
            topv, topi = torch.topk(excl_score, k=M, dim=1)
            neg_refs = torch.gather(R, 1, topi.unsqueeze(-1).expand(-1, -1, D))  # [B,M,D]

        aux = {
            'mean_sim_to_common': sim.mean().detach(),
            'kept_M': neg_refs.size(1)
        }
        return neg_refs, aux

import torch
import torch.nn as nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

class FusionNoGate(nn.Module):
    """
    Direct fusion (no gating): fuse neg_ref into Q_pos via one cross-attn.
    Variants:
      - 'residual_sub': Q_new = Q_pos - scale * LN(Z)
      - 'residual_add': Q_new = Q_pos + scale * LN(Z)
      - 'concat_linear': Q_new = Q_pos + Linear([Q_pos; Z])
    """
    def __init__(self, d_model=256, nhead=4, fusion_mode='residual_sub',
                 init_scale=0.2, dropout_p=0.0):
        super().__init__()
        assert fusion_mode in ('residual_sub', 'residual_add', 'concat_linear')
        self.fusion_mode = fusion_mode
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln_z = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        if fusion_mode == 'concat_linear':
            self.mix = nn.Linear(2 * d_model, d_model)
            nn.init.zeros_(self.mix.weight)
            nn.init.zeros_(self.mix.bias)

    def forward(self, Q_pos: torch.Tensor, neg_ref: torch.Tensor):
        """
        Q_pos:   [B, K, D]
        neg_ref: [B, M, D]
        return:  Q_new [B, K, D], stats dict
        """
        B, K, D = Q_pos.shape
        M = neg_ref.size(1)
        if M == 0:
            return Q_pos, {'kept': 0, 'scale': self.scale.detach()}

        # 1) Cross-attention:
        Z, attn_w = self.attn(query=Q_pos, key=neg_ref, value=neg_ref)  # Z:[B,K,D]
        Z = self.ln_z(Z)
        Z = self.drop(Z)

        # 2) wo gating
        if self.fusion_mode == 'residual_sub':
            Q_new = Q_pos - self.scale * Z
            # print("z: ", Z.sum())
            # print(torch.abs(Q_new - Q_pos).sum())
        elif self.fusion_mode == 'residual_add':
            Q_new = Q_pos + self.scale * Z
        else:  # 'concat_linear'
            fused = torch.cat([Q_pos, Z], dim=-1)      # [B,K,2D]
            delta = self.mix(fused)                    # [B,K,D]
            Q_new = Q_pos + delta

        stats = {
            'kept': M,
            'attn_mean': attn_w.mean().detach(),
            'fusion_scale': self.scale.detach()
        }
        return Q_new, stats

class QuerySideNegNaive(nn.Module):
    def __init__(self, d_model=256, r=64, M=64, nhead=4,
                 excl_mode='both', excl_thresh=0.5, gamma_max=0.7,
                 share_w=0.02, div_w=0.02):
        super().__init__()
        self.common = CommonFinderSimple(d_model, r, nhead, share_w, div_w)
        self.excl   = NegExclusiveSimple(mode=excl_mode, M=M, thresh=excl_thresh)
        self.fuse = FusionNoGate(d_model=d_model,
                    nhead=4,
                    fusion_mode='residual_sub',   # or 'concat_linear'
                    init_scale=0.25,
                    dropout_p=0.1)

    def forward(self, Q_pos: torch.Tensor, Q_neg: torch.Tensor):
        C_rows, l_common, common_stats = self.common(Q_pos, Q_neg)
        neg_refs, excl_stats = self.excl(Q_neg, C_rows)
        Q_new, fuse_stats = self.fuse(Q_pos, neg_refs)
        loss = l_common
        stats = {}
        stats.update(common_stats); stats.update(excl_stats); stats.update(fuse_stats)
        return Q_new, loss, stats
    
    def set_fusion_scale(self, scale: float):
        del self.fuse.scale
        self.fuse.scale = nn.Parameter(torch.tensor(scale))


class CountEX(GroundingDinoForObjectDetection):
    """
    Grounding DINO Model with negative caption support for improved object detection.
    
    This model extends the original GroundingDinoForObjectDetection by adding
    support for negative captions, which helps improve detection accuracy by
    learning what NOT to detect.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize negative fusion modules directly in __init__
        self.query_side_neg_pipeline = QuerySideNegNaive()
        self.density_head = DensityFPNHead()
        self.config = config
        self.box_threshold = getattr(config, 'box_threshold', 0.4)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Union[GroundingDinoEncoderOutput, Tuple]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: List[Dict[str, Union[torch.LongTensor, torch.FloatTensor]]] = None,
        # Negative prompt parameters
        neg_pixel_values: Optional[torch.FloatTensor] = None,
        neg_input_ids: Optional[torch.LongTensor] = None,
        neg_token_type_ids: Optional[torch.LongTensor] = None,
        neg_attention_mask: Optional[torch.LongTensor] = None,
        neg_pixel_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):  
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_neg = kwargs.get('use_neg', True)
        # Get positive outputs
        pos_kwargs = {
            'exemplars': kwargs.get('pos_exemplars', None),
        }
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **pos_kwargs,
        )

        spatial_shapes = outputs.spatial_shapes
        token_num = 0
        token_num_list = [0]
        for i in range(len(spatial_shapes)):
            token_num += spatial_shapes[i][0] * spatial_shapes[i][1]
            token_num_list.append(token_num.item())
        
        positive_feature_maps = []
        encoder_last_hidden_state_vision = outputs.encoder_last_hidden_state_vision
        for i in range(len(spatial_shapes)):
            feature_map = encoder_last_hidden_state_vision[:, token_num_list[i]:token_num_list[i+1], :]
            spatial_shape = spatial_shapes[i]
            b, t, d = feature_map.shape
            feature_map = feature_map.reshape(b, spatial_shape[0], spatial_shape[1], d)
            positive_feature_maps.append(feature_map)

        # Get negative outputs 
        neg_kwargs = {
            'exemplars': kwargs.get('neg_exemplars', None),
        }
        # print(kwargs)
        neg_outputs = self.model(
            pixel_values=neg_pixel_values,
            input_ids=neg_input_ids,
            token_type_ids=neg_token_type_ids,
            attention_mask=neg_attention_mask,
            pixel_mask=neg_pixel_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **neg_kwargs,
        )

        neg_encoder_last_hidden_state_vision = neg_outputs.encoder_last_hidden_state_vision
        neg_positive_feature_maps = []
        for i in range(len(spatial_shapes)):
            feature_map = neg_encoder_last_hidden_state_vision[:, token_num_list[i]:token_num_list[i+1], :]
            spatial_shape = spatial_shapes[i]
            b, t, d = feature_map.shape
            feature_map = feature_map.reshape(b, spatial_shape[0], spatial_shape[1], d)
            neg_positive_feature_maps.append(feature_map)

        if return_dict:
            hidden_states = outputs.intermediate_hidden_states
            neg_hidden_states = neg_outputs.intermediate_hidden_states
        else:
            hidden_states = outputs[2]
            neg_hidden_states = neg_outputs[2]
        
        idx = 5 + (1 if output_attentions else 0) + (1 if output_hidden_states else 0)
        enc_text_hidden_state = outputs.encoder_last_hidden_state_text if return_dict else outputs[idx]
        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        init_reference_points = outputs.init_reference_points if return_dict else outputs[1]
        inter_references_points = outputs.intermediate_reference_points if return_dict else outputs[3]

        # drop the exemplar tokens if used 
        pos_exemplars = pos_kwargs.get('pos_exemplars', None)
        neg_exemplars = neg_kwargs.get('neg_exemplars', None)
        if pos_exemplars is not None or neg_exemplars is not None or attention_mask.shape[1] != enc_text_hidden_state.shape[1]:
            enc_text_hidden_state = enc_text_hidden_state[:, :enc_text_hidden_state.shape[1] - 3, :]

        # class logits + predicted bounding boxes
        outputs_classes = []
        outputs_coords = []

        # Apply negative fusion
        if use_neg:
            # print("Using negative fusions")
            #neg_hidden_states = self.negative_semantic_extractor(neg_hidden_states)
            #hidden_states = self.negative_fusion_module(hidden_states, neg_hidden_states)
            hidden_states = hidden_states.squeeze(0)
            neg_hidden_states = neg_hidden_states.squeeze(0)
            hidden_states, extra_loss, logs = self.query_side_neg_pipeline(hidden_states, neg_hidden_states)
            hidden_states = hidden_states.unsqueeze(0)
            neg_hidden_states = neg_hidden_states.unsqueeze(0)
            # print("extra_loss: ", extra_loss)
        else:
            # print("Not using negative fusions")
            extra_loss = None
            logs = None
            # print("Not using negative fusion")
        # print("extra_loss: ", extra_loss)
        
        # predict class and bounding box deltas for each stage
        num_levels = hidden_states.shape[1]
        for level in range(num_levels):
            if level == 0:
                reference = init_reference_points
            else:
                reference = inter_references_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)

            # print("hidden_states[:, level]: ", hidden_states[:, level].shape)
            # print("enc_text_hidden_state: ", enc_text_hidden_state.shape)
            # print("attention_mask: ", attention_mask.shape)

            assert attention_mask.shape[1] == enc_text_hidden_state.shape[1], "Attention mask and text hidden state have different lengths: {} != {}".format(attention_mask.shape[1], enc_text_hidden_state.shape[1])
            outputs_class = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=attention_mask.bool(),
            )
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])

            reference_coordinates = reference.shape[-1]
            if reference_coordinates == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference_coordinates == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}")
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            tuple_outputs = ((loss, loss_dict) + output) if loss is not None else output

            return tuple_outputs
        
        all_feats = []
        for pf, npf in zip(positive_feature_maps, neg_positive_feature_maps):
            pf = pf.permute(0, 3, 1, 2)
            npf = npf.permute(0, 3, 1, 2)
            all_feats.append(torch.cat([pf, npf], dim=1))
        
        
        # pos_feat = positive_feature_maps[0].permute(0, 3, 1, 2)
        # neg_feat = neg_positive_feature_maps[0].permute(0, 3, 1, 2)
        # pos_minus_neg_feat = F.relu(pos_feat - neg_feat)
        # density_feat_map = torch.cat([pos_feat, neg_feat, pos_minus_neg_feat], dim=1)
        # density_feat_map = torch.cat([pos_feat, neg_feat], dim=1)
        density_map_pred = self.density_head(all_feats)

        dict_outputs = GroundingDinoObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            last_hidden_state=outputs.last_hidden_state,
            auxiliary_outputs=auxiliary_outputs,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state_vision=outputs.encoder_last_hidden_state_vision,
            encoder_last_hidden_state_text=outputs.encoder_last_hidden_state_text,
            encoder_vision_hidden_states=outputs.encoder_vision_hidden_states,
            encoder_text_hidden_states=outputs.encoder_text_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
            spatial_shapes=outputs.spatial_shapes,
            positive_feature_maps=positive_feature_maps,
            negative_feature_maps=neg_positive_feature_maps,
            density_map_pred=density_map_pred,
            extra_loss=extra_loss,
            extra_logs=logs,
        )

        return dict_outputs