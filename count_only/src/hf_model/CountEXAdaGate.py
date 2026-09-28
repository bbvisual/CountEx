# coding=utf-8
"""
CountEX with Adaptive MLP Gating (S1 + S3)

Changes from original CountEX:
1. S1: Density Head uses pos-neg difference feature (in_channels: 512 -> 768)
2. S3: FusionAdaGate uses MLP to predict per-query adaptive gate
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


def _bilinear(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class DensityFPNHead(nn.Module):
    """
    S1: in_channels changed from 512 to 768 to accommodate [pos, neg, relu(pos-neg)]
    """
    def __init__(self,
                 in_channels: int = 768,  # Changed: 512 -> 768
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
            ) for _ in range(3)
        ])

        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                act_layer(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                norm_layer(mid_channels),
                act_layer(inplace=True),
            ) for _ in range(3)
        ])

        # ---- output 3×3 conv -> 1 ----
        self.out_conv = nn.Conv2d(mid_channels, 1, 3, padding=1, bias=False)

    def forward(self, feats):
        assert len(feats) == 4, "Expect feats list = [P3,P4,P5,P6]"

        # lateral 1×1
        lat = [l(f) for l, f in zip(self.lateral, feats)]

        # top-down FPN fusion
        x = lat[-1]
        for i in range(3)[::-1]:
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


def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class CommonFinderSimple(nn.Module):
    """
    Unchanged from original.
    """
    def __init__(self, d_model=256, r=64, nhead=4,
                 share_w=0.02, div_w=0.02, ln_after=False):
        super().__init__()
        self.r = r
        self.share_w = share_w
        self.div_w = div_w

        proto = torch.randn(r, d_model)
        self.proto = nn.Parameter(l2norm(proto, -1))
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.post = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model) if ln_after else nn.Identity()

    def forward(self, Q_pos: torch.Tensor, Q_neg: torch.Tensor):
        B, K, D = Q_pos.shape
        seeds = self.proto[None].expand(B, -1, -1).contiguous()
        X = torch.cat([Q_pos, Q_neg], dim=1)

        C, _ = self.attn(query=seeds, key=X, value=X)
        C = l2norm(self.ln(self.post(C)), -1)

        cos_pos = torch.einsum('brd,bkd->brk', C, l2norm(Q_pos, -1))
        cos_neg = torch.einsum('brd,bkd->brk', C, l2norm(Q_neg, -1))
        share_term = -(cos_pos.amax(dim=-1).mean() + cos_neg.amax(dim=-1).mean())

        C0 = l2norm(self.proto, -1)
        gram = C0 @ C0.t()
        div_term = (gram - torch.eye(self.r, device=gram.device)).pow(2).mean()

        loss = self.share_w * share_term + self.div_w * div_term
        stats = {
            'share_term': share_term.detach(),
            'div_term': div_term.detach(),
            'mean_cos_pos': cos_pos.mean().detach(),
            'mean_cos_neg': cos_neg.mean().detach()
        }
        return C, loss, stats


class NegExclusiveSimple(nn.Module):
    """
    Unchanged from original.
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
        C = l2norm(C_rows, -1)

        sim = torch.einsum('bkd,brd->bkr', Qn, C).amax(dim=-1)

        outputs = {}
        if self.mode in ('residual', 'both'):
            w = torch.einsum('bkd,brd->bkr', Qn, C)
            proj = torch.einsum('bkr,brd->bkd', w, C)
            neg_resid = l2norm(Qn - proj, -1)
            outputs['residual'] = neg_resid

        if self.mode in ('filter', 'both'):
            excl_score = 1.0 - sim
            if self.thresh is not None:
                mask = (sim < self.thresh).float()
                excl_score = excl_score * mask + (-1e4) * (1 - mask)
            M = min(self.M, K)
            topv, topi = torch.topk(excl_score, k=M, dim=1)
            neg_top = torch.gather(Qn, 1, topi.unsqueeze(-1).expand(-1, -1, D))
            outputs['filtered'] = neg_top

        if self.mode == 'residual':
            neg_refs = outputs['residual']
        elif self.mode == 'filter':
            neg_refs = outputs['filtered']
        else:
            R = outputs['residual']
            excl_score = 1.0 - sim
            M = min(self.M, K)
            topv, topi = torch.topk(excl_score, k=M, dim=1)
            neg_refs = torch.gather(R, 1, topi.unsqueeze(-1).expand(-1, -1, D))

        aux = {
            'mean_sim_to_common': sim.mean().detach(),
            'kept_M': neg_refs.size(1)
        }
        return neg_refs, aux


class FusionAdaGate(nn.Module):
    """
    S3: Adaptive MLP Gating

    Use a small MLP to predict per-query adaptive gate based on [Q_pos, Z].
    The MLP learns when to suppress more and when to suppress less.

    Q_new = Q_pos - scale * gate * Z
    where gate = MLP([Q_pos; Z])
    """
    def __init__(self, d_model=256, nhead=4, init_scale=0.25, dropout_p=0.1, hidden_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln_z = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        # Adaptive gate MLP: predicts per-query gate from [Q_pos, Z]
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Initialize to output ~0.5 (neutral gate)
        self._init_gate_net()

    def _init_gate_net(self):
        """Initialize gate network to output values around 0.5"""
        for m in self.gate_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, Q_pos: torch.Tensor, neg_ref: torch.Tensor):
        """
        Q_pos:   [B, K, D]
        neg_ref: [B, M, D]
        return:  Q_new [B, K, D], stats dict
        """
        B, K, D = Q_pos.shape
        M = neg_ref.size(1)
        if M == 0:
            return Q_pos, {'kept': 0, 'gate_mean': 0.0, 'fusion_scale': self.scale.detach()}

        # Cross-attention
        Z, attn_w = self.attn(query=Q_pos, key=neg_ref, value=neg_ref)
        Z = self.ln_z(Z)
        Z = self.drop(Z)

        # Adaptive gate from MLP
        gate_input = torch.cat([Q_pos, Z], dim=-1)  # [B, K, 2D]
        gate = self.gate_net(gate_input)  # [B, K, 1]

        # Apply gated suppression
        Q_new = Q_pos - self.scale * gate * Z

        stats = {
            'kept': M,
            'gate_mean': gate.mean().detach(),
            'gate_std': gate.std().detach(),
            'gate_min': gate.min().detach(),
            'gate_max': gate.max().detach(),
            'fusion_scale': self.scale.detach(),
        }
        return Q_new, stats


class QuerySideNegAdaGate(nn.Module):
    """
    Query-side negative fusion pipeline with Adaptive MLP Gating.
    """
    def __init__(self, d_model=256, r=64, M=64, nhead=4,
                 excl_mode='both', excl_thresh=0.5,
                 share_w=0.02, div_w=0.02):
        super().__init__()
        self.common = CommonFinderSimple(d_model, r, nhead, share_w, div_w)
        self.excl = NegExclusiveSimple(mode=excl_mode, M=M, thresh=excl_thresh)
        self.fuse = FusionAdaGate(d_model=d_model, nhead=4, init_scale=0.25, dropout_p=0.1)

    def forward(self, Q_pos: torch.Tensor, Q_neg: torch.Tensor):
        C_rows, l_common, common_stats = self.common(Q_pos, Q_neg)
        neg_refs, excl_stats = self.excl(Q_neg, C_rows)
        Q_new, fuse_stats = self.fuse(Q_pos, neg_refs)
        loss = l_common
        stats = {}
        stats.update(common_stats)
        stats.update(excl_stats)
        stats.update(fuse_stats)
        return Q_new, loss, stats

    def set_fusion_scale(self, scale: float):
        self.fuse.scale = nn.Parameter(torch.tensor(scale))


class CountEXAdaGate(GroundingDinoForObjectDetection):
    """
    CountEX with Adaptive MLP Gating (S1 + S3)

    Changes:
    1. S1: Density Head uses [pos, neg, relu(pos-neg)] features (768 channels)
    2. S3: Fusion uses MLP-based per-query adaptive gating
    """

    def __init__(self, config):
        super().__init__(config)

        self.query_side_neg_pipeline = QuerySideNegAdaGate()
        self.density_head = DensityFPNHead(in_channels=768)  # S1: 512 -> 768
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
        neg_pixel_values: Optional[torch.FloatTensor] = None,
        neg_input_ids: Optional[torch.LongTensor] = None,
        neg_token_type_ids: Optional[torch.LongTensor] = None,
        neg_attention_mask: Optional[torch.LongTensor] = None,
        neg_pixel_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_neg = kwargs.get('use_neg', True)

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

        neg_kwargs = {
            'exemplars': kwargs.get('neg_exemplars', None),
        }
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
        neg_inter_references_points = neg_outputs.intermediate_reference_points if return_dict else neg_outputs[3]
        neg_init_reference_points = neg_outputs.init_reference_points if return_dict else neg_outputs[1]
        neg_enc_text_hidden_state = neg_outputs.encoder_last_hidden_state_text if return_dict else neg_outputs[idx]

        pos_exemplars = pos_kwargs.get('pos_exemplars', None)
        neg_exemplars = neg_kwargs.get('neg_exemplars', None)
        if pos_exemplars is not None or neg_exemplars is not None or attention_mask.shape[1] != enc_text_hidden_state.shape[1]:
            enc_text_hidden_state = enc_text_hidden_state[:, :enc_text_hidden_state.shape[1] - 3, :]
            neg_enc_text_hidden_state = neg_enc_text_hidden_state[:, :neg_enc_text_hidden_state.shape[1] - 3, :]

        outputs_classes = []
        outputs_coords = []

        if use_neg:
            hidden_states = hidden_states.squeeze(0)
            neg_hidden_states = neg_hidden_states.squeeze(0)
            hidden_states, extra_loss, logs = self.query_side_neg_pipeline(hidden_states, neg_hidden_states)
            hidden_states = hidden_states.unsqueeze(0)
            neg_hidden_states = neg_hidden_states.unsqueeze(0)
        else:
            extra_loss = None
            logs = None

        num_levels = hidden_states.shape[1]
        for level in range(num_levels):
            if level == 0:
                reference = init_reference_points
            else:
                reference = inter_references_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)

            assert attention_mask.shape[1] == enc_text_hidden_state.shape[1]
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

        neg_outputs_classes = []
        neg_outputs_coords = []
        for level in range(num_levels):
            if level == 0:
                neg_reference = neg_init_reference_points
            else:
                neg_reference = neg_inter_references_points[:, level - 1]
            neg_reference = torch.special.logit(neg_reference, eps=1e-5)

            neg_outputs_class = self.class_embed[level](
                vision_hidden_state=neg_hidden_states[:, level],
                text_hidden_state=neg_enc_text_hidden_state,
                text_token_mask=neg_attention_mask.bool(),
            )
            neg_delta_bbox = self.bbox_embed[level](neg_hidden_states[:, level])

            neg_reference_coordinates = neg_reference.shape[-1]
            if neg_reference_coordinates == 4:
                neg_outputs_coord_logits = neg_delta_bbox + neg_reference
            elif neg_reference_coordinates == 2:
                neg_delta_bbox[..., :2] += neg_reference
                neg_outputs_coord_logits = neg_delta_bbox
            else:
                raise ValueError(f"neg_reference.shape[-1] should be 4 or 2, but got {neg_reference.shape[-1]}")
            neg_outputs_coord = neg_outputs_coord_logits.sigmoid()
            neg_outputs_classes.append(neg_outputs_class)
            neg_outputs_coords.append(neg_outputs_coord)
        neg_outputs_class = torch.stack(neg_outputs_classes)
        neg_outputs_coord = torch.stack(neg_outputs_coords)

        neg_logits = neg_outputs_class[-1]
        neg_pred_boxes = neg_outputs_coord[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            tuple_outputs = ((loss, loss_dict) + output) if loss is not None else output
            return tuple_outputs

        # S1: Add pos-neg difference feature
        all_feats = []
        for pf, npf in zip(positive_feature_maps, neg_positive_feature_maps):
            pf = pf.permute(0, 3, 1, 2)
            npf = npf.permute(0, 3, 1, 2)
            diff = F.relu(pf - npf)  # S1: pos-neg difference
            all_feats.append(torch.cat([pf, npf, diff], dim=1))  # 768 channels

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
            neg_logits=neg_logits,
            neg_pred_boxes=neg_pred_boxes,
            pos_queries=hidden_states,
            neg_queries=neg_hidden_states,
        )

        return dict_outputs
