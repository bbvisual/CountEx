"""
CountEXStage2: Finetune a dot-pretrained CountEX with detection prior.

Key idea:
- Load a CountEX model pretrained with dot annotations
- Add detection prior generation from decoder outputs
- Everything remains trainable (finetune all parameters)
- Detection prior provides additional spatial supervision for weakly-supervised learning

Difference from CountEXWithDetectionPrior:
- CountEXWithDetectionPrior: freezes backbone/encoder/decoder, only trains density_head
- CountEXStage2: everything trainable, finetune from dot-pretrained model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple

from transformers.utils import ModelOutput
from .CountEX import CountEX, DensityFPNHead
from .CountEXWithDetectionPrior import create_soft_density_from_logits, DensityWithPriorOutput


class CountEXStage2(CountEX):
    """
    CountEX with detection prior for stage-2 finetuning.

    Loads from a dot-pretrained CountEX checkpoint. All parameters
    remain trainable. Detection prior is generated from decoder outputs
    to provide additional spatial supervision.
    """

    def __init__(self, config):
        super().__init__(config)

        # Temperature for detection prior score computation
        self.prior_temperature = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        # Tri-branch density head: [pos, neg, relu(pos - neg)] -> 768 channels
        # self.diff_density_head = DensityFPNHead(in_channels=768)
        # Freeze parent's density_head (superseded by diff_density_head)
        # for param in self.density_head.parameters():
        #     param.requires_grad = False
        self.config = config
        self._print_model_info()

    def _print_model_info(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print("=" * 60)
        print("CountEXStage2 - Finetune from dot-pretrained model")
        print("=" * 60)
        print("  All modules trainable (finetune mode)")
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print("=" * 60)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> DensityWithPriorOutput:
        """
        Forward: run full CountEX, then generate detection prior from decoder outputs.
        """
        # Call parent CountEX forward
        outputs = super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        # # Recompute density using tri-branch head: [pos, neg, relu(pos - neg)]
        # all_feats = []
        # for pf, npf in zip(outputs.positive_feature_maps, outputs.negative_feature_maps):
        #     pf = pf.permute(0, 3, 1, 2)    # [B, D, H, W]
        #     npf = npf.permute(0, 3, 1, 2)
        #     pos_minus_neg = F.relu(pf - npf)
        #     all_feats.append(torch.cat([pf, npf, pos_minus_neg], dim=1))  # [B, 3D, H, W]
        # raw_density = self.diff_density_head(all_feats)  # [B, 1, H, W]

        # Get detection outputs
        raw_density = outputs.density_map_pred  # [B, 1, H, W]
        logits = outputs.logits  # [B, N, num_classes]
        pred_boxes = outputs.pred_boxes  # [B, N, 4]

        # Generate detection prior from decoder outputs
        H, W = raw_density.shape[-2:]
        temp = F.softplus(self.prior_temperature[0]) + 0.1

        with torch.no_grad():
            detection_prior = create_soft_density_from_logits(
                logits=logits.detach(),
                boxes=pred_boxes.detach(),
                height=H,
                width=W,
                temperature=temp,
            )
        # print("detection_prior: ", detection_prior.shape)

        detection_prior = detection_prior.to(raw_density.dtype)
        density_map = raw_density
        pred_count = density_map.sum(dim=(1, 2, 3))
        pred_scores = logits.sigmoid().max(dim=-1)[0]

        return DensityWithPriorOutput(
            density_map=outputs.density_map_pred,
            density_log_var=None,
            pred_count=pred_count,
            detection_prior=detection_prior,
            raw_density=raw_density,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pos_features=outputs.positive_feature_maps,
            neg_features=outputs.negative_feature_maps,
            logits=logits,
        )
