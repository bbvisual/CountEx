"""
CountEXWithDetectionPrior: Weakly-supervised counting with detection prior as supervision.

Key idea:
- Use frozen detection model to generate pseudo density labels (detection prior)
- Train only the density head to regress this prior
- Detection prior provides localization signal for weakly-supervised learning

Architecture:
    Input Image + Text
           ↓
    Frozen Backbone (Swin)
           ↓
    Frozen Encoder
           ↓
    Frozen Decoder → Detection Prior (supervision only)
           ↓
    Trainable Density Head
           ↓
    density_map = raw_density (detection prior NOT added to output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple

from transformers.utils import ModelOutput
from .CountEX import CountEX


@dataclass
class DensityWithPriorOutput(ModelOutput):
    """Output for CountEXWithDetectionPrior model."""
    density_map: torch.FloatTensor = None          # Final density map (= raw_density)
    density_log_var: Optional[torch.FloatTensor] = None
    pred_count: torch.FloatTensor = None           # Predicted count
    detection_prior: torch.FloatTensor = None      # Detection prior (for supervision)
    raw_density: torch.FloatTensor = None          # Raw density head output
    pred_boxes: torch.FloatTensor = None           # Detection boxes
    pred_scores: torch.FloatTensor = None          # Detection scores
    pos_features: Optional[List[torch.FloatTensor]] = None
    neg_features: Optional[List[torch.FloatTensor]] = None
    logits: Optional[torch.FloatTensor] = None


def create_soft_density_from_logits(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    height: int,
    width: int,
    temperature: float = 1.0,
    sigma: float = 3.0,
) -> torch.Tensor:
    """
    Create soft density map from detection logits and boxes.
    Places Gaussian blobs at box centers, weighted by detection confidence.

    Args:
        logits: [B, N, num_classes] detection logits
        boxes: [B, N, 4] boxes in cxcywh normalized format
        height: output height
        width: output width
        temperature: temperature for softmax (lower = sharper)
        sigma: Gaussian sigma in pixels

    Returns:
        density: [B, 1, H, W] soft density map
    """
    B, N, _ = logits.shape
    device = logits.device
    dtype = logits.dtype

    # Get confidence scores: max over classes, apply temperature
    scores = (logits / temperature).sigmoid().max(dim=-1)[0]  # [B, N]

    # Get box centers
    cx = boxes[:, :, 0]  # [B, N] normalized 0-1
    cy = boxes[:, :, 1]  # [B, N] normalized 0-1

    # Create coordinate grids
    y_coords = torch.arange(height, device=device, dtype=dtype)
    x_coords = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W]

    # Initialize density
    density = torch.zeros(B, 1, height, width, device=device, dtype=dtype)

    # For each query, add a Gaussian blob weighted by score
    for b in range(B):
        for n in range(N):
            score = scores[b, n]
            if score < 0.01:  # Skip very low confidence detections
                continue

            cx_n = cx[b, n] * width
            cy_n = cy[b, n] * height

            # Compute Gaussian centered at (cx_n, cy_n)
            dist_sq = (xx - cx_n) ** 2 + (yy - cy_n) ** 2
            gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))

            # Normalize to sum to 1, then weight by score
            gaussian = gaussian / (gaussian.sum() + 1e-6)
            gaussian = gaussian * score

            density[b, 0] += gaussian

    return density


class CountEXWithDetectionPrior(CountEX):
    """
    CountEX with Detection Prior for weakly-supervised counting.

    Uses frozen detection model to generate pseudo density labels.
    Only the density head is trainable.

    Training:
        - count_loss: |pred_count - gt_count|
        - pseudo_density_loss: MSE(raw_density, normalized_detection_prior)

    Inference:
        - density_map = raw_density (detection prior not used)
    """

    def __init__(self, config):
        super().__init__(config)

        # Temperature for detection score computation
        self.prior_temperature = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        self.config = config

        # Freeze everything except density head
        self._freeze_for_density_training()
        self._print_model_info()

    def _freeze_for_density_training(self):
        """
        Freeze backbone, encoder, decoder to keep detection prior stable.
        Only density_head remains trainable.
        """
        # Freeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Freeze encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # Freeze decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = False

        # Freeze detection heads
        for param in self.bbox_embed.parameters():
            param.requires_grad = False
        for param in self.class_embed.parameters():
            param.requires_grad = False

        # Remove unused module
        if hasattr(self, 'query_side_neg_pipeline'):
            del self.query_side_neg_pipeline

        # Freeze query position embeddings
        if hasattr(self.model, 'query_position_embeddings'):
            self.model.query_position_embeddings.requires_grad = False

        # Freeze input projections
        if hasattr(self.model, 'input_proj_vision'):
            for proj in self.model.input_proj_vision:
                for param in proj.parameters():
                    param.requires_grad = False
        if hasattr(self.model, 'input_proj_text'):
            for param in self.model.input_proj_text.parameters():
                param.requires_grad = False

    def _print_model_info(self):
        """Print model information."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print("=" * 60)
        print("CountEXWithDetectionPrior")
        print("=" * 60)
        print("  Frozen: backbone, encoder, decoder")
        print("  Trainable: density_head only")
        print(f"  prior_temperature: {self.prior_temperature[0].item():.3f}")
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
        Forward pass.

        Returns:
            DensityWithPriorOutput containing:
            - density_map: raw_density (for prediction)
            - detection_prior: for pseudo density loss computation
            - pred_count: density_map.sum()
        """
        # Force use_neg=False (we don't use negative features)
        kwargs['use_neg'] = False

        # Don't use exemplars
        kwargs['pos_exemplars'] = None
        kwargs['neg_exemplars'] = None

        # Call parent forward to get density and detection outputs
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

        # Get raw density from density head
        raw_density = outputs.density_map_pred  # [B, 1, H, W]

        # Get detection outputs
        logits = outputs.logits  # [B, N, num_classes]
        pred_boxes = outputs.pred_boxes  # [B, N, 4]

        # Create detection prior from frozen detector
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

        # Output is just raw_density (detection_prior only for supervision)
        detection_prior = detection_prior.to(raw_density.dtype)
        density_map = raw_density

        pred_count = density_map.sum(dim=(1, 2, 3))
        pred_scores = logits.sigmoid().max(dim=-1)[0]

        return DensityWithPriorOutput(
            density_map=density_map,
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
