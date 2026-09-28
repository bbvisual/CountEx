# coding=utf-8
"""
CountEXDensityOnly: Simplified model for weakly-supervised counting.

Only encoder + density head, NO detection branch.
Unified architecture for counting and regression tasks.

Key features:
1. Vision-language encoder from GroundingDINO
2. Negative prompt guidance: relu(pos - neg)
3. Density head with optional uncertainty output
4. Count prediction via density.sum()
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch.nn.functional as F
from transformers import GroundingDinoConfig
from .modeling_grounding_dino import (
    GroundingDinoModel,
    GroundingDinoPreTrainedModel,
    GroundingDinoForObjectDetection,
)
from .CountEX import DensityFPNHead as CountEXDensityFPNHead, CountEX


def _bilinear(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


@dataclass
class DensityOnlyOutput:
    """Output class for CountEXDensityOnly model."""
    density_map: torch.FloatTensor
    density_log_var: Optional[torch.FloatTensor] = None
    pred_count: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    # Feature maps for visualization/analysis
    pos_features: Optional[List[torch.FloatTensor]] = None
    neg_features: Optional[List[torch.FloatTensor]] = None


class DensityFPNHead(nn.Module):
    """
    FPN-based density head with optional uncertainty output.

    Input: Multi-scale feature maps [P3, P4, P5, P6]
    Output: density map, optional log_variance
    """
    def __init__(self,
                 in_channels: int = 256,
                 mid_channels: int = 64,
                 num_scales: int = 4,
                 num_up_blocks: int = 3,
                 with_uncertainty: bool = True,
                 act_layer=nn.ReLU,
                 norm_layer=nn.GroupNorm):  # Use GroupNorm instead of BatchNorm for better stability
        super().__init__()
        self.with_uncertainty = with_uncertainty
        self.num_scales = num_scales

        # Lateral 1x1 convs
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_channels, mid_channels, 1) for _ in range(num_scales)
        ])

        # Smooth convs for FPN (using GroupNorm with 8 groups)
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                norm_layer(8, mid_channels),  # GroupNorm(num_groups, num_channels)
                act_layer(inplace=True),
            ) for _ in range(num_scales - 1)
        ])

        # Upsample blocks (removed leading ReLU, use Conv + GN + ReLU)
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                norm_layer(8, mid_channels),
                act_layer(inplace=True),
            ) for _ in range(num_up_blocks)
        ])

        # Output heads (add bias for density conv to allow positive baseline)
        self.density_conv = nn.Conv2d(mid_channels, 1, 3, padding=1, bias=True)

        if with_uncertainty:
            self.logvar_conv = nn.Conv2d(mid_channels, 1, 3, padding=1, bias=True)
            # Initialize log_var to output small values (low uncertainty initially)
            nn.init.zeros_(self.logvar_conv.weight)
            nn.init.constant_(self.logvar_conv.bias, -2.0)  # exp(-2) ≈ 0.14

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming initialization for ReLU activation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Special initialization for density_conv: zero bias for proper count scale
        # With bias=0.01 on 512x512 grid, sum would be ~2600 which is way too high
        nn.init.kaiming_normal_(self.density_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.density_conv.bias)

        # Re-initialize logvar_conv if present
        if self.with_uncertainty:
            nn.init.zeros_(self.logvar_conv.weight)
            nn.init.constant_(self.logvar_conv.bias, -2.0)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert len(feats) == self.num_scales, f"Expected {self.num_scales} feature maps, got {len(feats)}"

        # Lateral 1x1 convs
        lat = [l(f) for l, f in zip(self.lateral, feats)]

        # Top-down FPN fusion
        x = lat[-1]
        for i in range(self.num_scales - 2, -1, -1):
            x = _bilinear(x, lat[i].shape[-2:])
            x = x + lat[i]
            x = self.smooth[i](x)

        # Upsample blocks
        for up in self.up_blocks:
            h, w = x.shape[-2], x.shape[-1]
            x = _bilinear(x, (h * 2, w * 2))
            x = up(x)

        # Output
        density = F.relu(self.density_conv(x))

        if self.with_uncertainty:
            log_var = self.logvar_conv(x)
            return density, log_var
        else:
            return density, None


class CountEXDensityOnly(GroundingDinoPreTrainedModel):
    """
    Simplified counting model with only encoder + density head.

    Architecture:
        Image + Pos Prompt → Encoder → Pos Features ─┐
                                                      ├→ relu(pos-neg) → Density Head → Count
        Image + Neg Prompt → Encoder → Neg Features ─┘

    No detection branch - pure density-based counting.
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        # Vision-language encoder (from GroundingDINO)
        self.model = GroundingDinoModel(config)

        # Density head
        self.density_head = DensityFPNHead(
            in_channels=256,
            mid_channels=64,
            num_scales=4,
            num_up_blocks=3,
            with_uncertainty=True,
        )

        self.config = config

        # Initialize weights
        self.post_init()

    def _encode(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        exemplars: Optional[torch.FloatTensor] = None,
    ) -> List[torch.Tensor]:
        """
        Encode image with text prompt, return multi-scale feature maps.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            return_dict=True,
            exemplars=exemplars,
        )

        # Extract multi-scale feature maps from encoder
        spatial_shapes = outputs.spatial_shapes
        encoder_vision = outputs.encoder_last_hidden_state_vision

        # Split into multi-scale feature maps
        token_num = 0
        token_num_list = [0]
        for i in range(len(spatial_shapes)):
            token_num += spatial_shapes[i][0] * spatial_shapes[i][1]
            token_num_list.append(token_num.item())

        feature_maps = []
        for i in range(len(spatial_shapes)):
            feat = encoder_vision[:, token_num_list[i]:token_num_list[i+1], :]
            h, w = spatial_shapes[i]
            b, t, d = feat.shape
            feat = feat.reshape(b, h, w, d).permute(0, 3, 1, 2)  # [B, D, H, W]
            feature_maps.append(feat)

        return feature_maps

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        # Negative prompt inputs
        neg_pixel_values: Optional[torch.FloatTensor] = None,
        neg_input_ids: Optional[torch.LongTensor] = None,
        neg_token_type_ids: Optional[torch.LongTensor] = None,
        neg_attention_mask: Optional[torch.LongTensor] = None,
        neg_pixel_mask: Optional[torch.BoolTensor] = None,
        # Exemplars (optional)
        pos_exemplars: Optional[torch.FloatTensor] = None,
        neg_exemplars: Optional[torch.FloatTensor] = None,
        # Control flags
        use_neg: bool = True,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[DensityOnlyOutput, Tuple]:
        """
        Forward pass for density-based counting.

        Args:
            pixel_values: Input image [B, 3, H, W]
            input_ids: Positive prompt tokens
            neg_*: Negative prompt inputs (optional)
            use_neg: Whether to use negative prompt guidance

        Returns:
            DensityOnlyOutput with density_map, log_var, pred_count
        """
        # Encode positive prompt
        pos_features = self._encode(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            exemplars=pos_exemplars,
        )

        # Encode negative prompt (if provided and enabled)
        if use_neg and neg_input_ids is not None:
            neg_features = self._encode(
                pixel_values=neg_pixel_values if neg_pixel_values is not None else pixel_values,
                input_ids=neg_input_ids,
                token_type_ids=neg_token_type_ids,
                attention_mask=neg_attention_mask,
                pixel_mask=neg_pixel_mask if neg_pixel_mask is not None else pixel_mask,
                exemplars=neg_exemplars,
            )

            # Compute difference features: relu(pos - neg)
            diff_features = []
            for pf, nf in zip(pos_features, neg_features):
                diff = F.relu(pf - nf)
                diff_features.append(diff)
        else:
            diff_features = pos_features
            neg_features = None

        # Density prediction
        density_map, density_log_var = self.density_head(diff_features)

        # Compute predicted count
        pred_count = density_map.sum(dim=(1, 2, 3))  # [B]

        if not return_dict:
            return (density_map, density_log_var, pred_count)

        return DensityOnlyOutput(
            density_map=density_map,
            density_log_var=density_log_var,
            pred_count=pred_count,
            pos_features=pos_features,
            neg_features=neg_features,
        )


class CountEXDensityOnlyNoUncertainty(CountEXDensityOnly):
    """
    Same as CountEXDensityOnly but without uncertainty output.
    Lighter weight for basic experiments.
    """

    def __init__(self, config: GroundingDinoConfig):
        super(GroundingDinoPreTrainedModel, self).__init__(config)

        self.model = GroundingDinoModel(config)

        self.density_head = DensityFPNHead(
            in_channels=256,
            mid_channels=64,
            num_scales=4,
            num_up_blocks=3,
            with_uncertainty=False,  # No uncertainty
        )

        self.config = config
        self.post_init()


class SimpleDensityHead(nn.Module):
    """
    Simple density head that directly projects diff_features to density.

    Key insight: diff_features already have spatial structure from relu(pos-neg).
    This head preserves that structure by using minimal transformations:
    1. Learned channel weights (which features matter for density)
    2. Spatial smoothing (to reduce noise)
    3. Scale factor (to match count magnitude)

    This prevents the model from learning to output uniform density.
    """
    def __init__(self, in_channels: int = 256, with_uncertainty: bool = True):
        super().__init__()
        self.with_uncertainty = with_uncertainty

        # Channel attention: learn which feature channels contribute to density
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid(),
        )

        # Simple spatial conv (3x3) to smooth features
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=True),
        )

        if with_uncertainty:
            self.logvar_conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1, bias=True),
            )
            # Initialize to low uncertainty
            nn.init.zeros_(self.logvar_conv[-1].weight)
            nn.init.constant_(self.logvar_conv[-1].bias, -2.0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small positive bias for density output
        nn.init.constant_(self.spatial_conv[-1].bias, 0.01)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Use only highest resolution feature (P3, which is H/8 x W/8)
        feat = feats[0]  # [B, 256, H/8, W/8]

        # Apply channel attention
        attn = self.channel_attention(feat)  # [B, 256, 1, 1]
        feat = feat * attn  # Weighted features

        # Spatial convolution to get density
        density = self.spatial_conv(feat)  # [B, 1, H/8, W/8]
        density = F.relu(density)

        # Upsample to higher resolution (8x)
        density = F.interpolate(density, scale_factor=8, mode='bilinear', align_corners=False)

        if self.with_uncertainty:
            log_var = self.logvar_conv(feats[0])
            log_var = F.interpolate(log_var, scale_factor=8, mode='bilinear', align_corners=False)
            return density, log_var
        else:
            return density, None


class CountEXDensityOnlyDirect(GroundingDinoPreTrainedModel):
    """
    Simplified density model with direct feature-to-density projection.

    Uses SimpleDensityHead instead of FPN to preserve spatial structure from diff_features.
    This prevents the model from learning trivial uniform density solutions.
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.model = GroundingDinoModel(config)

        # Use simple density head instead of FPN
        self.density_head = SimpleDensityHead(
            in_channels=256,
            with_uncertainty=True,
        )

        self.config = config
        self.post_init()

    def _encode(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        exemplars: Optional[torch.FloatTensor] = None,
    ) -> List[torch.Tensor]:
        """Same as parent class."""
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            return_dict=True,
            exemplars=exemplars,
        )

        spatial_shapes = outputs.spatial_shapes
        encoder_vision = outputs.encoder_last_hidden_state_vision

        token_num = 0
        token_num_list = [0]
        for i in range(len(spatial_shapes)):
            token_num += spatial_shapes[i][0] * spatial_shapes[i][1]
            token_num_list.append(token_num.item())

        feature_maps = []
        for i in range(len(spatial_shapes)):
            feat = encoder_vision[:, token_num_list[i]:token_num_list[i+1], :]
            h, w = spatial_shapes[i]
            b, t, d = feat.shape
            feat = feat.reshape(b, h, w, d).permute(0, 3, 1, 2)
            feature_maps.append(feat)

        return feature_maps

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        neg_pixel_values: Optional[torch.FloatTensor] = None,
        neg_input_ids: Optional[torch.LongTensor] = None,
        neg_token_type_ids: Optional[torch.LongTensor] = None,
        neg_attention_mask: Optional[torch.LongTensor] = None,
        neg_pixel_mask: Optional[torch.BoolTensor] = None,
        pos_exemplars: Optional[torch.FloatTensor] = None,
        neg_exemplars: Optional[torch.FloatTensor] = None,
        use_neg: bool = True,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[DensityOnlyOutput, Tuple]:
        """Same forward logic as parent, but uses SimpleDensityHead."""
        pos_features = self._encode(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            exemplars=pos_exemplars,
        )

        if use_neg and neg_input_ids is not None:
            neg_features = self._encode(
                pixel_values=neg_pixel_values if neg_pixel_values is not None else pixel_values,
                input_ids=neg_input_ids,
                token_type_ids=neg_token_type_ids,
                attention_mask=neg_attention_mask,
                pixel_mask=neg_pixel_mask if neg_pixel_mask is not None else pixel_mask,
                exemplars=neg_exemplars,
            )

            diff_features = []
            for pf, nf in zip(pos_features, neg_features):
                diff = F.relu(pf - nf)
                diff_features.append(diff)
        else:
            diff_features = pos_features
            neg_features = None

        density_map, density_log_var = self.density_head(diff_features)
        pred_count = density_map.sum(dim=(1, 2, 3))

        if not return_dict:
            return (density_map, density_log_var, pred_count)

        return DensityOnlyOutput(
            density_map=density_map,
            density_log_var=density_log_var,
            pred_count=pred_count,
            pos_features=pos_features,
            neg_features=neg_features,
        )


@dataclass
class DensityWithPriorOutput:
    """Output class for CountEXDensityOnlyWithDetectionPrior model."""
    density_map: torch.FloatTensor
    density_log_var: Optional[torch.FloatTensor] = None
    pred_count: Optional[torch.FloatTensor] = None
    detection_prior: Optional[torch.FloatTensor] = None  # Prior from detections
    raw_density: Optional[torch.FloatTensor] = None  # Density before fusion (for debugging)
    pred_boxes: Optional[torch.FloatTensor] = None  # Detection boxes
    pred_scores: Optional[torch.FloatTensor] = None  # Detection scores
    loss: Optional[torch.FloatTensor] = None
    pos_features: Optional[List[torch.FloatTensor]] = None
    neg_features: Optional[List[torch.FloatTensor]] = None


def create_gaussian_density_from_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    height: int,
    width: int,
    score_threshold: float = 0.3,
    sigma_factor: float = 0.25,
) -> torch.Tensor:
    """
    Create a density map prior from detection boxes.

    Each detection creates a normalized Gaussian blob that sums to 1,
    then weighted by the detection score. This ensures that
    detection_prior.sum() ≈ sum of confident detection scores ≈ expected count.

    Args:
        boxes: [B, N, 4] normalized boxes in cxcywh format
        scores: [B, N] confidence scores
        height, width: output density map size
        score_threshold: only use boxes with score above this
        sigma_factor: Gaussian sigma = min(w, h) * sigma_factor

    Returns:
        density_prior: [B, 1, H, W] density map from detections
    """
    B, N, _ = boxes.shape
    device = boxes.device
    dtype = boxes.dtype

    # Create coordinate grids
    y_coords = torch.linspace(0, 1, height, device=device, dtype=dtype)
    x_coords = torch.linspace(0, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W]

    density_prior = torch.zeros(B, 1, height, width, device=device, dtype=dtype)

    for b in range(B):
        for n in range(N):
            score = scores[b, n]
            if score < score_threshold:
                continue

            cx, cy, w, h = boxes[b, n]
            # Gaussian sigma based on box size
            sigma = max(w, h) * sigma_factor
            sigma = max(sigma, 0.01)  # Minimum sigma

            # Gaussian blob
            gaussian = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

            # IMPORTANT: Normalize Gaussian to sum to 1, then weight by score
            # This ensures detection_prior.sum() ≈ number of detections (weighted by confidence)
            gaussian_sum = gaussian.sum()
            if gaussian_sum > 1e-6:
                gaussian = gaussian / gaussian_sum  # Now sums to 1

            gaussian = gaussian * score  # Weight by confidence (now contributes ~score to total count)

            density_prior[b, 0] += gaussian

    return density_prior


class LargeDensityFPNHead(nn.Module):
    """
    Larger FPN-based density head for when encoder is frozen.
    More capacity to learn the density mapping.
    """
    def __init__(self,
                 in_channels: int = 256,
                 mid_channels: int = 128,  # Larger than standard
                 num_scales: int = 4,
                 num_up_blocks: int = 4,  # More upsampling blocks
                 with_uncertainty: bool = True,
                 act_layer=nn.ReLU,
                 norm_layer=nn.GroupNorm):
        super().__init__()
        self.with_uncertainty = with_uncertainty
        self.num_scales = num_scales

        # Lateral 1x1 convs
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1),
                norm_layer(8, mid_channels),
                act_layer(inplace=True),
            ) for _ in range(num_scales)
        ])

        # Smooth convs for FPN with more layers
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                norm_layer(8, mid_channels),
                act_layer(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                norm_layer(8, mid_channels),
                act_layer(inplace=True),
            ) for _ in range(num_scales - 1)
        ])

        # More upsample blocks with residual connections
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                norm_layer(8, mid_channels),
                act_layer(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                norm_layer(8, mid_channels),
                act_layer(inplace=True),
            ) for _ in range(num_up_blocks)
        ])

        # Output head with more layers
        self.density_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            norm_layer(8, mid_channels // 2),
            act_layer(inplace=True),
            nn.Conv2d(mid_channels // 2, 1, 1, bias=True),
        )

        if with_uncertainty:
            self.logvar_conv = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
                norm_layer(8, mid_channels // 2),
                act_layer(inplace=True),
                nn.Conv2d(mid_channels // 2, 1, 1, bias=True),
            )
            nn.init.zeros_(self.logvar_conv[-1].weight)
            nn.init.constant_(self.logvar_conv[-1].bias, -2.0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # IMPORTANT: Set bias to 0 for proper count scale
        # With bias=0.01 on 512x512 grid, sum would be ~2600 which is way too high
        # For typical counts of 10-100, we want density.sum() to be in that range
        nn.init.zeros_(self.density_conv[-1].bias)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert len(feats) == self.num_scales

        # Lateral convs
        lat = [l(f) for l, f in zip(self.lateral, feats)]

        # Top-down FPN fusion
        x = lat[-1]
        for i in range(self.num_scales - 2, -1, -1):
            x = _bilinear(x, lat[i].shape[-2:])
            x = x + lat[i]
            x = self.smooth[i](x)

        # Upsample blocks
        for up in self.up_blocks:
            h, w = x.shape[-2], x.shape[-1]
            x = _bilinear(x, (h * 2, w * 2))
            x = up(x)

        density = F.relu(self.density_conv(x))

        if self.with_uncertainty:
            log_var = self.logvar_conv(x)
            return density, log_var
        else:
            return density, None


@dataclass
class LogitsDensityOutput:
    """Output class for CountEXLogitsDensity model."""
    density_map: torch.FloatTensor
    density_log_var: Optional[torch.FloatTensor] = None  # For trainer compatibility (always None)
    pred_count: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    pred_scores: Optional[torch.FloatTensor] = None
    raw_logits: Optional[torch.FloatTensor] = None  # For debugging
    loss: Optional[torch.FloatTensor] = None


def create_soft_density_from_logits(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    height: int,
    width: int,
    temperature: Union[float, torch.Tensor] = 1.0,
    sigma_factor: Union[float, torch.Tensor] = 0.25,
    min_sigma: float = 0.02,
    internal_size: int = 64,
) -> torch.Tensor:
    """
    Create pseudo density map from detection logits using box centers.

    Simple approach: place a small Gaussian dot at each box center,
    weighted by detection confidence score. This creates a pseudo
    ground truth for the density head to learn from.

    Args:
        logits: [B, N, T] raw logits or [B, N] if already reduced
        boxes: [B, N, 4] predicted boxes in cxcywh normalized format
        height, width: output density map size
        temperature: temperature for sigmoid
        sigma_factor: not used (kept for API compatibility)
        min_sigma: not used
        internal_size: not used

    Returns:
        density: [B, 1, H, W] pseudo density map where sum ≈ number of confident detections
    """
    B, N = boxes.shape[:2]
    device = boxes.device
    dtype = torch.float32

    # Ensure float32 for stable computation
    logits = logits.float()
    boxes = boxes.float()

    # Get per-query scores from logits
    if logits.dim() == 3:
        scores = (logits / temperature).sigmoid().max(dim=-1)[0]  # [B, N]
    else:
        scores = (logits / temperature).sigmoid()  # [B, N]

    # Box centers in pixel coordinates
    cx = boxes[..., 0]  # [B, N] normalized 0-1
    cy = boxes[..., 1]  # [B, N] normalized 0-1

    # Convert to pixel coordinates
    cx_pix = (cx * width).long().clamp(0, width - 1)  # [B, N]
    cy_pix = (cy * height).long().clamp(0, height - 1)  # [B, N]

    # Create density map by placing Gaussian dots at each center
    # Use a fixed small sigma for the dots (e.g., 3 pixels)
    sigma = 3.0

    # Create coordinate grids
    y_coords = torch.arange(height, device=device, dtype=dtype)
    x_coords = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W]

    # Initialize density
    density = torch.zeros(B, 1, height, width, device=device, dtype=dtype)

    # For each query, add a Gaussian blob weighted by score
    # Process in smaller batches to avoid memory issues
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


class CountEXLogitsDensity(GroundingDinoPreTrainedModel):
    """
    Density estimation purely from detection logits - no learned density head.

    Key idea:
    - Use ALL decoder queries (not just high-confidence after NMS)
    - Soft scores from logits weight Gaussian blobs
    - Learnable temperature to calibrate logit-to-density mapping

    Architecture:
        Image + Text → Encoder-Decoder → Logits + Boxes
                                              ↓
                              Soft Density from ALL queries
                                              ↓
                                     Count = density.sum()

    Minimal learnable parameters: just temperature and scale.
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        from .modeling_grounding_dino import (
            GroundingDinoContrastiveEmbedding,
            GroundingDinoMLPPredictionHead,
        )

        # Full model with decoder
        self.model = GroundingDinoModel(config)

        # Detection heads (from GroundingDinoForObjectDetection)
        _class_embed = GroundingDinoContrastiveEmbedding(config)
        if config.decoder_bbox_embed_share:
            _bbox_embed = GroundingDinoMLPPredictionHead(
                input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
            )
            self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        else:
            self.bbox_embed = nn.ModuleList([
                GroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                ) for _ in range(config.decoder_layers)
            ])
        self.class_embed = nn.ModuleList([_class_embed for _ in range(config.decoder_layers)])

        # Link to decoder for box refinement
        self.model.decoder.bbox_embed = self.bbox_embed
        self.model.decoder.class_embed = self.class_embed

        # Learnable parameters for logits-to-density mapping
        # Temperature: controls sharpness of sigmoid (higher = more uniform, lower = sharper)
        # Note: Use 1D tensors (not scalars) for better DeepSpeed/DDP compatibility
        self.temperature = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        # Scale: global scaling of density (to match count magnitude)
        self.density_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        # Sigma factor: controls Gaussian spread
        self.sigma_factor = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))

        self.config = config
        self.post_init()

        # Freeze encoder and decoder
        self._freeze_encoder_decoder()

    def _freeze_encoder_decoder(self):
        """Freeze all parameters except learnable scalars."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.bbox_embed.parameters():
            param.requires_grad = False
        for param in self.class_embed.parameters():
            param.requires_grad = False

        print("=" * 60)
        print("CountEXLogitsDensity: Frozen encoder/decoder")
        print("Trainable parameters:")
        print(f"  - temperature: {self.temperature[0].item():.3f}")
        print(f"  - density_scale: {self.density_scale[0].item():.3f}")
        print(f"  - sigma_factor: {self.sigma_factor[0].item():.3f}")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable} / {total:,} ({100*trainable/total:.4f}%)")
        print("=" * 60)

    def _get_detections(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        exemplars: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run frozen encoder-decoder and get logits + boxes.

        Returns:
            logits: [B, N, T] raw logits (before sigmoid)
            boxes: [B, N, 4] predicted boxes (cxcywh normalized)
            scores: [B, N] confidence scores (for reference)
        """
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                pixel_mask=pixel_mask,
                return_dict=True,
                exemplars=exemplars,
            )

            hidden_states = outputs.intermediate_hidden_states
            init_reference_points = outputs.init_reference_points
            inter_references_points = outputs.intermediate_reference_points
            enc_text_hidden_state = outputs.encoder_last_hidden_state_text

            # Get last layer predictions
            num_levels = hidden_states.shape[1]
            level = num_levels - 1

            reference = inter_references_points[:, level - 1] if level > 0 else init_reference_points
            reference = torch.special.logit(reference, eps=1e-5)

            B, T, _ = enc_text_hidden_state.shape
            text_token_mask = torch.ones(B, T, dtype=torch.bool, device=enc_text_hidden_state.device)

            # Raw logits (before sigmoid)
            logits = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=text_token_mask,
            )  # [B, N, T]

            # Box predictions
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            else:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            pred_boxes = outputs_coord_logits.sigmoid()  # [B, N, 4]

            # Scores for reference
            pred_scores = logits.sigmoid().max(dim=-1)[0]  # [B, N]

        return logits.detach(), pred_boxes.detach(), pred_scores.detach()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        neg_pixel_values: Optional[torch.FloatTensor] = None,
        neg_input_ids: Optional[torch.LongTensor] = None,
        neg_token_type_ids: Optional[torch.LongTensor] = None,
        neg_attention_mask: Optional[torch.LongTensor] = None,
        neg_pixel_mask: Optional[torch.BoolTensor] = None,
        pos_exemplars: Optional[torch.FloatTensor] = None,
        neg_exemplars: Optional[torch.FloatTensor] = None,
        use_neg: bool = True,
        output_size: int = 512,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[LogitsDensityOutput, Tuple]:
        """
        Forward pass: logits → soft density → count.

        If use_neg=True, computes:
            density = pos_density - neg_density (ReLU applied)
        """
        # Get positive detections
        pos_logits, pos_boxes, pos_scores = self._get_detections(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            exemplars=pos_exemplars,
        )

        # Create positive density
        # Use learnable parameters (require_grad=True for these)
        # Index [0] to get scalar from 1D tensor
        temp = F.softplus(self.temperature[0]) + 0.1  # Ensure positive, min 0.1
        scale = F.softplus(self.density_scale[0])  # Ensure positive
        sigma_f = torch.sigmoid(self.sigma_factor[0]) * 0.5 + 0.1  # Range [0.1, 0.6]

        pos_density = create_soft_density_from_logits(
            logits=pos_logits,
            boxes=pos_boxes,
            height=output_size,
            width=output_size,
            temperature=temp,
            sigma_factor=sigma_f,
        )

        # Process negative prompt
        if use_neg and neg_input_ids is not None:
            neg_logits, neg_boxes, neg_scores = self._get_detections(
                pixel_values=neg_pixel_values if neg_pixel_values is not None else pixel_values,
                input_ids=neg_input_ids,
                token_type_ids=neg_token_type_ids,
                attention_mask=neg_attention_mask,
                pixel_mask=neg_pixel_mask if neg_pixel_mask is not None else pixel_mask,
                exemplars=neg_exemplars,
            )

            neg_density = create_soft_density_from_logits(
                logits=neg_logits,
                boxes=neg_boxes,
                height=output_size,
                width=output_size,
                temperature=temp,
                sigma_factor=sigma_f,
            )

            # Subtract negative density
            density_map = F.relu(pos_density - neg_density) * scale
        else:
            density_map = pos_density * scale

        pred_count = density_map.sum(dim=(1, 2, 3))

        if not return_dict:
            return (density_map, pred_count, pos_boxes, pos_scores, pos_logits)

        return LogitsDensityOutput(
            density_map=density_map,
            pred_count=pred_count,
            pred_boxes=pos_boxes,
            pred_scores=pos_scores,
            raw_logits=pos_logits,
        )


class CountEXLogitsDensityV2(GroundingDinoPreTrainedModel):
    """
    V2: Logits density with a small learnable refinement network.

    Instead of just temperature scaling, learn a small network to
    transform logits → per-query density contribution.
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        from .modeling_grounding_dino import (
            GroundingDinoContrastiveEmbedding,
            GroundingDinoMLPPredictionHead,
        )

        self.model = GroundingDinoModel(config)

        _class_embed = GroundingDinoContrastiveEmbedding(config)
        if config.decoder_bbox_embed_share:
            _bbox_embed = GroundingDinoMLPPredictionHead(
                input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
            )
            self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        else:
            self.bbox_embed = nn.ModuleList([
                GroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                ) for _ in range(config.decoder_layers)
            ])
        self.class_embed = nn.ModuleList([_class_embed for _ in range(config.decoder_layers)])

        self.model.decoder.bbox_embed = self.bbox_embed
        self.model.decoder.class_embed = self.class_embed

        # Small network to transform logits → density weight
        # Input: max logit value per query
        # Output: density contribution (positive scalar)
        self.logit_to_density = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensure positive output
        )

        # Box size to sigma mapping
        self.box_to_sigma = nn.Sequential(
            nn.Linear(2, 16),  # Input: (w, h)
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Output in [0, 1], will scale to [0.05, 0.5]
        )

        self.config = config
        self.post_init()
        self._freeze_encoder_decoder()

    def _freeze_encoder_decoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.bbox_embed.parameters():
            param.requires_grad = False
        for param in self.class_embed.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"CountEXLogitsDensityV2: Trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _get_detections(self, pixel_values, input_ids, token_type_ids=None,
                        attention_mask=None, pixel_mask=None, exemplars=None):
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                pixel_mask=pixel_mask,
                return_dict=True,
                exemplars=exemplars,
            )

            hidden_states = outputs.intermediate_hidden_states
            init_reference_points = outputs.init_reference_points
            inter_references_points = outputs.intermediate_reference_points
            enc_text_hidden_state = outputs.encoder_last_hidden_state_text

            num_levels = hidden_states.shape[1]
            level = num_levels - 1

            reference = inter_references_points[:, level - 1] if level > 0 else init_reference_points
            reference = torch.special.logit(reference, eps=1e-5)

            B, T, _ = enc_text_hidden_state.shape
            text_token_mask = torch.ones(B, T, dtype=torch.bool, device=enc_text_hidden_state.device)

            logits = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=text_token_mask,
            )

            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            else:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            pred_boxes = outputs_coord_logits.sigmoid()

        return logits.detach(), pred_boxes.detach()

    def _create_density(self, logits, boxes, H, W):
        """Create density map using learned transformations."""
        B, N, T = logits.shape
        device = logits.device
        dtype = logits.dtype

        # Get max logit per query
        max_logits = logits.max(dim=-1)[0]  # [B, N]

        # Transform logits to density weights (LEARNABLE)
        weights = self.logit_to_density(max_logits.unsqueeze(-1)).squeeze(-1)  # [B, N]

        # Get box sizes and compute sigma (LEARNABLE)
        box_wh = boxes[..., 2:4]  # [B, N, 2] - w, h
        sigma_raw = self.box_to_sigma(box_wh).squeeze(-1)  # [B, N]
        sigma = sigma_raw * 0.45 + 0.05  # Scale to [0.05, 0.5]

        # Box centers
        cx = boxes[..., 0]  # [B, N]
        cy = boxes[..., 1]  # [B, N]

        # Create coordinate grids
        y_coords = torch.linspace(0, 1, H, device=device, dtype=dtype)
        x_coords = torch.linspace(0, 1, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Reshape for broadcasting
        cx = cx.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        cy = cy.unsqueeze(-1).unsqueeze(-1)
        sigma = sigma.unsqueeze(-1).unsqueeze(-1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        xx = xx.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        yy = yy.unsqueeze(0).unsqueeze(0)

        # Gaussian blobs
        gaussian = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

        # Normalize and weight
        gaussian_sum = gaussian.sum(dim=(-2, -1), keepdim=True) + 1e-6
        gaussian = gaussian / gaussian_sum

        weighted = gaussian * weights
        density = weighted.sum(dim=1, keepdim=True)  # [B, 1, H, W]

        return density

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        neg_pixel_values: Optional[torch.FloatTensor] = None,
        neg_input_ids: Optional[torch.LongTensor] = None,
        neg_token_type_ids: Optional[torch.LongTensor] = None,
        neg_attention_mask: Optional[torch.LongTensor] = None,
        neg_pixel_mask: Optional[torch.BoolTensor] = None,
        pos_exemplars: Optional[torch.FloatTensor] = None,
        neg_exemplars: Optional[torch.FloatTensor] = None,
        use_neg: bool = True,
        output_size: int = 512,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[LogitsDensityOutput, Tuple]:

        pos_logits, pos_boxes = self._get_detections(
            pixel_values, input_ids, token_type_ids, attention_mask, pixel_mask, pos_exemplars
        )

        pos_density = self._create_density(pos_logits, pos_boxes, output_size, output_size)

        if use_neg and neg_input_ids is not None:
            neg_logits, neg_boxes = self._get_detections(
                neg_pixel_values if neg_pixel_values is not None else pixel_values,
                neg_input_ids, neg_token_type_ids, neg_attention_mask,
                neg_pixel_mask if neg_pixel_mask is not None else pixel_mask,
                neg_exemplars
            )
            neg_density = self._create_density(neg_logits, neg_boxes, output_size, output_size)
            density_map = F.relu(pos_density - neg_density)
        else:
            density_map = pos_density

        pred_count = density_map.sum(dim=(1, 2, 3))
        pred_scores = pos_logits.sigmoid().max(dim=-1)[0]

        if not return_dict:
            return (density_map, pred_count, pos_boxes, pred_scores, pos_logits)

        return LogitsDensityOutput(
            density_map=density_map,
            pred_count=pred_count,
            pred_boxes=pos_boxes,
            pred_scores=pred_scores,
            raw_logits=pos_logits,
        )


class CountEXDensityOnlyWithDetectionPrior(GroundingDinoPreTrainedModel):
    """
    Density model that uses decoder detections as localization prior.

    Architecture:
        1. Run full encoder-decoder to get detection boxes (FROZEN)
        2. Create density prior from high-confidence detections
        3. Guide density estimation with this prior

    The encoder and decoder are frozen to preserve pretrained detection ability.
    Only the density head is trained.
    """

    def __init__(self, config: GroundingDinoConfig, freeze_encoder_decoder: bool = True):
        super().__init__(config)

        from .modeling_grounding_dino import (
            GroundingDinoContrastiveEmbedding,
            GroundingDinoMLPPredictionHead,
        )

        # Full model with decoder
        self.model = GroundingDinoModel(config)

        # Detection heads (from GroundingDinoForObjectDetection)
        _class_embed = GroundingDinoContrastiveEmbedding(config)
        if config.decoder_bbox_embed_share:
            _bbox_embed = GroundingDinoMLPPredictionHead(
                input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
            )
            self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        else:
            self.bbox_embed = nn.ModuleList([
                GroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                ) for _ in range(config.decoder_layers)
            ])
        self.class_embed = nn.ModuleList([_class_embed for _ in range(config.decoder_layers)])

        # Link to decoder for box refinement
        self.model.decoder.bbox_embed = self.bbox_embed
        self.model.decoder.class_embed = self.class_embed

        # Larger density head since encoder is frozen
        # Note: num_up_blocks=3 for 512x512 output (64x64 -> 128 -> 256 -> 512)
        self.density_head = LargeDensityFPNHead(
            in_channels=256,
            mid_channels=128,  # Larger
            num_scales=4,
            num_up_blocks=3,  # 3 blocks for 512x512 output
            with_uncertainty=True,
        )

        # Prior fusion parameters
        # prior_weight: controls spatial modulation strength (sigmoid constrained to [0, 1])
        # prior_scale: learnable scale for detection prior count contribution
        self.prior_weight = nn.Parameter(torch.tensor(0.0))  # Start at sigmoid(0)=0.5
        self.prior_scale = nn.Parameter(torch.tensor(1.0))  # Learnable scale for prior

        self.config = config
        self.freeze_encoder_decoder = freeze_encoder_decoder
        self.post_init()

        # Freeze encoder and decoder after post_init
        if freeze_encoder_decoder:
            self._freeze_encoder_decoder()

    def _freeze_encoder_decoder(self):
        """Freeze all parameters except density_head, prior_weight, and prior_scale."""
        # Freeze the main model (encoder + decoder)
        for param in self.model.parameters():
            param.requires_grad = False

        # Freeze detection heads
        for param in self.bbox_embed.parameters():
            param.requires_grad = False
        for param in self.class_embed.parameters():
            param.requires_grad = False

        print("Frozen encoder and decoder. Trainable: density_head, prior_weight, prior_scale")
        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _get_encoder_features(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        exemplars: Optional[torch.FloatTensor] = None,
    ) -> Tuple[List[torch.Tensor], any]:
        """Run full model and return encoder features + model outputs.

        Note: Runs with torch.no_grad() since encoder/decoder are frozen.
        Feature maps are detached to prevent gradient flow through frozen params.
        """
        # Run frozen encoder/decoder without computing gradients
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                pixel_mask=pixel_mask,
                return_dict=True,
                exemplars=exemplars,
            )

        # Extract multi-scale feature maps from encoder
        spatial_shapes = outputs.spatial_shapes
        encoder_vision = outputs.encoder_last_hidden_state_vision

        token_num = 0
        token_num_list = [0]
        for i in range(len(spatial_shapes)):
            token_num += spatial_shapes[i][0] * spatial_shapes[i][1]
            token_num_list.append(token_num.item())

        feature_maps = []
        for i in range(len(spatial_shapes)):
            feat = encoder_vision[:, token_num_list[i]:token_num_list[i+1], :]
            h, w = spatial_shapes[i]
            b, t, d = feat.shape
            feat = feat.reshape(b, h, w, d).permute(0, 3, 1, 2)
            # Detach to ensure no gradient flows back through frozen encoder
            feature_maps.append(feat.detach())

        return feature_maps, outputs

    def _get_detections(
        self,
        outputs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract detection boxes and scores from model outputs.

        Note: Runs with torch.no_grad() since detection heads are frozen.
        Boxes and scores are detached to prevent gradient flow.
        """
        # Run frozen detection heads without computing gradients
        with torch.no_grad():
            hidden_states = outputs.intermediate_hidden_states
            init_reference_points = outputs.init_reference_points
            inter_references_points = outputs.intermediate_reference_points
            enc_text_hidden_state = outputs.encoder_last_hidden_state_text

            # Get last layer predictions
            num_levels = hidden_states.shape[1]
            level = num_levels - 1

            reference = inter_references_points[:, level - 1] if level > 0 else init_reference_points
            reference = torch.special.logit(reference, eps=1e-5)

            # Create text_token_mask matching the encoded text hidden state shape
            # The text encoder may add special tokens, so we use the actual output shape
            B, T, _ = enc_text_hidden_state.shape
            text_token_mask = torch.ones(B, T, dtype=torch.bool, device=enc_text_hidden_state.device)

            # Class predictions
            outputs_class = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=text_token_mask,
            )

            # Box predictions
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            else:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            pred_boxes = outputs_coord_logits.sigmoid()  # [B, N, 4] in cxcywh

            # Get max score across text tokens for each query
            pred_scores = outputs_class.sigmoid().max(dim=-1)[0]  # [B, N]

        # Detach to ensure no gradient flows back through frozen detection heads
        return pred_boxes.detach(), pred_scores.detach()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        neg_pixel_values: Optional[torch.FloatTensor] = None,
        neg_input_ids: Optional[torch.LongTensor] = None,
        neg_token_type_ids: Optional[torch.LongTensor] = None,
        neg_attention_mask: Optional[torch.LongTensor] = None,
        neg_pixel_mask: Optional[torch.BoolTensor] = None,
        pos_exemplars: Optional[torch.FloatTensor] = None,
        neg_exemplars: Optional[torch.FloatTensor] = None,
        use_neg: bool = True,
        score_threshold: float = 0.3,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[DensityWithPriorOutput, Tuple]:
        """
        Forward pass with detection-based localization prior.

        Args:
            score_threshold: Detection confidence threshold for prior
        """
        # Get positive features and detections
        pos_features, pos_outputs = self._get_encoder_features(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            exemplars=pos_exemplars,
        )

        # Get detection boxes and scores
        pred_boxes, pred_scores = self._get_detections(pos_outputs)

        # Process negative prompt
        if use_neg and neg_input_ids is not None:
            neg_features, _ = self._get_encoder_features(
                pixel_values=neg_pixel_values if neg_pixel_values is not None else pixel_values,
                input_ids=neg_input_ids,
                token_type_ids=neg_token_type_ids,
                attention_mask=neg_attention_mask,
                pixel_mask=neg_pixel_mask if neg_pixel_mask is not None else pixel_mask,
                exemplars=neg_exemplars,
            )

            diff_features = []
            for pf, nf in zip(pos_features, neg_features):
                diff = F.relu(pf - nf)
                diff_features.append(diff)
        else:
            diff_features = pos_features
            neg_features = None

        # Density prediction from head
        raw_density, density_log_var = self.density_head(diff_features)

        # Create detection prior (spatial guidance from frozen detector)
        H, W = raw_density.shape[-2:]
        detection_prior = create_gaussian_density_from_boxes(
            pred_boxes, pred_scores, H, W,
            score_threshold=score_threshold,
        )

        # Simple fusion mechanism:
        # density_map = raw_density + prior_scale * detection_prior
        #
        # The detection prior from frozen detector provides localization guidance.
        # raw_density learns to predict density, detection_prior guides where.
        # prior_scale learns the optimal balance.
        #
        # IMPORTANT: Always use prior_scale in computation to ensure consistent
        # gradient shapes across DDP ranks (no conditional branches).

        scale = F.relu(self.prior_scale)
        # Simple addition: detection prior provides localization, density head refines
        density_map = raw_density + scale * detection_prior

        # Also include prior_weight in computation for gradient consistency
        # (even though we're not using attention anymore, keep it for backwards compat)
        w = torch.sigmoid(self.prior_weight)
        density_map = density_map + 0.0 * w  # No-op but keeps gradient flowing

        pred_count = density_map.sum(dim=(1, 2, 3))

        if not return_dict:
            return (density_map, density_log_var, pred_count, detection_prior, pred_boxes, pred_scores)

        return DensityWithPriorOutput(
            density_map=density_map,
            density_log_var=density_log_var,
            pred_count=pred_count,
            detection_prior=detection_prior,
            raw_density=raw_density,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pos_features=pos_features,
            neg_features=neg_features,
        )


class CountEXDensityWithLogitsPrior(GroundingDinoForObjectDetection):
    """
    Weakly-supervised counting model inheriting from GroundingDinoForObjectDetection.
    Uses detection logits as a soft prior to guide density estimation.

    Inherits from GroundingDinoForObjectDetection to ensure proper weight loading.

    Architecture:
        Image + Pos Prompt → Encoder → Pos Features ─┐
                                   ↓                  │
                              Decoder → Logits ───────┼──→ Detection Prior
                                                      │           ↓
        Image + Neg Prompt → Encoder → Neg Features ─┘           │
                                   ↓                              │
                          relu(pos - neg)                         │
                                   ↓                              │
                            Density Head ─────────────────────────┴──→ Final Density
                                   ↓
                              density.sum() → Count

    Key features:
    1. Vision encoder trainable, decoder/detection heads frozen
    2. Detection logits provide soft localization prior
    3. Prior is fused with learned density via learnable gate
    4. Negative prompt guidance: relu(pos - neg) at density level
    """

    def __init__(self, config: GroundingDinoConfig):
        # Initialize parent class (GroundingDinoForObjectDetection)
        # This properly sets up self.model, self.bbox_embed, self.class_embed
        super().__init__(config)

        # Density head for converting features to density map
        # in_channels=512 because we concatenate pos and neg features like CountEX
        self.density_head = CountEXDensityFPNHead(
            in_channels=512,  # pos_feat + neg_feat
            mid_channels=128,
        )

        # Prior fusion parameters
        # temperature: controls sharpness of logits → scores
        self.temperature = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        # sigma_factor: controls Gaussian spread for detection prior
        self.sigma_factor = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
        # prior_scale: learnable weight for detection prior contribution
        self.prior_scale = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        # density_scale: scale factor for density head output
        # Initial raw_sum is ~600, we want ~50-100, so scale ~0.1
        # Using exp(density_scale), init to log(0.1) ≈ -2.3
        self.density_scale = nn.Parameter(torch.tensor([-2.3], dtype=torch.float32))

        self.config = config

        # Freeze decoder and detection heads to save memory
        self._freeze_decoder()

        # Print trainable params info
        self._print_trainable_info()

    def _freeze_decoder(self):
        """Freeze decoder, detection heads, and text backbone to save memory.

        Trainable: vision encoder, density head, prior fusion params
        Frozen: decoder, bbox_embed, class_embed, text_backbone
        """
        # Freeze decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = False

        # Freeze detection heads
        for param in self.bbox_embed.parameters():
            param.requires_grad = False
        for param in self.class_embed.parameters():
            param.requires_grad = False

        # Freeze text backbone
        if hasattr(self.model, 'text_backbone'):
            for param in self.model.text_backbone.parameters():
                param.requires_grad = False

    def _print_trainable_info(self):
        """Print information about trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print("=" * 60)
        print("CountEXDensityWithLogitsPrior:")
        print("  Trainable: vision encoder, density_head, prior fusion params")
        print("  Frozen: decoder, bbox_embed, class_embed, text_backbone")
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print("=" * 60)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        neg_pixel_values: Optional[torch.FloatTensor] = None,
        neg_input_ids: Optional[torch.LongTensor] = None,
        neg_token_type_ids: Optional[torch.LongTensor] = None,
        neg_attention_mask: Optional[torch.LongTensor] = None,
        neg_pixel_mask: Optional[torch.BoolTensor] = None,
        pos_exemplars: Optional[torch.FloatTensor] = None,
        neg_exemplars: Optional[torch.FloatTensor] = None,
        use_neg: bool = True,
        output_size: int = 512,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[DensityWithPriorOutput, Tuple]:
        """
        Forward pass following CountEX pattern for stable weight loading.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # === Positive branch ===
        pos_kwargs = {'exemplars': pos_exemplars}
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            return_dict=True,
            **pos_kwargs,
        )

        # Extract multi-scale feature maps (following CountEX pattern)
        spatial_shapes = outputs.spatial_shapes
        token_num = 0
        token_num_list = [0]
        for i in range(len(spatial_shapes)):
            token_num += spatial_shapes[i][0] * spatial_shapes[i][1]
            token_num_list.append(token_num.item())

        positive_feature_maps = []
        encoder_vision = outputs.encoder_last_hidden_state_vision
        for i in range(len(spatial_shapes)):
            feat = encoder_vision[:, token_num_list[i]:token_num_list[i+1], :]
            h, w = spatial_shapes[i]
            b, t, d = feat.shape
            feat = feat.reshape(b, h, w, d)
            positive_feature_maps.append(feat)

        # === Negative branch ===
        neg_kwargs = {'exemplars': neg_exemplars}
        neg_outputs = self.model(
            pixel_values=neg_pixel_values if neg_pixel_values is not None else pixel_values,
            input_ids=neg_input_ids,
            token_type_ids=neg_token_type_ids,
            attention_mask=neg_attention_mask,
            pixel_mask=neg_pixel_mask if neg_pixel_mask is not None else pixel_mask,
            return_dict=True,
            **neg_kwargs,
        )

        negative_feature_maps = []
        neg_encoder_vision = neg_outputs.encoder_last_hidden_state_vision
        for i in range(len(spatial_shapes)):
            feat = neg_encoder_vision[:, token_num_list[i]:token_num_list[i+1], :]
            h, w = spatial_shapes[i]
            b, t, d = feat.shape
            feat = feat.reshape(b, h, w, d)
            negative_feature_maps.append(feat)

        # === Density head (concatenate pos and neg features like CountEX) ===
        all_feats = []
        for pf, npf in zip(positive_feature_maps, negative_feature_maps):
            pf = pf.permute(0, 3, 1, 2)  # [B, D, H, W]
            npf = npf.permute(0, 3, 1, 2)
            all_feats.append(torch.cat([pf, npf], dim=1))  # [B, 2D, H, W]

        raw_density = self.density_head(all_feats)  # [B, 1, H, W]

        # Apply density scale to bring values to reasonable range
        # exp(-2.3) ≈ 0.1, so raw_sum ~600 -> ~60
        density_scale = torch.exp(torch.clamp(self.density_scale[0], min=-10.0, max=5.0))
        raw_density = raw_density * density_scale

        # === Detection prior from decoder outputs ===
        hidden_states = outputs.intermediate_hidden_states
        init_reference_points = outputs.init_reference_points
        inter_references_points = outputs.intermediate_reference_points
        enc_text_hidden_state = outputs.encoder_last_hidden_state_text

        # Handle exemplar tokens
        if pos_exemplars is not None or attention_mask.shape[1] != enc_text_hidden_state.shape[1]:
            enc_text_hidden_state = enc_text_hidden_state[:, :enc_text_hidden_state.shape[1] - 3, :]

        # Get last layer logits and boxes
        num_levels = hidden_states.shape[1]
        level = num_levels - 1

        if level == 0:
            reference = init_reference_points
        else:
            reference = inter_references_points[:, level - 1]
        reference = torch.special.logit(reference, eps=1e-5)

        with torch.no_grad():
            outputs_class = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=attention_mask.bool(),
            )
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            else:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            pred_boxes = outputs_coord_logits.sigmoid()

        logits = outputs_class
        pred_scores = logits.sigmoid().max(dim=-1)[0]

        # Create detection prior from logits
        H, W = raw_density.shape[-2:]
        temp = F.softplus(self.temperature[0]) + 0.1
        sigma_f = torch.sigmoid(self.sigma_factor[0]) * 0.5 + 0.1
        prior_scale = F.softplus(self.prior_scale[0])

        detection_prior = create_soft_density_from_logits(
            logits=logits.detach(),
            boxes=pred_boxes.detach(),
            height=H,
            width=W,
            temperature=temp,
            sigma_factor=sigma_f,
        )

        # Fuse density with detection prior
        # raw_density is already scaled, detection_prior sum ≈ gt_count
        detection_prior = detection_prior.to(raw_density.dtype)
        density_map = raw_density + prior_scale * detection_prior

        pred_count = density_map.sum(dim=(1, 2, 3))

        if not return_dict:
            return (density_map, None, pred_count, detection_prior, pred_boxes, pred_scores)

        return DensityWithPriorOutput(
            density_map=density_map,
            density_log_var=None,
            pred_count=pred_count,
            detection_prior=detection_prior,
            raw_density=raw_density,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pos_features=positive_feature_maps,
            neg_features=negative_feature_maps,
        )


class CountEXWithDetectionPrior(CountEX):
    """
    CountEX with detection prior for weakly-supervised counting.

    Inherits from CountEX to ensure proper weight loading and density head.
    Adds detection prior from decoder outputs to guide density estimation.

    Architecture:
        CountEX forward → density_map_pred (from density head)
                       → logits, pred_boxes (from decoder)
                                    ↓
                          Detection Prior (Gaussian dots at box centers)
                                    ↓
                       density_map = density_map_pred + prior_scale * detection_prior
                                    ↓
                              pred_count = density_map.sum()

    Key features:
    1. Inherits all CountEX functionality (encoder, decoder, density head, neg fusion)
    2. Adds detection prior from decoder logits/boxes
    3. Learnable prior_scale to weight the detection prior contribution
    4. No density_scale - uses density head output directly (known to converge)
    """

    def __init__(self, config):
        super().__init__(config)

        # Prior fusion parameters
        # temperature: controls sharpness of logits → scores
        self.prior_temperature = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        # prior_scale: fixed at 1.0 (no scaling)

        self.config = config

        # Freeze unused parameters to avoid DDP errors
        self._freeze_unused_params()

        # Print info
        self._print_info()

    def _freeze_unused_params(self):
        """Freeze backbone, encoder, decoder to keep detection prior stable.

        Only the density head is trainable. Detection prior comes from frozen
        encoder+decoder, providing stable pseudo labels for density regression.
        """
        # Freeze backbone (vision encoder)
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Freeze encoder (keeps detection prior stable)
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

        # Remove query_side_neg_pipeline (not used in density-only training)
        if hasattr(self, 'query_side_neg_pipeline'):
            del self.query_side_neg_pipeline

        # Freeze query position embeddings (used by decoder)
        if hasattr(self.model, 'query_position_embeddings'):
            self.model.query_position_embeddings.requires_grad = False

        # Freeze input projections and other embeddings
        if hasattr(self.model, 'input_proj_vision'):
            for proj in self.model.input_proj_vision:
                for param in proj.parameters():
                    param.requires_grad = False
        if hasattr(self.model, 'input_proj_text'):
            for param in self.model.input_proj_text.parameters():
                param.requires_grad = False

    def _print_info(self):
        """Print model information."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print("=" * 60)
        print("CountEXWithDetectionPrior:")
        print("  Trainable: density_head only")
        print("  Frozen: backbone, encoder, decoder (for stable detection prior)")
        print("  Removed: query_side_neg_pipeline")
        print(f"  prior_temperature: {self.prior_temperature[0].item():.3f}")
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print("=" * 60)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        neg_pixel_values: Optional[torch.FloatTensor] = None,
        neg_input_ids: Optional[torch.LongTensor] = None,
        neg_token_type_ids: Optional[torch.LongTensor] = None,
        neg_attention_mask: Optional[torch.LongTensor] = None,
        neg_pixel_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        """
        Forward pass that adds detection prior to CountEX output.
        """
        # Force use_neg=False since we removed query_side_neg_pipeline
        kwargs['use_neg'] = False

        # Call parent CountEX forward
        outputs = super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            neg_pixel_values=neg_pixel_values,
            neg_input_ids=neg_input_ids,
            neg_token_type_ids=neg_token_type_ids,
            neg_attention_mask=neg_attention_mask,
            neg_pixel_mask=neg_pixel_mask,
            **kwargs,
        )

        # Get density from CountEX (this is the raw density head output)
        raw_density = outputs.density_map_pred  # [B, 1, H, W]

        # Get logits and boxes from decoder
        logits = outputs.logits  # [B, N, T]
        pred_boxes = outputs.pred_boxes  # [B, N, 4]

        # Create detection prior from logits
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

        # detection_prior is only used as supervision signal, not added to output
        detection_prior = detection_prior.to(raw_density.dtype)
        density_map = raw_density  # Only use raw_density for prediction

        pred_count = density_map.sum(dim=(1, 2, 3))
        pred_scores = logits.sigmoid().max(dim=-1)[0]

        # Return DensityWithPriorOutput
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
        )
