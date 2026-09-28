# coding=utf-8
"""
AlternatingTrainer: Trainer with alternating training between density head and detection decoder.

GAN-like mutual learning:
- Phase 1 (density): Freeze decoder, train density head with pseudo_density_loss
- Phase 2 (detection): Freeze density head, train decoder with detection_alignment_loss

Based on DensityTrainer with added alternating training logic.
"""

import copy
import torch
import wandb
import numpy as np
import os
import random
from transformers import Trainer
from typing import Dict, Optional, List
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from utils import interval_huber, relative_huber


class AlternatingTrainer(Trainer):
    """
    Trainer with alternating training between density head and detection decoder.
    Based on DensityTrainer with added alternating training logic.
    """

    def __init__(self, *args, llmdet_processor=None, save_qualitative_results=False, **kwargs):
        # Pop custom arguments
        self.val_dataset = kwargs.pop('val_dataset', None)
        self.test_dataset = kwargs.pop('test_dataset', None)
        self.output_dir = kwargs.pop('output_dir', './outputs')

        # Loss settings
        self.count_loss_type = kwargs.pop('count_loss_type', 'interval_huber')
        self.count_loss_weight = kwargs.pop('count_loss_weight', 1.0)

        # Uncertainty settings
        self.use_uncertainty_loss = kwargs.pop('use_uncertainty_loss', False)

        # Negative prompt settings
        self.use_neg_prob = kwargs.pop('use_neg_prob', 0.0)  # Default: no negative for detection prior

        # Feature-guided localization loss
        self.use_feature_guidance_loss = kwargs.pop('use_feature_guidance_loss', False)
        self.feature_guidance_weight = kwargs.pop('feature_guidance_weight', 0.01)

        # Pseudo density loss (density phase)
        self.use_pseudo_density_loss = kwargs.pop('use_pseudo_density_loss', True)
        self.pseudo_density_weight = kwargs.pop('pseudo_density_weight', 1000.0)

        # === Alternating training settings ===
        self.use_alternating_training = kwargs.pop('use_alternating_training', True)
        self.density_phase_length = kwargs.pop('density_phase_length', 100)
        self.detection_phase_length = kwargs.pop('detection_phase_length', 10)
        self.detection_alignment_weight = kwargs.pop('detection_alignment_weight', 1.0)
        self.detection_count_loss_weight = kwargs.pop('detection_count_loss_weight', 1.0)

        # === Mean Teacher settings ===
        self.use_mean_teacher = kwargs.pop('use_mean_teacher', False)
        self.consistency_weight = kwargs.pop('consistency_weight', 1.0)
        self.ema_decay = kwargs.pop('ema_decay', 0.999)
        self._teacher_model_init = kwargs.pop('teacher_model', None)

        # === Count noise (robustness ablation) ===
        self.count_noise_ratio = kwargs.pop('count_noise_ratio', 0.0)

        # === Detection-head validation tracking (rebuttal: matched inference) ===
        self.save_best_det = kwargs.pop('save_best_det', False)
        self.det_eval_threshold = kwargs.pop('det_eval_threshold', 0.42)
        self.best_val_det_mae = float('inf')

        # Current phase tracking
        self.current_phase = 'density'  # Start with density phase

        # Validate loss type
        assert self.count_loss_type in ['interval_huber', 'ae', 'mse', 'relative_huber'], \
            f"Invalid count_loss_type: {self.count_loss_type}"

        self.best_val_mae = float('inf')
        self.best_test_mae = float('inf')

        super().__init__(*args, **kwargs)

        self.llmdet_processor = llmdet_processor
        self.device = self.args.device if hasattr(self.args, 'device') else 'cuda'
        self.save_qualitative_results = save_qualitative_results

        # Mean Teacher: accept pre-created teacher model (e.g. CountEX)
        self.teacher_model = self._teacher_model_init
        del self._teacher_model_init
        if self.use_mean_teacher and self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            if self._is_main_process():
                teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
                print(f"  Mean Teacher: using external teacher ({type(self.teacher_model).__name__}) with {teacher_params:,} frozen params (ema_decay={self.ema_decay})")

        # Metrics tracking
        self.loss_count_list = []
        self.loss_uncertainty_list = []
        self.train_mae = 0
        self.train_rmse = 0
        self.train_sample_num = 0

        # Prediction statistics tracking
        self.pred_count_list = []
        self.gt_count_list = []
        self.density_max_list = []
        self.density_mean_list = []
        self.density_nonzero_ratio_list = []

        # Detection Prior specific tracking
        self.detection_count_list = []
        self.prior_weight_list = []
        self.prior_scale_list = []
        self.detection_prior_sum_list = []
        self.raw_density_sum_list = []
        self.pseudo_density_loss_list = []

        # Detection alignment loss tracking (for detection phase)
        self.detection_alignment_loss_list = []

        # Mean Teacher consistency loss tracking
        self.consistency_loss_list = []

        # Note: We no longer freeze/unfreeze parameters during alternating training
        # Only the loss switches between phases, parameters stay the same

        if self._is_main_process():
            print("=" * 60)
            print("AlternatingTrainer initialized")
            print(f"  use_alternating_training: {self.use_alternating_training}")
            print(f"  density_phase_length: {self.density_phase_length}")
            print(f"  detection_phase_length: {self.detection_phase_length}")
            print(f"  pseudo_density_weight: {self.pseudo_density_weight}")
            print(f"  detection_alignment_weight: {self.detection_alignment_weight}")
            print(f"  use_mean_teacher: {self.use_mean_teacher}")
            if self.use_mean_teacher:
                print(f"  consistency_weight: {self.consistency_weight}")
            print("=" * 60)

    def _is_main_process(self):
        """Check if current process is main (rank 0)."""
        from torch import distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0

    def _get_student_model(self):
        """Get the underlying student model (unwrap DDP/DeepSpeed)."""
        model = self.model
        # Unwrap DeepSpeed engine
        if hasattr(model, 'module'):
            model = model.module
        return model

    @torch.no_grad()
    def _ema_update_teacher(self):
        """Update teacher parameters via EMA from student (named-param matching for different architectures)."""
        student_dict = dict(self._get_student_model().named_parameters())
        for name, t_param in self.teacher_model.named_parameters():
            if name in student_dict:
                s_param = student_dict[name]
                t_param.data.mul_(self.ema_decay).add_(s_param.data.to(t_param.device), alpha=1.0 - self.ema_decay)

    def _compute_detection_alignment_loss(self, pred_logits, pred_boxes, density_map):
        """Detection alignment loss: high confidence boxes should be at high density locations."""
        # Get predicted confidence
        pred_conf = pred_logits.sigmoid().max(dim=-1)[0]  # [B, N]

        # Get box centers
        cx = pred_boxes[:, :, 0]  # [B, N]
        cy = pred_boxes[:, :, 1]  # [B, N]

        # Sample density at box centers
        # grid_sample expects grid shape [B, H_out, W_out, 2]
        # We want to sample at N points, so use [B, N, 1, 2] -> output [B, C, N, 1]
        grid_x = cx * 2 - 1
        grid_y = cy * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # [B, N, 1, 2]
        grid = grid.to(density_map.dtype)

        # Output shape: [B, 1, N, 1] -> squeeze to [B, N]
        density_at_centers = F.grid_sample(
            density_map.detach(),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # [B, 1, N, 1]
        density_at_centers = density_at_centers.squeeze(1).squeeze(-1)  # [B, N]

        # Normalize to [0, 1]
        density_max = density_at_centers.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        density_target = density_at_centers / density_max  # [B, N]

        # BCE loss (convert to float32 as BCE doesn't support bfloat16)
        loss = F.binary_cross_entropy(pred_conf.float(), density_target.float())
        return loss

    def _compute_detection_count_loss(self, pred_logits, gt_count):
        """Detection count loss: soft detection count should match gt_count."""
        pred_conf = pred_logits.sigmoid().max(dim=-1)[0]  # [B, N]
        soft_det_count = pred_conf.sum(dim=-1)  # [B]
        gt_count_tensor = torch.tensor([gt_count], dtype=soft_det_count.dtype, device=soft_det_count.device)
        loss = F.l1_loss(soft_det_count, gt_count_tensor)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with alternating training phases.

        Density phase: count_loss + pseudo_density_loss
        Detection phase: count_loss + detection_alignment_loss
        """
        # === Phase switching logic ===
        if self.use_alternating_training:
            # Asymmetric phase lengths: density_phase_length + detection_phase_length = one cycle
            cycle_length = self.density_phase_length + self.detection_phase_length
            step_in_cycle = self.state.global_step % cycle_length
            new_phase = 'density' if step_in_cycle < self.density_phase_length else 'detection'

            if new_phase != self.current_phase:
                if self._is_main_process():
                    print(f"\n{'='*60}")
                    print(f"[Step {self.state.global_step}] PHASE SWITCH: {self.current_phase.upper()} -> {new_phase.upper()} (loss only)")
                    print(f"{'='*60}")
                self.current_phase = new_phase
                # Note: We only switch loss, not trainable parameters

        # Extract inputs
        pos_llm_det_inputs = inputs['pos_llm_det_inputs']
        gt_count = inputs['pos_count']

        # Apply count noise for robustness ablation
        if self.count_noise_ratio > 0 and self.model.training:
            noise_factor = 1.0 + random.uniform(-self.count_noise_ratio, self.count_noise_ratio)
            gt_count = max(0, round(gt_count * noise_factor))

        # Move inputs to device
        pos_llm_det_inputs = pos_llm_det_inputs.to(self.device)
        pos_llm_det_inputs['pixel_values'] = pos_llm_det_inputs['pixel_values'].to(torch.bfloat16)

        # Add exemplars if available
        if 'pos_exemplars' in inputs:
            pos_llm_det_inputs['pos_exemplars'] = inputs['pos_exemplars']
        if 'neg_exemplars' in inputs:
            pos_llm_det_inputs['neg_exemplars'] = inputs['neg_exemplars']

        # Process negative inputs
        neg_llm_det_inputs = inputs['neg_llm_det_inputs']
        neg_llm_det_inputs = {k: v.to(self.device) for k, v in neg_llm_det_inputs.items()}
        neg_llm_det_inputs['pixel_values'] = neg_llm_det_inputs['pixel_values'].to(torch.bfloat16)

        # Merge negative inputs
        pos_llm_det_inputs['neg_token_type_ids'] = neg_llm_det_inputs['token_type_ids']
        pos_llm_det_inputs['neg_attention_mask'] = neg_llm_det_inputs['attention_mask']
        pos_llm_det_inputs['neg_pixel_mask'] = neg_llm_det_inputs['pixel_mask']
        pos_llm_det_inputs['neg_pixel_values'] = neg_llm_det_inputs['pixel_values']
        pos_llm_det_inputs['neg_input_ids'] = neg_llm_det_inputs['input_ids']

        # Determine whether to use negative prompt (for augmentation)
        from torch import distributed as dist
        if dist.is_initialized():
            if dist.get_rank() == 0:
                use_neg = float(random.random() <= self.use_neg_prob)
            else:
                use_neg = 0.0
            use_neg_tensor = torch.tensor([use_neg], dtype=torch.float32, device=self.device)
            dist.broadcast(use_neg_tensor, src=0)
            use_neg = bool(use_neg_tensor.item())
        else:
            use_neg = random.random() <= self.use_neg_prob

        pos_llm_det_inputs['use_neg'] = True

        # Forward pass (student, managed by DeepSpeed/DDP)
        outputs = model(**pos_llm_det_inputs)

        # Mean Teacher: run teacher forward separately + EMA update
        teacher_density_map = None
        if self.use_mean_teacher and self.teacher_model is not None:
            # Lazy move teacher to correct device on first use
            if not hasattr(self, '_teacher_on_device'):
                self.teacher_model = self.teacher_model.to(self.device)
                self._teacher_on_device = True
                if self._is_main_process():
                    print(f"[Mean Teacher] Teacher model moved to {self.device}")

            # EMA update teacher every step
            if self.state.global_step > 0:
                self._ema_update_teacher()

            # Teacher forward (separate model, not managed by DeepSpeed)
            # Teacher may be CountEX (density_map_pred) or CountEXStage2 (density_map)
            with torch.no_grad():
                teacher_out = self.teacher_model(**pos_llm_det_inputs)
                if hasattr(teacher_out, 'density_map'):
                    teacher_density_map = teacher_out.density_map
                else:
                    teacher_density_map = teacher_out.density_map_pred

        # Get predictions
        pred_count = outputs.pred_count  # [B]
        density_map = outputs.density_map
        density_log_var = getattr(outputs, 'density_log_var', None)

        # Convert gt_count to tensor
        gt_count_tensor = torch.tensor([gt_count], dtype=pred_count.dtype, device=self.device)

        # Compute base count loss
        if self.count_loss_type == 'interval_huber':
            base_loss = interval_huber(pred_count, gt_count_tensor)
        elif self.count_loss_type == 'ae':
            base_loss = F.l1_loss(pred_count, gt_count_tensor)
        elif self.count_loss_type == 'mse':
            base_loss = F.mse_loss(pred_count, gt_count_tensor)
        elif self.count_loss_type == 'relative_huber':
            base_loss = relative_huber(pred_count, gt_count_tensor)
        else:
            raise ValueError(f"Unknown count_loss_type: {self.count_loss_type}")

        # Apply uncertainty weighting if enabled
        if self.use_uncertainty_loss and density_log_var is not None:
            # Average log_var across spatial dimensions
            avg_log_var = density_log_var.mean()

            # Uncertainty-weighted loss (Kendall et al. NIPS 2017)
            # loss = 0.5 * exp(-log_var) * base_loss + 0.5 * log_var
            precision = torch.exp(-avg_log_var)
            loss = 0.5 * precision * base_loss + 0.5 * avg_log_var

            self.loss_uncertainty_list.append(avg_log_var.item())
        else:
            loss = base_loss

        loss = loss * self.count_loss_weight

        # In detection phase, zero out count_loss to only use detection_alignment_loss
        if self.use_alternating_training and self.current_phase == 'detection':
            loss = loss * 0.0  # Keep gradient graph but zero contribution

        # Feature guidance loss: encourage density to correlate with diff_features
        if self.use_feature_guidance_loss and outputs.pos_features is not None:
            feature_guidance_loss = self._compute_feature_guidance_loss(
                density_map=density_map,
                pos_features=outputs.pos_features,
                neg_features=outputs.neg_features,
                use_neg=use_neg,
            )
            loss = loss + self.feature_guidance_weight * feature_guidance_loss

        # === Phase-specific losses ===
        pseudo_density_loss = None
        detection_alignment_loss = None

        # Determine if we should use density phase loss
        is_density_phase = (not self.use_alternating_training) or (self.current_phase == 'density')
        is_detection_phase = self.use_alternating_training and (self.current_phase == 'detection')

        # Density phase: pseudo density loss
        if is_density_phase and self.use_pseudo_density_loss and hasattr(outputs, 'detection_prior') and outputs.detection_prior is not None:
            raw_density = outputs.raw_density if hasattr(outputs, 'raw_density') and outputs.raw_density is not None else density_map
            detection_prior = outputs.detection_prior
            detection_prior = detection_prior.to(raw_density.dtype)
            prior_sum = detection_prior.sum(dim=(1, 2, 3), keepdim=True)
            normalized_prior = detection_prior / (prior_sum + 1e-6) * float(gt_count)
            # cast to fp32 like the BCE term below: under DDP + autocast the bf16
            # reduction otherwise yields a grad dtype the backward pass rejects
            pseudo_density_loss = F.mse_loss(raw_density.float(), normalized_prior.float(), reduction='sum') / (float(gt_count) + 1e-6)
            loss = loss + self.pseudo_density_weight * pseudo_density_loss
            self.pseudo_density_loss_list.append(pseudo_density_loss.item())

        # Detection phase: detection alignment loss + detection count loss
        detection_count_loss = None
        if is_detection_phase and hasattr(outputs, 'logits') and outputs.logits is not None:
            detection_alignment_loss = self._compute_detection_alignment_loss(
                pred_logits=outputs.logits,
                pred_boxes=outputs.pred_boxes,
                density_map=density_map.detach()
            )
            detection_count_loss = self._compute_detection_count_loss(
                pred_logits=outputs.logits,
                gt_count=gt_count,
            )
            loss = loss + self.detection_alignment_weight * detection_alignment_loss
            loss = loss + self.detection_count_loss_weight * detection_count_loss
            self.detection_alignment_loss_list.append(detection_alignment_loss.item())

        # Mean Teacher consistency loss: MSE between student and teacher density maps
        consistency_loss = None
        if self.use_mean_teacher and teacher_density_map is not None:
            consistency_loss = F.mse_loss(density_map.float(), teacher_density_map.float())
            loss = loss + self.consistency_weight * consistency_loss
            self.consistency_loss_list.append(consistency_loss.item())
            # print(f"Consistency loss: {consistency_loss.item()}")

        # Debug: print losses at early steps (only main process)
        if self._is_main_process() and (self.state.global_step < 10 or self.state.global_step % 20 == 0):
            phase_str = f"[{self.current_phase.upper()}]" if self.use_alternating_training else ""
            consist_str = f", consist={self.consistency_weight * consistency_loss.item():.4f}" if consistency_loss is not None else ""
            if pseudo_density_loss is not None:
                weighted_pseudo = self.pseudo_density_weight * pseudo_density_loss.item()
                print(f"{phase_str} [Step {self.state.global_step}] pred={pred_count.item():.1f}, gt={gt_count}, loss={loss.item():.2f}, pseudo={weighted_pseudo:.2f}{consist_str}")
            elif detection_alignment_loss is not None:
                weighted_det = self.detection_alignment_weight * detection_alignment_loss.item()
                weighted_cnt = self.detection_count_loss_weight * detection_count_loss.item() if detection_count_loss is not None else 0
                soft_det_cnt = outputs.logits.sigmoid().max(dim=-1)[0].sum().item()
                print(f"{phase_str} [Step {self.state.global_step}] pred={pred_count.item():.1f}, gt={gt_count}, loss={loss.item():.2f}, det_align={weighted_det:.4f}, det_cnt={weighted_cnt:.2f} (soft_det={soft_det_cnt:.1f}){consist_str}")
            else:
                print(f"{phase_str} [Step {self.state.global_step}] pred={pred_count.item():.1f}, gt={gt_count}, loss={loss.item():.2f}{consist_str}")

        self.loss_count_list.append(loss.item())

        # Track metrics
        pred_cnt = pred_count.item()
        gt_cnt = float(gt_count)
        cnt_err = abs(pred_cnt - gt_cnt)
        self.train_mae += cnt_err
        self.train_rmse += cnt_err ** 2
        self.train_sample_num += 1

        # Track prediction statistics
        self.pred_count_list.append(pred_cnt)
        self.gt_count_list.append(gt_cnt)
        self.density_max_list.append(density_map.max().item())
        self.density_mean_list.append(density_map.mean().item())
        nonzero_ratio = (density_map > 1e-6).float().mean().item()
        self.density_nonzero_ratio_list.append(nonzero_ratio)

        # Track Detection Prior specific metrics
        if hasattr(outputs, 'detection_prior') and outputs.detection_prior is not None:
            # Count high-confidence detections
            if hasattr(outputs, 'pred_scores') and outputs.pred_scores is not None:
                num_detections = (outputs.pred_scores > 0.3).sum().item()
                self.detection_count_list.append(num_detections)
            # Track prior weight
            if hasattr(model, 'prior_weight'):
                prior_w = torch.sigmoid(model.prior_weight).item()
                self.prior_weight_list.append(prior_w)
            # Track prior scale
            if hasattr(model, 'prior_scale'):
                prior_s = F.relu(model.prior_scale).item()
                self.prior_scale_list.append(prior_s)
            # Track detection prior sum
            self.detection_prior_sum_list.append(outputs.detection_prior.sum().item())
            # Track raw density sum (before fusion)
            if hasattr(outputs, 'raw_density') and outputs.raw_density is not None:
                self.raw_density_sum_list.append(outputs.raw_density.sum().item())

        # Logging
        if hasattr(self.state, 'global_step') and self.state.global_step % self.args.logging_steps == 0 and self.state.global_step > 0:
            self._log_metrics()

        # Evaluation
        if hasattr(self.state, 'global_step') and hasattr(self.args, 'eval_steps'):
            if self.state.global_step % self.args.eval_steps == 0 and self.state.global_step > 0:
                for dataset, prefix in [(self.val_dataset, "val"), (self.test_dataset, "test")]:
                    if dataset is not None:
                        self.evaluate(eval_dataset=dataset, metric_key_prefix=prefix,
                                      save_qualitative_results=self.save_qualitative_results)

        if return_outputs:
            return loss, outputs
        return loss

    def _compute_feature_guidance_loss(self, density_map, pos_features, neg_features, use_neg):
        """
        Compute feature guidance loss to encourage density to be spatially correlated
        with diff_features magnitude.

        This loss serves as an inductive bias for localization in weakly-supervised setting.

        Args:
            density_map: [B, 1, H, W] predicted density
            pos_features: List of [B, C, H_i, W_i] positive features at different scales
            neg_features: List of [B, C, H_i, W_i] negative features (or None)
            use_neg: Whether negative features were used

        Returns:
            Scalar loss value
        """
        # Compute diff_features at highest resolution (P3)
        if use_neg and neg_features is not None:
            diff_feat = F.relu(pos_features[0] - neg_features[0])  # [B, C, H/8, W/8]
        else:
            diff_feat = pos_features[0]

        # Compute feature magnitude (L2 norm across channels)
        feat_magnitude = diff_feat.norm(dim=1, keepdim=True)  # [B, 1, H/8, W/8]

        # Upsample to match density resolution
        feat_magnitude = F.interpolate(
            feat_magnitude, size=density_map.shape[-2:],
            mode='bilinear', align_corners=False
        )

        # Normalize both to [0, 1] for comparison
        # Use min-max normalization per sample
        B = density_map.shape[0]
        density_flat = density_map.view(B, -1)
        feat_flat = feat_magnitude.view(B, -1)

        # Add small epsilon for numerical stability
        eps = 1e-6

        # Min-max normalize
        d_min = density_flat.min(dim=1, keepdim=True)[0]
        d_max = density_flat.max(dim=1, keepdim=True)[0]
        density_norm = (density_flat - d_min) / (d_max - d_min + eps)

        f_min = feat_flat.min(dim=1, keepdim=True)[0]
        f_max = feat_flat.max(dim=1, keepdim=True)[0]
        feat_norm = (feat_flat - f_min) / (f_max - f_min + eps)

        # Cosine similarity loss (encourage high correlation)
        # Higher correlation -> lower loss
        cos_sim = F.cosine_similarity(density_norm, feat_norm, dim=1)  # [B]
        loss = (1 - cos_sim).mean()

        return loss

    def _log_metrics(self):
        """Log training metrics to wandb."""
        # Average losses
        avg_loss_count = sum(self.loss_count_list) / len(self.loss_count_list) if self.loss_count_list else 0.0

        loss_logs = {
            "train/loss_count": avg_loss_count,
        }

        if self.loss_uncertainty_list:
            avg_uncertainty = sum(self.loss_uncertainty_list) / len(self.loss_uncertainty_list)
            loss_logs["train/avg_log_var"] = avg_uncertainty

        # Add prediction statistics
        if self.pred_count_list:
            pred_arr = np.array(self.pred_count_list)
            gt_arr = np.array(self.gt_count_list)
            density_max_arr = np.array(self.density_max_list)
            density_mean_arr = np.array(self.density_mean_list)
            nonzero_ratio_arr = np.array(self.density_nonzero_ratio_list)

            loss_logs["train/pred_count_mean"] = pred_arr.mean()
            loss_logs["train/pred_count_std"] = pred_arr.std()
            loss_logs["train/pred_count_min"] = pred_arr.min()
            loss_logs["train/pred_count_max"] = pred_arr.max()
            loss_logs["train/gt_count_mean"] = gt_arr.mean()

            # Check for trivial outputs
            loss_logs["train/density_max_mean"] = density_max_arr.mean()
            loss_logs["train/density_mean_mean"] = density_mean_arr.mean()
            loss_logs["train/density_nonzero_ratio"] = nonzero_ratio_arr.mean()

            # Check if predictions are all zeros or near-constant
            zero_pred_ratio = (pred_arr < 0.01).mean()
            loss_logs["train/zero_pred_ratio"] = zero_pred_ratio

        # Detection Prior specific logs
        if self.detection_count_list:
            loss_logs["train/detection_count_mean"] = np.mean(self.detection_count_list)
            loss_logs["train/detection_count_std"] = np.std(self.detection_count_list)
        if self.prior_weight_list:
            loss_logs["train/prior_weight"] = np.mean(self.prior_weight_list)
        if self.prior_scale_list:
            loss_logs["train/prior_scale"] = np.mean(self.prior_scale_list)
        if self.detection_prior_sum_list:
            loss_logs["train/detection_prior_sum_mean"] = np.mean(self.detection_prior_sum_list)
        if self.raw_density_sum_list:
            loss_logs["train/raw_density_sum_mean"] = np.mean(self.raw_density_sum_list)
        if self.pseudo_density_loss_list:
            loss_logs["train/pseudo_density_loss"] = np.mean(self.pseudo_density_loss_list)
        if self.detection_alignment_loss_list:
            loss_logs["train/detection_alignment_loss"] = np.mean(self.detection_alignment_loss_list)
        if self.consistency_loss_list:
            loss_logs["train/consistency_loss"] = np.mean(self.consistency_loss_list)

        # Alternating training phase
        if self.use_alternating_training:
            loss_logs["train/current_phase"] = 0 if self.current_phase == 'density' else 1

        # Gather metrics across processes
        if hasattr(self, 'accelerator'):
            train_mae = self.accelerator.gather(torch.tensor(self.train_mae, device=self.device))
            train_rmse = self.accelerator.gather(torch.tensor(self.train_rmse, device=self.device))
            train_sample_num = self.accelerator.gather(torch.tensor(self.train_sample_num, device=self.device))

            train_mae = train_mae.sum().item()
            train_rmse = train_rmse.sum().item()
            train_sample_num = train_sample_num.sum().item()

            if train_sample_num > 0:
                train_mae = train_mae / train_sample_num
                train_rmse = (train_rmse / train_sample_num) ** 0.5

                loss_logs["train/mae"] = train_mae
                loss_logs["train/rmse"] = train_rmse

            if self.accelerator.is_main_process:
                wandb.log(loss_logs, step=self.state.global_step)
        else:
            if self.train_sample_num > 0:
                loss_logs["train/mae"] = self.train_mae / self.train_sample_num
                loss_logs["train/rmse"] = (self.train_rmse / self.train_sample_num) ** 0.5
            wandb.log(loss_logs, step=self.state.global_step)

        # Reset tracking
        self.loss_count_list.clear()
        self.loss_uncertainty_list.clear()
        self.train_mae = 0
        self.train_rmse = 0
        self.train_sample_num = 0
        self.pred_count_list.clear()
        self.gt_count_list.clear()
        self.density_max_list.clear()
        self.density_mean_list.clear()
        self.density_nonzero_ratio_list.clear()
        # Detection Prior specific
        self.detection_count_list.clear()
        self.prior_weight_list.clear()
        self.prior_scale_list.clear()
        self.detection_prior_sum_list.clear()
        self.raw_density_sum_list.clear()
        self.pseudo_density_loss_list.clear()
        self.detection_alignment_loss_list.clear()
        self.consistency_loss_list.clear()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", save_qualitative_results=None):
        """
        Evaluate using density-based counting.
        """
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        if save_qualitative_results is None:
            save_qualitative_results = self.save_qualitative_results

        # Initialize metrics
        eval_mae = 0.0
        eval_rmse = 0.0
        total_samples = 0

        # Detection-head accumulators (matched-inference tracking)
        eval_det_mae = 0.0
        eval_det_rmse = 0.0

        # Prediction statistics for evaluation
        eval_pred_counts = []
        eval_gt_counts = []
        eval_density_maxs = []
        eval_density_means = []

        # Detection Prior specific
        eval_detection_counts = []
        eval_prior_weights = []
        eval_prior_scales = []

        # Per-sample results for JSON export
        per_sample_results = []

        # Wandb visualization samples (collect first N samples)
        wandb_vis_samples = []
        max_wandb_samples = 8

        # Qualitative results directory
        if save_qualitative_results:
            global_step = self.state.global_step if hasattr(self.state, 'global_step') else 0
            qual_dir = os.path.join(self.output_dir, f"QualRes_{global_step}_{metric_key_prefix}")
            os.makedirs(qual_dir, exist_ok=True)

        for step, inputs in enumerate(eval_dataloader):
            with torch.no_grad():
                # Prepare inputs
                pos_llm_det_inputs = inputs['pos_llm_det_inputs']
                gt_count = inputs['pos_count']

                pos_llm_det_inputs = pos_llm_det_inputs.to(self.device)
                pos_llm_det_inputs['pixel_values'] = pos_llm_det_inputs['pixel_values'].to(torch.bfloat16)

                if 'pos_exemplars' in inputs:
                    pos_llm_det_inputs['pos_exemplars'] = inputs['pos_exemplars']
                if 'neg_exemplars' in inputs:
                    pos_llm_det_inputs['neg_exemplars'] = inputs['neg_exemplars']

                neg_llm_det_inputs = inputs['neg_llm_det_inputs']
                neg_llm_det_inputs = {k: v.to(self.device) for k, v in neg_llm_det_inputs.items()}
                neg_llm_det_inputs['pixel_values'] = neg_llm_det_inputs['pixel_values'].to(torch.bfloat16)

                pos_llm_det_inputs['neg_token_type_ids'] = neg_llm_det_inputs['token_type_ids']
                pos_llm_det_inputs['neg_attention_mask'] = neg_llm_det_inputs['attention_mask']
                pos_llm_det_inputs['neg_pixel_mask'] = neg_llm_det_inputs['pixel_mask']
                pos_llm_det_inputs['neg_pixel_values'] = neg_llm_det_inputs['pixel_values']
                pos_llm_det_inputs['neg_input_ids'] = neg_llm_det_inputs['input_ids']
                pos_llm_det_inputs['use_neg'] = True

                # Forward pass
                outputs = self.model(**pos_llm_det_inputs)

                # Get prediction
                pred_count = outputs.pred_count.item()
                gt_cnt = float(gt_count)
                density_map = outputs.density_map

                # Calculate error
                cnt_err = abs(pred_count - gt_cnt)
                eval_mae += cnt_err
                eval_rmse += cnt_err ** 2
                total_samples += 1

                # Detection-head count (thresholded decoder confidences)
                if getattr(outputs, 'logits', None) is not None:
                    det_scores = outputs.logits.sigmoid().max(dim=-1)[0][0]
                    det_count = float((det_scores > self.det_eval_threshold).sum().item())
                    det_err = abs(det_count - gt_cnt)
                    eval_det_mae += det_err
                    eval_det_rmse += det_err ** 2

                # Track statistics
                eval_pred_counts.append(pred_count)
                eval_gt_counts.append(gt_cnt)
                eval_density_maxs.append(density_map.max().item())
                eval_density_means.append(density_map.mean().item())

                # Collect per-sample result for JSON export
                sample_result = {
                    'sample_idx': step,
                    'pred_count': pred_count,
                    'gt_count': gt_cnt,
                    'error': pred_count - gt_cnt,
                    'abs_error': cnt_err,
                    'density_max': density_map.max().item(),
                    'density_mean': density_map.mean().item(),
                }
                # Add caption if available
                if 'pos_caption' in inputs:
                    caption = inputs['pos_caption']
                    if isinstance(caption, list):
                        caption = caption[0][0] if isinstance(caption[0], list) else caption[0]
                    sample_result['caption'] = str(caption)
                # Add detection prior sum if available
                if hasattr(outputs, 'detection_prior') and outputs.detection_prior is not None:
                    sample_result['prior_sum'] = outputs.detection_prior.sum().item()
                if hasattr(outputs, 'raw_density') and outputs.raw_density is not None:
                    sample_result['raw_density_sum'] = outputs.raw_density.sum().item()
                per_sample_results.append(sample_result)

                # Detection Prior specific tracking
                detection_prior = None
                pred_boxes = None
                pred_scores = None
                if hasattr(outputs, 'detection_prior') and outputs.detection_prior is not None:
                    detection_prior = outputs.detection_prior.squeeze().cpu().float().numpy()
                    if hasattr(outputs, 'pred_scores') and outputs.pred_scores is not None:
                        num_det = (outputs.pred_scores > 0.3).sum().item()
                        eval_detection_counts.append(num_det)
                        pred_boxes = outputs.pred_boxes.cpu().float().numpy()
                        pred_scores = outputs.pred_scores.cpu().float().numpy()
                    if hasattr(self.model, 'prior_weight'):
                        prior_w = torch.sigmoid(self.model.prior_weight).item()
                        eval_prior_weights.append(prior_w)
                    if hasattr(self.model, 'prior_scale'):
                        prior_s = F.relu(self.model.prior_scale).item()
                        eval_prior_scales.append(prior_s)

                # Collect samples for wandb visualization
                if len(wandb_vis_samples) < max_wandb_samples and 'image' in inputs and inputs['image'] is not None:
                    sample_data = {
                        'image': inputs['image'],
                        'density_map': density_map.squeeze().cpu().float().numpy(),  # Convert bfloat16 to float32
                        'pred_count': pred_count,
                        'gt_count': gt_cnt,
                        'pos_caption': inputs.get('pos_caption', [['unknown']])[0][0] if isinstance(inputs.get('pos_caption', [['unknown']]), list) else inputs.get('pos_caption', 'unknown'),
                        'neg_caption': inputs.get('neg_caption', [['unknown']])[0][0] if isinstance(inputs.get('neg_caption', [['unknown']]), list) else inputs.get('neg_caption', 'unknown'),
                    }
                    # Add detection prior data if available
                    if detection_prior is not None:
                        sample_data['detection_prior'] = detection_prior
                        sample_data['pred_boxes'] = pred_boxes
                        sample_data['pred_scores'] = pred_scores
                    # Add raw_density (density before fusion with prior)
                    if hasattr(outputs, 'raw_density') and outputs.raw_density is not None:
                        sample_data['raw_density'] = outputs.raw_density.squeeze().cpu().float().numpy()
                    wandb_vis_samples.append(sample_data)

                # Save qualitative results
                if save_qualitative_results and 'image' in inputs and inputs['image'] is not None:
                    self._save_qualitative(
                        inputs['image'],
                        outputs.density_map,
                        pred_count,
                        gt_cnt,
                        inputs.get('pos_caption', [''])[0],
                        inputs.get('neg_caption', [''])[0],
                        qual_dir,
                        step
                    )

        # Gather results from all processes
        if hasattr(self, 'accelerator'):
            gathered_mae = self.accelerator.gather(torch.tensor(eval_mae, device=self.device))
            gathered_rmse = self.accelerator.gather(torch.tensor(eval_rmse, device=self.device))
            gathered_total = self.accelerator.gather(torch.tensor(total_samples, device=self.device))

            gathered_det_mae = self.accelerator.gather(torch.tensor(eval_det_mae, device=self.device))
            gathered_det_rmse = self.accelerator.gather(torch.tensor(eval_det_rmse, device=self.device))

            eval_mae = gathered_mae.sum().item()
            eval_rmse = gathered_rmse.sum().item()
            total_samples = gathered_total.sum().item()
            eval_det_mae = gathered_det_mae.sum().item()
            eval_det_rmse = gathered_det_rmse.sum().item()

        # Calculate final metrics
        if total_samples > 0:
            eval_mae = eval_mae / total_samples
            eval_rmse = (eval_rmse / total_samples) ** 0.5
            eval_det_mae = eval_det_mae / total_samples
            eval_det_rmse = (eval_det_rmse / total_samples) ** 0.5

        metrics = {
            f"{metric_key_prefix}/mae": eval_mae,
            f"{metric_key_prefix}/rmse": eval_rmse,
            f"{metric_key_prefix}/det_mae": eval_det_mae,
            f"{metric_key_prefix}/det_rmse": eval_det_rmse,
        }

        # Add prediction statistics
        if eval_pred_counts:
            pred_arr = np.array(eval_pred_counts)
            gt_arr = np.array(eval_gt_counts)
            density_max_arr = np.array(eval_density_maxs)
            density_mean_arr = np.array(eval_density_means)

            metrics[f"{metric_key_prefix}/pred_count_mean"] = pred_arr.mean()
            metrics[f"{metric_key_prefix}/pred_count_std"] = pred_arr.std()
            metrics[f"{metric_key_prefix}/pred_count_min"] = pred_arr.min()
            metrics[f"{metric_key_prefix}/pred_count_max"] = pred_arr.max()
            metrics[f"{metric_key_prefix}/gt_count_mean"] = gt_arr.mean()
            metrics[f"{metric_key_prefix}/density_max_mean"] = density_max_arr.mean()
            metrics[f"{metric_key_prefix}/density_mean_mean"] = density_mean_arr.mean()

            # Check for trivial outputs
            zero_pred_ratio = (pred_arr < 0.01).mean()
            metrics[f"{metric_key_prefix}/zero_pred_ratio"] = zero_pred_ratio

            # Additional analysis metrics
            errors = pred_arr - gt_arr
            abs_errors = np.abs(errors)
            metrics[f"{metric_key_prefix}/error_std"] = abs_errors.std()
            metrics[f"{metric_key_prefix}/over_count_ratio"] = (errors > 0).mean()  # pred > gt
            metrics[f"{metric_key_prefix}/under_count_ratio"] = (errors < 0).mean()  # pred < gt

            # Correlation between pred and gt
            if len(pred_arr) > 1 and pred_arr.std() > 0 and gt_arr.std() > 0:
                correlation = np.corrcoef(pred_arr, gt_arr)[0, 1]
                metrics[f"{metric_key_prefix}/pred_gt_correlation"] = correlation

            # Relative error (for scale-invariant analysis)
            rel_errors = abs_errors / (gt_arr + 1e-6)
            metrics[f"{metric_key_prefix}/relative_error_mean"] = rel_errors.mean()
            metrics[f"{metric_key_prefix}/relative_error_median"] = np.median(rel_errors)

        # Detection Prior specific metrics
        if eval_detection_counts:
            metrics[f"{metric_key_prefix}/detection_count_mean"] = np.mean(eval_detection_counts)
            metrics[f"{metric_key_prefix}/detection_count_std"] = np.std(eval_detection_counts)
        if eval_prior_weights:
            metrics[f"{metric_key_prefix}/prior_weight"] = np.mean(eval_prior_weights)
        if eval_prior_scales:
            metrics[f"{metric_key_prefix}/prior_scale"] = np.mean(eval_prior_scales)

        # Save per-sample results to JSON (each rank saves its own file)
        if per_sample_results:
            global_step = self.state.global_step if hasattr(self.state, 'global_step') else 0
            # Get rank for filename
            rank = 0
            if hasattr(self, 'accelerator'):
                rank = self.accelerator.process_index
            elif torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()

            # Save to src/eval_predictions for easy access
            src_dir = os.path.dirname(os.path.abspath(__file__))
            json_dir = os.path.join(src_dir, "eval_predictions")
            os.makedirs(json_dir, exist_ok=True)
            json_path = os.path.join(json_dir, f"{metric_key_prefix}_step{global_step}_rank{rank}.json")

            import json
            with open(json_path, 'w') as f:
                json.dump({'global_step': global_step, 'rank': rank, 'prefix': metric_key_prefix,
                           'mae': eval_mae, 'rmse': eval_rmse, 'samples': per_sample_results}, f)
            print(f"  Saved {len(per_sample_results)} predictions to {json_path}")

        # Log to wandb
        is_main = (hasattr(self, 'accelerator') and self.accelerator.is_main_process) or not hasattr(self, 'accelerator')
        if is_main:
            wandb.log(metrics, step=self.state.global_step)

            # Log visualizations to wandb
            if wandb_vis_samples:
                self._log_wandb_visualizations(wandb_vis_samples, metric_key_prefix)

        # Save best model
        if metric_key_prefix == "val" and eval_mae < self.best_val_mae:
            self.best_val_mae = eval_mae
            self.save_model(os.path.join(self.output_dir, "best_val_model"), _internal_call=True)

        # Save best model according to the DETECTION head (used by the decoder-only control)
        if self.save_best_det and metric_key_prefix == "val" and eval_det_mae < self.best_val_det_mae:
            self.best_val_det_mae = eval_det_mae
            self.save_model(os.path.join(self.output_dir, "best_val_det_model"), _internal_call=True)

        print(f"\n{metric_key_prefix.upper()} Results: MAE={eval_mae:.4f}, RMSE={eval_rmse:.4f} | det MAE={eval_det_mae:.4f}, det RMSE={eval_det_rmse:.4f}")
        if eval_pred_counts:
            print(f"  Pred stats: mean={np.mean(eval_pred_counts):.2f}, min={np.min(eval_pred_counts):.2f}, max={np.max(eval_pred_counts):.2f}")
            print(f"  Density stats: max_mean={np.mean(eval_density_maxs):.4f}, mean_mean={np.mean(eval_density_means):.6f}")

        self.model.train()
        return metrics

    def _log_wandb_visualizations(self, samples, prefix):
        """Log density map visualizations to wandb by saving to temp files.

        For Detection Prior model, shows:
        1. Original image with detection boxes
        2. Detection prior map
        3. Final fused density map
        4. Overlay of density on image
        """
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.patches as patches
        matplotlib.use('Agg')
        from io import BytesIO

        # Save to temp directory
        vis_dir = "${EXP_ROOT:-./experiments}/tmp_vis"
        os.makedirs(vis_dir, exist_ok=True)

        wandb_images = []
        for i, sample in enumerate(samples):
            # Check if this is a Detection Prior model output
            has_prior = 'detection_prior' in sample and sample['detection_prior'] is not None
            has_raw_density = 'raw_density' in sample and sample['raw_density'] is not None

            if has_prior:
                # 6-panel visualization for Detection Prior model (includes raw_density)
                num_panels = 6 if has_raw_density else 5
                fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 5))

                # 1. Original image with detection boxes
                axes[0].imshow(sample['image'])
                if sample.get('pred_boxes') is not None and sample.get('pred_scores') is not None:
                    img_w, img_h = sample['image'].size
                    pred_boxes = sample['pred_boxes'][0]  # [N, 4] cxcywh normalized
                    pred_scores = sample['pred_scores'][0]  # [N]
                    num_boxes = 0
                    for box, score in zip(pred_boxes, pred_scores):
                        if score > 0.3:
                            cx, cy, w, h = box
                            # Convert to pixel coordinates
                            x1 = (cx - w/2) * img_w
                            y1 = (cy - h/2) * img_h
                            box_w = w * img_w
                            box_h = h * img_h
                            rect = patches.Rectangle((x1, y1), box_w, box_h,
                                                     linewidth=2, edgecolor='lime', facecolor='none')
                            axes[0].add_patch(rect)
                            num_boxes += 1
                    axes[0].set_title(f"Detections ({num_boxes} boxes)")
                else:
                    axes[0].set_title(f"Pos: {sample['pos_caption'][:20]}...")
                axes[0].axis('off')

                # 2. Detection prior map (pseudo density from frozen detector)
                prior = sample['detection_prior']
                im1 = axes[1].imshow(prior, cmap='hot')
                axes[1].set_title(f"Detection Prior (sum={prior.sum():.1f})")
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                # Panel index tracker
                panel_idx = 2

                # 3. Raw density (learned density head output, before fusion)
                if has_raw_density:
                    raw_density = sample['raw_density']
                    im_raw = axes[panel_idx].imshow(raw_density, cmap='jet')
                    axes[panel_idx].set_title(f"Raw Density (sum={raw_density.sum():.1f})")
                    axes[panel_idx].axis('off')
                    plt.colorbar(im_raw, ax=axes[panel_idx], fraction=0.046)
                    panel_idx += 1

                # 4. Final fused density map
                density = sample['density_map']
                im2 = axes[panel_idx].imshow(density, cmap='jet')
                axes[panel_idx].set_title(f"Fused Density (sum={density.sum():.1f})")
                axes[panel_idx].axis('off')
                plt.colorbar(im2, ax=axes[panel_idx], fraction=0.046)
                panel_idx += 1

                # 5. Overlay on image
                img_resized = sample['image'].resize((density.shape[1], density.shape[0]))
                axes[panel_idx].imshow(img_resized)
                axes[panel_idx].imshow(density, cmap='jet', alpha=0.5)
                axes[panel_idx].set_title(f"Pred: {sample['pred_count']:.1f}, GT: {sample['gt_count']:.0f}")
                axes[panel_idx].axis('off')
                panel_idx += 1

                # 6. Side-by-side prior vs raw density comparison
                axes[panel_idx].imshow(prior, cmap='hot', alpha=0.5)
                if has_raw_density:
                    axes[panel_idx].imshow(raw_density, cmap='jet', alpha=0.5)
                    axes[panel_idx].set_title("Prior (hot) + Raw Density (jet)")
                else:
                    axes[panel_idx].imshow(density, cmap='jet', alpha=0.5)
                    axes[panel_idx].set_title("Prior (hot) + Density (jet)")
                axes[panel_idx].axis('off')

            else:
                # Standard 3-panel visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                axes[0].imshow(sample['image'])
                axes[0].set_title(f"Pos: {sample['pos_caption'][:30]}...")
                axes[0].axis('off')

                # Density map
                density = sample['density_map']
                im = axes[1].imshow(density, cmap='jet')
                axes[1].set_title(f"Density (sum={density.sum():.1f})")
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046)

                # Overlay
                img_resized = sample['image'].resize((density.shape[1], density.shape[0]))
                axes[2].imshow(img_resized)
                axes[2].imshow(density, cmap='jet', alpha=0.5)
                axes[2].set_title(f"Pred: {sample['pred_count']:.1f}, GT: {sample['gt_count']:.0f}")
                axes[2].axis('off')

            plt.tight_layout()

            # Save to file and load for wandb
            step = self.state.global_step if hasattr(self.state, 'global_step') else 0
            filepath = os.path.join(vis_dir, f"{prefix}_step{step}_sample{i}.png")
            plt.savefig(filepath, bbox_inches='tight', dpi=100)
            plt.close(fig)

            # Load and add to wandb
            wandb_images.append(wandb.Image(filepath, caption=f"GT={sample['gt_count']:.0f}, Pred={sample['pred_count']:.1f}"))

        wandb.log({f"{prefix}/density_samples": wandb_images}, step=self.state.global_step)

    def _save_qualitative(self, image, density_map, pred_count, gt_count, pos_caption, neg_caption, save_dir, step):
        """Save qualitative visualization."""
        import matplotlib.pyplot as plt

        # Create figure with image and density map
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f"Pos: {pos_caption}\nNeg: {neg_caption}")
        axes[0].axis('off')

        # Density map
        density = density_map.squeeze().cpu().float().numpy()  # Convert bfloat16 to float32
        im = axes[1].imshow(density, cmap='jet')
        axes[1].set_title(f"Pred: {pred_count:.1f}, GT: {gt_count:.0f}")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        # Save
        filename = f"step_{step}_pred_{pred_count:.1f}_gt_{gt_count:.0f}.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=100)
        plt.close()

    def compute_metrics(self, eval_preds):
        """Not used since we override evaluate()."""
        return {}