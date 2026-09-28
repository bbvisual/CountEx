import torch
import wandb
import numpy as np
import os
import random
from transformers import Trainer
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from accelerate import Accelerator
from utils import prepare_targets, post_process_grounded_object_detection, generate_pseudo_density_map
from utils import supcon_pos_neg, filter_overlap, extract_pos_tokens_single, build_point_count_map
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from utils import interval_huber, relative_huber


class FineGrainedCountingTrainer(Trainer):
    """
    Custom trainer for fine-grained counting task with support for negative prompts.
    """
    
    def __init__(self, *args, criterion=None, llmdet_processor=None, save_qualitative_results=False, **kwargs):
        self.val_dataset = kwargs.pop('val_dataset')
        self.test_dataset = kwargs.pop('test_dataset')
        self.output_dir = kwargs.pop('output_dir')
        self.density_loss_weight = kwargs.pop('density_loss_weight', 200)
        self.use_contrastive_loss = kwargs.pop('use_contrastive_loss', False)
        self.contrastive_loss_weight = kwargs.pop('contrastive_loss_weight', 0.01)
        self.contrastive_temperature = kwargs.pop('contrastive_temperature', 0.25)
        self.contrastive_scales = kwargs.pop('contrastive_scales', '0')
        self.contrastive_type = kwargs.pop('contrastive_type', 'feature')
        self.weakly_supervised_loss = kwargs.pop('weakly_supervised_loss', 'interval_huber')
        self.use_uncertainty_loss = kwargs.pop('use_uncertainty_loss', False)

        # Parse contrastive scales
        if self.contrastive_scales == 'all':
            self.contrastive_scale_list = [0, 1, 2, 3]
        else:
            self.contrastive_scale_list = [int(s) for s in self.contrastive_scales.split(',')]

        # Validate contrastive type
        assert self.contrastive_type in ['feature', 'query', 'prototype'], \
            f"Invalid contrastive_type: {self.contrastive_type}. Must be 'feature', 'query', or 'prototype'"
        self.best_val_mae = 1e10
        self.best_test_mae = 1e10
        # 50% not using negative captions during training
        self.use_neg_prob = kwargs.pop('use_neg_prob', None)
        if self.use_neg_prob is not None:
            assert 0 <= self.use_neg_prob <= 1, "Invalid use_neg_prob"
            self.use_neg_aug = True
        else:
            self.use_neg_aug = False
        
        assert self.weakly_supervised_loss in ['interval_huber', 'ae', 'relative_huber'], "Invalid weakly supervised loss"

        super().__init__(*args, **kwargs)
        self.criterion = criterion
        self.llmdet_processor = llmdet_processor
        self.device = self.args.device if hasattr(self.args, 'device') else 'cuda'
        self.save_qualitative_results = save_qualitative_results
        
        # Initialize loss tracking lists
        self.loss_label_list = []
        self.loss_point_list = []
        self.loss_density_list = []
        self.loss_contrastive_list = []
        self.weakly_supervised_loss_list = []
        self.loss_extra_list = []
        self.fusion_scale_list = []
        # Gate statistics for SimGate/AdaGate models
        self.gate_mean_list = []
        self.gate_std_list = []
        self.train_mae = 0
        self.train_rmse = 0
        self.train_mae_anno = 0
        self.train_rmse_anno = 0
        self.train_sample_num = 0
        self.eval_sample_num = 0
        # FIXME: contrastive loss weight is hard-coded here
        
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss for fine-grained counting task.
        
        Args:
            model: The model to compute loss for
            inputs: Dictionary containing batch data
            return_outputs: Whether to return model outputs along with loss
            
        Returns:
            loss: The computed loss
            outputs: Model outputs (if return_outputs=True)
        """
        # Extract inputs
        pos_llm_det_inputs = inputs['pos_llm_det_inputs']
        pos_caption = inputs['pos_caption']
        shapes = inputs['shapes']
        pos_points = inputs['pos_points']
        # pos_exemplars = inputs['pos_exemplars']
        # neg_exemplars = inputs['neg_exemplars']
        pos_count = inputs['pos_count']
        assert inputs['type'] in ['dot_anno', 'weak_supervised', 'dot_anno_pseudo'], "Invalid data type"
        if inputs['type'] == 'dot_anno' or inputs['type'] == 'dot_anno_pseudo':
            if inputs['type'] == 'dot_anno':
                dot_anno_overall_weight = 1.0
            else:
                dot_anno_overall_weight = 0.25
            dot_anno = True
        else:
            dot_anno = False
        annotated_pos_count = inputs['annotated_pos_count']
        
        # Move inputs to device
        pos_llm_det_inputs = pos_llm_det_inputs.to(self.device)
        if 'pos_exemplars' in inputs:
            pos_llm_det_inputs['pos_exemplars'] = inputs['pos_exemplars']
        if 'neg_exemplars' in inputs:
            pos_llm_det_inputs['neg_exemplars'] = inputs['neg_exemplars']
        pos_llm_det_inputs['pixel_values'] = pos_llm_det_inputs['pixel_values'].to(torch.bfloat16)
        neg_llm_det_inputs = inputs['neg_llm_det_inputs']
        neg_llm_det_inputs = {k: v.to(self.device) for k, v in neg_llm_det_inputs.items()}
        neg_llm_det_inputs['pixel_values'] = neg_llm_det_inputs['pixel_values'].to(torch.bfloat16)
        pos_llm_det_inputs['neg_token_type_ids'] = neg_llm_det_inputs['token_type_ids']
        pos_llm_det_inputs['neg_attention_mask'] = neg_llm_det_inputs['attention_mask']
        pos_llm_det_inputs['neg_pixel_mask'] = neg_llm_det_inputs['pixel_mask']
        pos_llm_det_inputs['neg_pixel_values'] = neg_llm_det_inputs['pixel_values']
        pos_llm_det_inputs['neg_input_ids'] = neg_llm_det_inputs['input_ids']
        # sync sample on rank0 and make all ranks consistent
        from torch import distributed as dist
        if dist.get_rank() == 0:
            use_neg = float(random.random() <= self.use_neg_prob)
        else:
            use_neg = 0.0
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        use_neg_tensor = torch.tensor([use_neg], dtype=torch.float32, device=self.device)
        dist.broadcast(use_neg_tensor, src=0)
        use_neg = bool(use_neg_tensor.item())
        pos_llm_det_inputs['use_neg'] = use_neg
        outputs = model(**pos_llm_det_inputs)
        
        # Prepare outputs for loss computation
        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]
        outputs["pred_logits"] = outputs["logits"]
        if 'extra_loss' in outputs:
            extra_loss = outputs['extra_loss'] * 0.01
            if 'fusion_scale' in outputs['extra_logs']:
                fusion_scale = outputs['extra_logs']['fusion_scale']
            else:
                fusion_scale = 0.0
            # Extract gate statistics for SimGate/AdaGate models
            if 'gate_mean' in outputs['extra_logs']:
                gate_mean = outputs['extra_logs']['gate_mean']
                gate_mean = gate_mean.item() if isinstance(gate_mean, torch.Tensor) else gate_mean
                self.gate_mean_list.append(gate_mean)
            if 'gate_std' in outputs['extra_logs']:
                gate_std = outputs['extra_logs']['gate_std']
                gate_std = gate_std.item() if isinstance(gate_std, torch.Tensor) else gate_std
                self.gate_std_list.append(gate_std)
        else:
            extra_loss = 0.0
            fusion_scale = 0.0
        if isinstance(fusion_scale, torch.Tensor):
            self.fusion_scale_list.append(fusion_scale.item())
        else:
            self.fusion_scale_list.append(fusion_scale)
        if isinstance(extra_loss, torch.Tensor):
            self.loss_extra_list.append(extra_loss.item())
        else:
            self.loss_extra_list.append(extra_loss)
        
        # Prepare targets
        emb_size = outputs["pred_logits"].shape[2]
        caption = pos_caption[0]
        targets = prepare_targets(pos_points, caption, shapes, emb_size, self.device, self.llmdet_processor)
        
        # Compute loss using criterion
        # For weakly-supervised training, we only use the density map branch
        # FIXME: Hard code for supervised training
        if dot_anno:
            # weighting with the dot_anno
            # if dot_anno : 1.0
            # if pseudo dot_anno : 0.25
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss = loss * dot_anno_overall_weight
            results = post_process_grounded_object_detection(outputs, box_threshold=0.42)[0]
            boxes = results["boxes"]
            boxes = [box.tolist() for box in boxes]
            points = [[box[0], box[1]] for box in boxes]
            pred_cnt = len(points)
            gt_cnt = pos_count
            anno_cnt = annotated_pos_count
            gt_cnt = int(gt_cnt)
            anno_cnt = int(anno_cnt)
            cnt_err = abs(pred_cnt - gt_cnt)
            cnt_err_anno = abs(pred_cnt - anno_cnt)
            self.train_mae += cnt_err
            self.train_mae_anno += cnt_err_anno
            self.train_rmse += cnt_err ** 2
            self.train_rmse_anno += cnt_err_anno ** 2
            self.train_sample_num += 1
        
            # Record individual losses for logging
            if 'loss_label' in loss_dict and 'loss_label' in weight_dict:
                self.loss_label_list.append(loss_dict['loss_label'].item() * weight_dict['loss_label'])
            if 'loss_point' in loss_dict and 'loss_point' in weight_dict:
                self.loss_point_list.append(loss_dict['loss_point'].item() * weight_dict['loss_point'])

            if outputs.density_map_pred is not None:
                density_map_pred = outputs.density_map_pred
                H, W = density_map_pred.shape[-2:]
                
                if len(pos_points[0]) > 0 and len(pos_points[0]) <= 350:
                    pos_points_normalized = [np.array(pos_points) / np.array(shapes)[::-1]]
                    pos_points_normalized[0] = pos_points_normalized[0].squeeze(0)
                    pos_points_normalized = [torch.from_numpy(img_points).float() for img_points in pos_points_normalized]
                    pseudo_density = generate_pseudo_density_map(
                        pos_points_normalized[0].to(density_map_pred.device),
                        (H, W),
                        sigma=6.0,          
                        normalize=True,
                    )
                elif len(pos_points[0]) > 350:
                    points_1 = [pos_points[0][:350]]
                    points_2 = [pos_points[0][350:]]
                    points_1_normalized = [np.array(points_1) / np.array(shapes)[::-1]]
                    points_1_normalized[0] = points_1_normalized[0].squeeze(0)
                    points_1_normalized = [torch.from_numpy(img_points).float() for img_points in points_1_normalized]
                    points_2_normalized = [np.array(points_2) / np.array(shapes)[::-1]]
                    points_2_normalized[0] = points_2_normalized[0].squeeze(0)
                    points_2_normalized = [torch.from_numpy(img_points).float() for img_points in points_2_normalized]
                    pseudo_density_1 = generate_pseudo_density_map(
                        points_1_normalized[0].to(density_map_pred.device),
                        (H, W),
                        sigma=6.0,          
                        normalize=True,
                    )
                    pseudo_density_2 = generate_pseudo_density_map(
                        points_2_normalized[0].to(density_map_pred.device),
                        (H, W),
                        sigma=6.0,          
                        normalize=True,
                    )
                    pseudo_density = pseudo_density_1 + pseudo_density_2
                else:
                    pseudo_density = torch.zeros_like(density_map_pred)
                
                assert pseudo_density.shape == density_map_pred.shape
                pseudo_density = pseudo_density.to(density_map_pred.device)
                pseudo_density = pseudo_density.to(torch.bfloat16)
                
                loss_density = F.mse_loss(density_map_pred, pseudo_density) * self.density_loss_weight
                loss_density = loss_density * dot_anno_overall_weight
                self.loss_density_list.append(loss_density.item())
                loss += loss_density

                import gc
                del density_map_pred, pseudo_density
                torch.cuda.empty_cache()
                gc.collect()
                
            # print(f"loss_density: {loss_density.item()}")
        else:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss = loss * 0.0
            annotated_pos_count = inputs['pos_count']
            if outputs.density_map_pred is not None:
                density_map_pred = outputs.density_map_pred
                density_map_pred_count = density_map_pred.sum()
                annotated_pos_count = torch.tensor(annotated_pos_count, dtype=density_map_pred_count.dtype).to(self.device)

                # Check if model outputs uncertainty (log_var)
                has_uncertainty = hasattr(outputs, 'density_log_var') and outputs.density_log_var is not None

                if self.use_uncertainty_loss and has_uncertainty:
                    # Uncertainty-weighted loss (Kendall et al. NIPS 2017)
                    # loss = 0.5 * exp(-log_var) * base_loss + 0.5 * log_var
                    density_log_var = outputs.density_log_var

                    # Compute base loss (count-level)
                    if self.weakly_supervised_loss == 'interval_huber':
                        base_loss = interval_huber(density_map_pred_count, annotated_pos_count)
                    elif self.weakly_supervised_loss == 'ae':
                        base_loss = F.l1_loss(density_map_pred_count, annotated_pos_count)
                    elif self.weakly_supervised_loss == 'relative_huber':
                        base_loss = relative_huber(density_map_pred_count, annotated_pos_count)
                    else:
                        raise ValueError(f"Invalid weakly supervised loss: {self.weakly_supervised_loss}")

                    # Average log_var across spatial dimensions for count-level uncertainty
                    avg_log_var = density_log_var.mean()

                    # Uncertainty-weighted loss
                    # exp(-log_var) = 1/variance, so high uncertainty -> low weight
                    precision = torch.exp(-avg_log_var)
                    weakly_supervised_loss = 0.5 * precision * base_loss + 0.5 * avg_log_var
                    weakly_supervised_loss = weakly_supervised_loss * 0.001
                else:
                    # Standard loss without uncertainty
                    if self.weakly_supervised_loss == 'interval_huber':
                        weakly_supervised_loss = interval_huber(density_map_pred_count, annotated_pos_count) * 0.001
                    elif self.weakly_supervised_loss == 'ae':
                        weakly_supervised_loss = F.l1_loss(density_map_pred_count, annotated_pos_count) * 0.001
                    elif self.weakly_supervised_loss == 'relative_huber':
                        weakly_supervised_loss = relative_huber(density_map_pred_count, annotated_pos_count) * 0.001
                    else:
                        raise ValueError(f"Invalid weakly supervised loss: {self.weakly_supervised_loss}")

                self.weakly_supervised_loss_list.append(weakly_supervised_loss.item())
                loss += weakly_supervised_loss
            else:
                loss += 0
            # raise ValueError("Not implemented")

        # add extra loss
        if extra_loss is not None:
            loss += extra_loss
                
        if self.use_contrastive_loss:
            loss_cl = None

            if self.contrastive_type == 'feature':
                # L1: Feature-level contrastive loss (multi-scale)
                neg_points = inputs['neg_points']
                positive_feature_maps = outputs.positive_feature_maps
                negative_feature_maps = outputs.negative_feature_maps

                pos_points_normalized = [np.array(pos_points) / np.array(shapes)[::-1]]
                pos_points_normalized[0] = pos_points_normalized[0].squeeze(0)
                pos_points_normalized = [torch.from_numpy(img_points).float() for img_points in pos_points_normalized]

                neg_points_normalized = [np.array(neg_points) / np.array(shapes)[::-1]]
                neg_points_normalized[0] = neg_points_normalized[0].squeeze(0)
                neg_points_normalized = [torch.from_numpy(img_points).float() for img_points in neg_points_normalized]

                loss_cl_total = 0.0
                num_valid_scales = 0
                for scale_idx in self.contrastive_scale_list:
                    if scale_idx >= len(positive_feature_maps):
                        continue
                    pos_fm = positive_feature_maps[scale_idx]
                    neg_fm = negative_feature_maps[scale_idx]

                    point_count_map = build_point_count_map(pos_fm, pos_points_normalized)
                    point_count_map_neg = build_point_count_map(neg_fm, neg_points_normalized)

                    pos_tokens, lin_index_pos = extract_pos_tokens_single(pos_fm, point_count_map)
                    neg_tokens, lin_index_neg = extract_pos_tokens_single(neg_fm, point_count_map_neg)

                    pos_tokens, neg_tokens = filter_overlap(pos_tokens, lin_index_pos,
                                                            neg_tokens, lin_index_neg)

                    if pos_tokens.numel() > 0 and neg_tokens.numel() > 0:
                        loss_cl_scale = supcon_pos_neg(pos_tokens, neg_tokens, temperature=self.contrastive_temperature)
                        loss_cl_total += loss_cl_scale
                        num_valid_scales += 1

                if num_valid_scales > 0:
                    loss_cl = (loss_cl_total / num_valid_scales) * self.contrastive_loss_weight

            elif self.contrastive_type == 'query':
                # L2: Query-level contrastive loss
                # Use decoder queries (detected objects) for contrastive learning
                pos_queries = outputs.pos_queries  # [1, num_levels, num_queries, D]
                neg_queries = outputs.neg_queries  # [1, num_levels, num_queries, D]
                pos_logits = outputs.logits  # [B, num_queries, vocab]
                neg_logits = outputs.neg_logits  # [B, num_queries, vocab]

                # Use last level queries
                pos_q = pos_queries[:, -1, :, :].squeeze(0)  # [num_queries, D]
                neg_q = neg_queries[:, -1, :, :].squeeze(0)  # [num_queries, D]

                # Get detection scores
                pos_scores = torch.sigmoid(pos_logits).max(dim=-1)[0].squeeze(0)  # [num_queries]
                neg_scores = torch.sigmoid(neg_logits).max(dim=-1)[0].squeeze(0)  # [num_queries]

                # Select high-confidence queries as positives/negatives
                pos_threshold = 0.3
                neg_threshold = 0.3
                pos_mask = pos_scores > pos_threshold
                neg_mask = neg_scores > neg_threshold

                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    pos_tokens = pos_q[pos_mask]
                    neg_tokens = neg_q[neg_mask]
                    loss_cl = supcon_pos_neg(pos_tokens, neg_tokens, temperature=self.contrastive_temperature) * self.contrastive_loss_weight

            elif self.contrastive_type == 'prototype':
                # L3: Prototype-guided contrastive loss
                # Use common prototypes from QuerySideNeg to guide contrastive learning
                neg_points = inputs['neg_points']
                positive_feature_maps = outputs.positive_feature_maps
                negative_feature_maps = outputs.negative_feature_maps

                pos_points_normalized = [np.array(pos_points) / np.array(shapes)[::-1]]
                pos_points_normalized[0] = pos_points_normalized[0].squeeze(0)
                pos_points_normalized = [torch.from_numpy(img_points).float() for img_points in pos_points_normalized]

                neg_points_normalized = [np.array(neg_points) / np.array(shapes)[::-1]]
                neg_points_normalized[0] = neg_points_normalized[0].squeeze(0)
                neg_points_normalized = [torch.from_numpy(img_points).float() for img_points in neg_points_normalized]

                # Get tokens from feature maps
                pos_fm = positive_feature_maps[0]
                neg_fm = negative_feature_maps[0]

                point_count_map = build_point_count_map(pos_fm, pos_points_normalized)
                point_count_map_neg = build_point_count_map(neg_fm, neg_points_normalized)

                pos_tokens, lin_index_pos = extract_pos_tokens_single(pos_fm, point_count_map)
                neg_tokens, lin_index_neg = extract_pos_tokens_single(neg_fm, point_count_map_neg)

                pos_tokens, neg_tokens = filter_overlap(pos_tokens, lin_index_pos,
                                                        neg_tokens, lin_index_neg)

                if pos_tokens.numel() > 0 and neg_tokens.numel() > 0:
                    # Standard contrastive
                    loss_cl_base = supcon_pos_neg(pos_tokens, neg_tokens, temperature=self.contrastive_temperature)

                    # Prototype-guided: encourage pos/neg to be far from common prototypes
                    # Try to get prototypes from model if available
                    loss_cl_proto = 0.0
                    if hasattr(self.model, 'query_side_neg_pipeline') and hasattr(self.model.query_side_neg_pipeline, 'common'):
                        common_proto = self.model.query_side_neg_pipeline.common.proto  # [r, D]
                        common_proto = F.normalize(common_proto, dim=-1)
                        pos_tokens_norm = F.normalize(pos_tokens, dim=-1)
                        neg_tokens_norm = F.normalize(neg_tokens, dim=-1)

                        # pos and neg should be separable (margin loss)
                        pos_neg_sim = torch.einsum('pd,nd->pn', pos_tokens_norm, neg_tokens_norm)
                        margin = 0.3
                        loss_sep = F.relu(pos_neg_sim - margin).mean()

                        # Both should be away from common (common is shared, not discriminative)
                        pos_common_sim = torch.einsum('pd,rd->pr', pos_tokens_norm, common_proto).max(dim=-1)[0]
                        neg_common_sim = torch.einsum('nd,rd->nr', neg_tokens_norm, common_proto).max(dim=-1)[0]
                        loss_excl = (pos_common_sim.mean() + neg_common_sim.mean()) * 0.5

                        loss_cl_proto = loss_sep + 0.1 * loss_excl

                    loss_cl = (loss_cl_base + loss_cl_proto) * self.contrastive_loss_weight

            if loss_cl is not None:
                self.loss_contrastive_list.append(loss_cl.item())
                loss += loss_cl

        
        # Check if we should run evaluation
        if hasattr(self.state, 'global_step') and hasattr(self.args, 'eval_steps'):
            if self.state.global_step % self.args.eval_steps == 0 and self.state.global_step > 0:
                for dataset, prefix in [(self.val_dataset, "val"), (self.test_dataset, "test")]:
                    eval_metrics = self.evaluate(eval_dataset=dataset, metric_key_prefix=prefix, save_qualitative_results=self.save_qualitative_results)
        
        if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step > 0:
            if len(self.loss_label_list) > 0:
                avg_loss_label = sum(self.loss_label_list) / len(self.loss_label_list)
            else:
                avg_loss_label = 0.0
            if len(self.loss_point_list) > 0:
                avg_loss_point = sum(self.loss_point_list) / len(self.loss_point_list)
            else:
                avg_loss_point = 0.0
            if len(self.loss_contrastive_list) > 0:
                avg_loss_contrastive = sum(self.loss_contrastive_list) / len(self.loss_contrastive_list)
            else:
                avg_loss_contrastive = 0.0
            if len(self.loss_density_list) > 0:
                avg_loss_density = sum(self.loss_density_list) / len(self.loss_density_list)
            else:
                avg_loss_density = 0.0
            if len(self.loss_extra_list) > 0:
                avg_loss_extra = sum(self.loss_extra_list) / len(self.loss_extra_list)
            else:
                avg_loss_extra = 0.0
            if len(self.fusion_scale_list) > 0:
                avg_fusion_scale = sum(self.fusion_scale_list) / len(self.fusion_scale_list)
            else:
                avg_fusion_scale = 0.0
            if len(self.weakly_supervised_loss_list) > 0:
                avg_weakly_supervised_loss = sum(self.weakly_supervised_loss_list) / len(self.weakly_supervised_loss_list)
            else:
                avg_weakly_supervised_loss = 0.0
            loss_logs = {}
            loss_logs.update({
                "train/loss_label": avg_loss_label,
                "train/loss_point": avg_loss_point,
            })     
            if avg_loss_contrastive is not 0.0:
                loss_logs.update({
                    "train/loss_contrastive": avg_loss_contrastive,
                })
            if avg_loss_density is not 0.0:
                loss_logs.update({
                    "train/loss_density": avg_loss_density,
                })
            if avg_weakly_supervised_loss is not None:
                loss_logs.update({
                    "train/weakly_supervised_loss": avg_weakly_supervised_loss,
                })
            if avg_loss_extra is not None:
                loss_logs.update({
                    "train/loss_extra": avg_loss_extra,
                })
            if avg_fusion_scale is not None:
                loss_logs.update({
                    "train/fusion_scale": avg_fusion_scale,
                })
            # Gate statistics for SimGate/AdaGate models
            if len(self.gate_mean_list) > 0:
                avg_gate_mean = sum(self.gate_mean_list) / len(self.gate_mean_list)
                loss_logs.update({"train/gate_mean": avg_gate_mean})
            if len(self.gate_std_list) > 0:
                avg_gate_std = sum(self.gate_std_list) / len(self.gate_std_list)
                loss_logs.update({"train/gate_std": avg_gate_std})
            # Clear the lists after logging
            self.loss_label_list.clear()
            self.loss_point_list.clear()
            self.gate_mean_list.clear()
            self.gate_std_list.clear()
            if self.accelerator.is_main_process:
                wandb.log(loss_logs, step=self.state.global_step)
            
            train_mae = self.accelerator.gather(torch.tensor(self.train_mae, device=self.device))
            train_rmse = self.accelerator.gather(torch.tensor(self.train_rmse, device=self.device))
            train_mae_anno = self.accelerator.gather(torch.tensor(self.train_mae_anno, device=self.device))
            train_rmse_anno = self.accelerator.gather(torch.tensor(self.train_rmse_anno, device=self.device))
            train_sample_num = self.accelerator.gather(torch.tensor(self.train_sample_num, device=self.device))
            
            train_mae = train_mae.sum().item()
            train_mae_anno = train_mae_anno.sum().item()
            train_rmse = train_rmse.sum().item()
            train_rmse_anno = train_rmse_anno.sum().item()
            train_sample_num = train_sample_num.sum().item()
            train_mae = train_mae / train_sample_num
            train_mae_anno = train_mae_anno / train_sample_num
            train_rmse = train_rmse / train_sample_num
            train_rmse_anno = train_rmse_anno / train_sample_num
            train_rmse = train_rmse ** 0.5
            train_rmse_anno = train_rmse_anno ** 0.5

            if self.accelerator.is_main_process:
                wandb.log({
                    "train/mae": train_mae,
                    "train/rmse": train_rmse,
                    "train/mae_anno": train_mae_anno,
                    "train/rmse_anno": train_rmse_anno,
                }, step=self.state.global_step)
            self.train_mae = 0
            self.train_rmse = 0
            self.train_mae_anno = 0
            self.train_rmse_anno = 0
            self.train_sample_num = 0
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", save_qualitative_results=None):
        """
        Override evaluate method to handle custom evaluation for fine-grained counting.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Use default eval dataset if none provided
        eval_dataset = eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Determine whether to save qualitative results
        if save_qualitative_results is None:
            save_qualitative_results = self.save_qualitative_results
        
        # Initialize metrics
        eval_mae = 0.0
        eval_rmse = 0.0
        total_samples = 0
        
        # Create qualitative results directory
        if save_qualitative_results:
            global_step = self.state.global_step if hasattr(self.state, 'global_step') else 0
            qual_img_save_root = os.path.join(self.output_dir, f"QualRes_{global_step}_{metric_key_prefix}")
            if not os.path.exists(qual_img_save_root):
                os.makedirs(qual_img_save_root, exist_ok=True)
        
        # Run evaluation
        for step, inputs in enumerate(eval_dataloader):
            with torch.no_grad():
                # Extract inputs
                pos_llm_det_inputs = inputs['pos_llm_det_inputs']
                pos_caption = inputs['pos_caption']
                shapes = inputs['shapes']
                pos_points = inputs['pos_points']
                pos_count = inputs['pos_count']
                annotated_pos_count = inputs['annotated_pos_count']
                
                # Move inputs to device
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
                pos_llm_det_inputs['use_neg'] = True  # Enable negative prompt during evaluation
                outputs = self.model(**pos_llm_det_inputs)
                
                # Post-process outputs
                outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]
                outputs["pred_logits"] = outputs["logits"]

                if outputs.density_map_pred is not None:
                    density_map_pred = outputs.density_map_pred
                    density_map_pred = density_map_pred.sum()
                    density_map_cnt_err = abs(density_map_pred - pos_count)
                    
                
                results = post_process_grounded_object_detection(outputs, box_threshold=0.42)[0]
                boxes = results["boxes"]
                boxes = [box.tolist() for box in boxes]
                points = [[box[0], box[1]] for box in boxes]
                
                # Calculate metrics
                pred_cnt = len(points)
                gt_cnt = int(pos_count)
                cnt_err = abs(pred_cnt - gt_cnt)
                eval_mae += cnt_err
                eval_rmse += cnt_err ** 2
                total_samples += 1
                
                # Save qualitative results
                if 'image' in inputs and inputs['image'] is not None and save_qualitative_results:
                    pil_image = inputs['image']
                    img_w, img_h = pil_image.size
                    img_draw = pil_image.copy()
                    draw = ImageDraw.Draw(img_draw)
                    point_radius = 5
                    point_color = "red"
                    
                    for point in points:
                        x = point[0] * img_w  # Scale x coordinate to image width
                        y = point[1] * img_h  # Scale y coordinate to image height
                        draw.ellipse([x-point_radius, y-point_radius, x+point_radius, y+point_radius], 
                                    fill=point_color)
                    
                    # Add text overlay
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=8)
                    except:
                        font = ImageFont.load_default()
                    
                    caption_text = pos_caption[0] if isinstance(pos_caption, list) else pos_caption
                    neg_caption_text = inputs['neg_caption'][0] 
                    text = f"{caption_text}, Pred: {pred_cnt}, GT: {gt_cnt}"
                    if neg_caption_text:
                        text += f", Neg: {neg_caption_text}"
                    draw.text((5, 5), text, fill="white", stroke_width=2, stroke_fill="black", font=font)
                    # Save image
                    filename = f"{caption_text[0]}_{neg_caption_text[0]}_pred_{pred_cnt}_gt_{gt_cnt}_err_{cnt_err}.jpg"
                    filename = filename.replace("/", "_").replace(" ", "_")  # Sanitize filename
                    img_draw.save(os.path.join(qual_img_save_root, filename))
        
        # Calculate final metrics
        # Gather results from all processes for distributed evaluation
        if hasattr(self.accelerator, 'gather'):
            # Gather all evaluation results from all processes
            gathered_mae = self.accelerator.gather(torch.tensor(eval_mae, device=self.device))
            gathered_rmse = self.accelerator.gather(torch.tensor(eval_rmse, device=self.device))
            gathered_total_samples = self.accelerator.gather(torch.tensor(total_samples, device=self.device))
            
            eval_mae = gathered_mae.sum().item()
            eval_rmse = gathered_rmse.sum().item()
            total_samples = gathered_total_samples.sum().item()

            
            # Use gathered results for final calculation
            eval_mae = eval_mae / total_samples
            eval_rmse = eval_rmse / total_samples
            eval_rmse = eval_rmse ** 0.5
        
            metrics = {
                f"{metric_key_prefix}/mae": eval_mae,
                f"{metric_key_prefix}/rmse": eval_rmse,
            }   
        
        # Log metrics
        if self.accelerator.is_main_process:
            wandb.log(metrics, step=self.state.global_step)
        
        # Log qualitative samples if enabled
        if save_qualitative_results and self.accelerator.is_main_process:
            self.log_qualitative_samples(qual_img_save_root, metric_key_prefix)
        
        # save the best val model
        if metric_key_prefix == "val" and eval_mae < self.best_val_mae:
            self.best_val_mae = eval_mae
            self.save_model(os.path.join(self.output_dir, "best_val_model"), _internal_call=True)
        return metrics
    
    def log_qualitative_samples(self, qual_img_save_root, metric_key_prefix, num_samples=4):
        """
        Randomly sample images from qualitative results and log them to wandb.
        
        Args:
            qual_img_save_root: Path to the directory containing qualitative results
            metric_key_prefix: Prefix for the metric (val/test)
            num_samples: Number of images to sample and log
        """
        if not os.path.exists(qual_img_save_root):
            return
            
        # Get all image files in the directory
        image_files = []
        for file in os.listdir(qual_img_save_root):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(file)
        
        if len(image_files) == 0:
            return
            
        # Randomly sample images
        sampled_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        for i, filename in enumerate(sampled_files):
            image_path = os.path.join(qual_img_save_root, filename)
            image = Image.open(image_path)
            if self.accelerator.is_main_process:
                wandb.log({f"{metric_key_prefix}/Qual{i+1}": wandb.Image(image)}, step=self.state.global_step)
    
    def compute_metrics(self, eval_preds):
        """
        This method is not used since we override evaluate().
        """
        return {}