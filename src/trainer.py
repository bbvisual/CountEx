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
        self.weakly_supervised_loss = kwargs.pop('weakly_supervised_loss', 'interval_huber')
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
        self.train_mae = 0
        self.train_rmse = 0
        self.train_mae_anno = 0
        self.train_rmse_anno = 0
        self.train_sample_num = 0
        self.eval_sample_num = 0
        # yifeng Fixme: For quick test, hard code the contrastive loss here
        
        
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
        assert inputs['type'] in ['dot_anno', 'weak_supervised'], "Invalid data type"
        if inputs['type'] == 'dot_anno':
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
        if dot_anno:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
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
                # calculate the density map loss
                density_map_pred = outputs.density_map_pred
                pos_points_normalized = [np.array(pos_points) / np.array(shapes)[::-1] ]
                pos_points_normalized[0] = pos_points_normalized[0].squeeze(0)
                pos_points_normalized = [torch.from_numpy(img_points).float() for img_points in pos_points_normalized] 
                H, W = density_map_pred.shape[-2:]
                pseudo_density = generate_pseudo_density_map(
                    pos_points_normalized[0].to(density_map_pred.device),
                    (H, W),
                    sigma=6.0,          
                    normalize=True,
                )
                assert pseudo_density.shape == density_map_pred.shape
                pseudo_density = pseudo_density.to(density_map_pred.device)
                pseudo_density = pseudo_density.to(torch.bfloat16)
                loss_density = F.mse_loss(density_map_pred, pseudo_density) * self.density_loss_weight
                self.loss_density_list.append(loss_density.item())
                loss += loss_density
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
                if self.weakly_supervised_loss == 'interval_huber':
                    weakly_supervised_loss = interval_huber(density_map_pred_count, annotated_pos_count) * 0.001
                elif self.weakly_supervised_loss == 'ae':
                    weakly_supervised_loss = F.l1_loss(density_map_pred_count, annotated_pos_count) * 0.001
                elif self.weakly_supervised_loss == 'relative_huber':
                    weakly_supervised_loss = relative_huber(density_map_pred_count, annotated_pos_count) * 0.001
                else:
                    raise ValueError(f"Invalid weakly supervised loss: {self.weakly_supervised_loss}")
                # print(f"weakly_supervised_loss: {weakly_supervised_loss.item()}")

                self.weakly_supervised_loss_list.append(weakly_supervised_loss.item())
                loss += weakly_supervised_loss
            else:
                loss += 0

        # add extra loss
        if extra_loss is not None:
            loss += extra_loss
                
        if self.use_contrastive_loss:
            neg_points = inputs['neg_points']
            positive_feature_maps = outputs.positive_feature_maps
            negative_feature_maps = outputs.negative_feature_maps

            pos_points_normalized = [np.array(pos_points) / np.array(shapes)[::-1] ]
            pos_points_normalized[0] = pos_points_normalized[0].squeeze(0)
            pos_points_normalized = [torch.from_numpy(img_points).float() for img_points in pos_points_normalized] 

            
            neg_points_normalized = [np.array(neg_points) / np.array(shapes)[::-1] ]
            neg_points_normalized[0] = neg_points_normalized[0].squeeze(0)
            neg_points_normalized = [torch.from_numpy(img_points).float() for img_points in neg_points_normalized] 

            point_count_map = build_point_count_map(positive_feature_maps[0], pos_points_normalized)
            point_count_map_neg = build_point_count_map(negative_feature_maps[0], neg_points_normalized)

            pos_tokens, lin_index_pos = extract_pos_tokens_single(positive_feature_maps[0], point_count_map)
            neg_tokens, lin_index_neg = extract_pos_tokens_single(negative_feature_maps[0], point_count_map_neg)

            pos_tokens, neg_tokens = filter_overlap(pos_tokens, lin_index_pos,
                                        neg_tokens, lin_index_neg)

            loss_cl = supcon_pos_neg(pos_tokens, neg_tokens, temperature=0.25) * self.contrastive_loss_weight
            self.loss_contrastive_list.append(loss_cl.item())
            loss += loss_cl

            # print(f"loss_contrastive: {loss_cl.item()}")

        
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
            # Clear the lists after logging
            self.loss_label_list.clear()
            self.loss_point_list.clear()
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
                outputs = self.model(**pos_llm_det_inputs)
                
                # Post-process outputs
                outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]
                outputs["pred_logits"] = outputs["logits"]

                if outputs.density_map_pred is not None:
                    density_map_pred = outputs.density_map_pred
                    density_map_pred = density_map_pred.sum()
                    density_map_cnt_err = abs(density_map_pred - pos_count)
                    
                
                results = post_process_grounded_object_detection(outputs, box_threshold=0.40)[0]
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