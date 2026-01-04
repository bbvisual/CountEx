# 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
import os
import torch
from PIL import ImageDraw
from dataclasses import dataclass, field
from tqdm import tqdm
from transformers import (
    GroundingDinoProcessor, 
    HfArgumentParser
)
from hf_model import CountEX
from torch.utils.data import DataLoader
from utils import collator, build_dataset, post_process_grounded_object_detection
from datetime import datetime


@dataclass
class EvalArguments:
    """Arguments for model configuration."""
    model: str = field(
        default="neg_seg_with_fpn_dhead",
        metadata={"help": "Model size to use: tiny, base, or large"}
    )
    backbone_size: str = field(
        default="tiny",
        metadata={"help": "Backbone size: tiny, base, or large"}
    )
    ckpt_path: str = field(
        default="/nfs/bigrail/add_disk4/yifeng/fg_count/llmdet_for_fg_count/NegSegWdH_3eph_2025_07_05_00/",
        metadata={"help": "Path to checkpoint"}
    )
    data_split: str = field(
        default="HOU",
        metadata={"help": "Data split mode: 'all' (use all data) or specific category (FOO, FUN, OFF, OTR, HOU) for cross-validation"}
    )
    output_dir: str = field(
        default="/nfs/bigrail/add_disk4/yifeng/fg_count/fg_count_qual_result",
        metadata={"help": "Output directory for evaluation results"}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for evaluation"}
    )
    train_data_path: str = field(
        default="/nfs/bigrail/add_disk4/yifeng/fg_count/benchmark/hf_fg_count_dot_train_v3",
        metadata={"help": "Path to training dataset"}
    )
    val_data_path: str = field(
        default="/nfs/bigrail/add_disk4/yifeng/fg_count/benchmark/hf_fg_count_val_v3",
        metadata={"help": "Path to validation dataset"}
    )
    test_data_path: str = field(
        default="/nfs/bigrail/add_disk4/yifeng/fg_count/benchmark/hf_fg_count_test_v3",
        metadata={"help": "Path to test dataset"}
    )
    weakly_supervised_data_path: str = field(
        default="/nfs/bigrail/add_disk4/yifeng/fg_count/benchmark/weak_supervised_train_sample_v3",
        metadata={"help": "Path to weakly-supervised training dataset"}
    )

parser = HfArgumentParser((EvalArguments))
eval_args = parser.parse_args_into_dataclasses()[0]

# Create output directory
print(eval_args.output_dir)

# Load datasets
_, val_dataset, test_dataset, _ = build_dataset(eval_args)

if eval_args.backbone_size == 'tiny':
    model_id = "fushh7/llmdet_swin_tiny_hf"
elif eval_args.backbone_size == 'base':
    model_id = "fushh7/llmdet_swin_base_hf"
elif eval_args.backbone_size == 'large':
    model_id = "fushh7/llmdet_swin_large_hf"
else:
    raise ValueError(f"Unknown backbone size: {eval_args.backbone_size}")

if eval_args.model == 'countex':
    model = CountEX.from_pretrained(eval_args.ckpt_path)
else:
    raise ValueError(f"Unknown model: {eval_args.model}")

llmdet_processor = GroundingDinoProcessor.from_pretrained(model_id)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = model.to(torch.bfloat16)  # Set model to bfloat16
model.eval()

def evaluate_single_gpu(eval_dataset, metric_key_prefix="eval", save_qualitative_results=True):
    """
    Single GPU evaluation function for fine-grained counting.
    """
    # Create dataloader
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=eval_args.batch_size, 
        shuffle=False, 
        collate_fn=collator()
    )
    
    # Initialize metrics
    eval_mae = 0.0
    eval_rmse = 0.0
    
    total_samples = 0
    
    # Create qualitative results directory
    if save_qualitative_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        qual_img_save_root = os.path.join(eval_args.output_dir, metric_key_prefix)
        os.makedirs(qual_img_save_root, exist_ok=True)
        print(f"Saving qualitative results to: {qual_img_save_root}")
    
    # Run evaluation
    pbar = tqdm(eval_dataloader, desc=f"Evaluating {metric_key_prefix}")
    overall_results = {}
    all_errors = []
    for step, inputs in enumerate(pbar):
        with torch.no_grad():
            # Extract inputs
            pos_llm_det_inputs = inputs['pos_llm_det_inputs']
            pos_caption = inputs['pos_caption']
            neg_caption = inputs['neg_caption']
            image_name = inputs['image_name']
            shapes = inputs['shapes']
            pos_points = inputs['pos_points']
            pos_count = inputs['pos_count']
            annotated_pos_count = inputs['annotated_pos_count']
            category = inputs['category']
            # print("category: ", category)
            image_name = image_name + "_" + str(step)
            overall_results[image_name] = {}
            
            
            # Move inputs to device and set to bfloat16
            pos_llm_det_inputs = pos_llm_det_inputs.to(device)
            pos_llm_det_inputs['pixel_values'] = pos_llm_det_inputs['pixel_values'].to(torch.bfloat16)
            if 'pos_exemplars' in inputs:
                pos_llm_det_inputs['pos_exemplars'] = inputs['pos_exemplars']
            if 'neg_exemplars' in inputs:
                pos_llm_det_inputs['neg_exemplars'] = inputs['neg_exemplars']
            
            neg_llm_det_inputs = inputs['neg_llm_det_inputs']
            neg_llm_det_inputs = {k: v.to(device) for k, v in neg_llm_det_inputs.items()}
            neg_llm_det_inputs['pixel_values'] = neg_llm_det_inputs['pixel_values'].to(torch.bfloat16)
            
            # Merge negative inputs into positive inputs
            pos_llm_det_inputs['neg_token_type_ids'] = neg_llm_det_inputs['token_type_ids']
            pos_llm_det_inputs['neg_attention_mask'] = neg_llm_det_inputs['attention_mask']
            pos_llm_det_inputs['neg_pixel_mask'] = neg_llm_det_inputs['pixel_mask']
            pos_llm_det_inputs['neg_pixel_values'] = neg_llm_det_inputs['pixel_values']
            pos_llm_det_inputs['neg_input_ids'] = neg_llm_det_inputs['input_ids']
            pos_llm_det_inputs['use_neg'] = True

            pil_image = inputs['image']
            img_w, img_h = pil_image.size
            outputs = model(**pos_llm_det_inputs)
            
            # Post-process outputs
            outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]
            outputs["pred_logits"] = outputs["logits"]
            
            results = post_process_grounded_object_detection(outputs, box_threshold=model.box_threshold)[0]
            # print("box_threshold: ", model.box_threshold)
            boxes = results["boxes"]
            boxes = [box.tolist() for box in boxes]
            points = [[box[0], box[1]] for box in boxes]
            
            # Calculate metrics
            pred_cnt = len(points)
            gt_cnt = int(pos_count)
            cnt_err = abs(pred_cnt - gt_cnt)
            all_errors.append(cnt_err)
            eval_mae += cnt_err
            eval_rmse += cnt_err ** 2
            total_samples += 1
            
            # Update progress bar with current metrics
            current_mae = eval_mae / total_samples
            current_rmse = (eval_rmse / total_samples) ** 0.5
            pbar.set_postfix({
                'Pred': pred_cnt,
                'GT': gt_cnt,   
                'MAE': f'{current_mae:.3f}',
                'RMSE': f'{current_rmse:.3f}',
            })
    
            overall_results[image_name]['category'] = category
            overall_results[image_name]['pred_cnt'] = pred_cnt
            overall_results[image_name]['gt_cnt'] = gt_cnt
            overall_results[image_name]['cnt_err'] = cnt_err
            overall_results[image_name]['pos_caption'] = pos_caption
            overall_results[image_name]['neg_caption'] = neg_caption
            
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
                
                filename = "{}.jpg".format(image_name)
                save_path = os.path.join(qual_img_save_root, filename)
                img_draw.save(save_path)

                # save the original image
                ori_filename = filename.replace(".jpg", "_ori.jpg")
                pil_image.save(os.path.join(qual_img_save_root, ori_filename))

    
    # Calculate final metrics
    eval_mae = eval_mae / total_samples
    eval_rmse = eval_rmse / total_samples
    eval_rmse = eval_rmse ** 0.5
    import numpy as np
    print(len(all_errors))
    print(np.mean(np.array(all_errors)))


    # save the overall results
    with open(os.path.join(eval_args.output_dir, metric_key_prefix, "overall_results.json"), "w") as f:
        json.dump(overall_results, f)
    
    metrics = {
        f"{metric_key_prefix}/mae": eval_mae,
        f"{metric_key_prefix}/rmse": eval_rmse,
    }
    
    # Print results
    print(f"\nEvaluation Results ({metric_key_prefix}):")
    print(f"MAE: {eval_mae:.4f}")
    print(f"RMSE: {eval_rmse:.4f}")
    print(f"Total samples: {total_samples}")
    
    return metrics


# Main evaluation
if __name__ == "__main__":
    import json
    print(f"Starting evaluation with model: {eval_args.model}, backbone: {eval_args.backbone_size}")
    print(f"Device: {device}")
    
    # evaluate_single_gpu(train_dataset, "train", save_qualitative_results=True)
    all_metrics = {}
    all_metrics.update({
        "model": eval_args.model,
        "backbone_size": eval_args.backbone_size,
        "ckpt_path": eval_args.ckpt_path,
        "data_split": eval_args.data_split,
    })
    all_metrics.update({
        "train_data_path": eval_args.train_data_path,
        "val_data_path": eval_args.val_data_path,
        "test_data_path": eval_args.test_data_path,
    })
    # Evaluate on test dataset
    if test_dataset is not None:
        print("\nEvaluating on test dataset...")
        test_metrics = evaluate_single_gpu(test_dataset, "test", save_qualitative_results=True)
        all_metrics.update(test_metrics)
    
    # Evaluate on validation dataset
    if val_dataset is not None:
        print("\nEvaluating on validation dataset...")
        val_metrics = evaluate_single_gpu(val_dataset, "val", save_qualitative_results=True)
        all_metrics.update(val_metrics)
    
    print("\nEvaluation completed!")
    # save all metrics
    with open(os.path.join(eval_args.output_dir, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f)