# coding=utf-8
"""
Evaluation script for density-only counting model.

Uses density.sum() as the predicted count.

Usage:
    python eval_density.py --model countex_density_only --ckpt_path /path/to/checkpoint
"""

import os
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from transformers import (
    GroundingDinoProcessor,
    HfArgumentParser,
)
from torch.utils.data import DataLoader
from hf_model import CountEXDensityOnly, CountEXDensityOnlyNoUncertainty
from hf_model.CountEX import CountEX
from hf_model.CountEXStage2 import CountEXStage2
from utils import collator, build_dataset


@dataclass
class EvalArguments:
    """Arguments for evaluation."""
    model: str = field(
        default="countex_density_only",
        metadata={"help": "Model type: countex, countex_stage2, countex_density_only, countex_density_only_no_uncertainty"}
    )
    backbone_size: str = field(
        default="tiny",
        metadata={"help": "Backbone size: tiny, base, or large"}
    )
    ckpt_path: str = field(
        default=None,
        metadata={"help": "Path to checkpoint"}
    )
    output_dir: str = field(
        default="./eval_results",
        metadata={"help": "Output directory for evaluation results"}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for evaluation"}
    )
    # Data paths
    train_data_path: str = field(
        default="BBVisual/CoCount-train",
        metadata={"help": "Path to training dataset"}
    )
    val_data_path: str = field(
        default="BBVisual/CoCount-val",
        metadata={"help": "Path to validation dataset"}
    )
    test_data_path: str = field(
        default="BBVisual/CoCount-test",
        metadata={"help": "Path to test dataset"}
    )
    weakly_supervised_data_path: str = field(
        default=None,
        metadata={"help": "Path to weakly-supervised dataset (not used in eval)"}
    )
    data_split: str = field(
        default="all",
        metadata={"help": "Data split mode"}
    )
    save_visualizations: bool = field(
        default=True,
        metadata={"help": "Whether to save density map visualizations"}
    )
    max_vis_samples: int = field(
        default=50,
        metadata={"help": "Maximum number of samples to visualize"}
    )


def load_model(args):
    """Load model from checkpoint."""
    model_class_map = {
        'countex': CountEX,
        'countex_stage2': CountEXStage2,
        'countex_density_only': CountEXDensityOnly,
        'countex_density_only_no_uncertainty': CountEXDensityOnlyNoUncertainty,
    }
    if args.model not in model_class_map:
        raise ValueError(f"Unknown model: {args.model}. Choose from: {list(model_class_map.keys())}")
    model = model_class_map[args.model].from_pretrained(args.ckpt_path)
    return model


def evaluate_dataset(model, dataloader, device, save_dir=None, max_vis=50, prefix="eval"):
    """
    Evaluate model on a dataset using density.sum() as prediction.

    Returns:
        metrics: dict with MAE, RMSE
        results: list of per-sample results
    """
    model.eval()

    eval_mae = 0.0
    eval_rmse = 0.0
    total_samples = 0
    results = []
    all_count_errors = []
    pbar = tqdm(dataloader, desc=f"Evaluating {prefix}")

    for step, inputs in enumerate(pbar):
        with torch.no_grad():
            # Prepare inputs
            pos_llm_det_inputs = inputs['pos_llm_det_inputs']
            gt_count = inputs['pos_count']
            pos_caption = inputs['pos_caption']
            neg_caption = inputs['neg_caption']

            pos_llm_det_inputs = pos_llm_det_inputs.to(device)
            pos_llm_det_inputs['pixel_values'] = pos_llm_det_inputs['pixel_values'].to(torch.bfloat16)

            if 'pos_exemplars' in inputs:
                pos_llm_det_inputs['pos_exemplars'] = inputs['pos_exemplars']
            if 'neg_exemplars' in inputs:
                pos_llm_det_inputs['neg_exemplars'] = inputs['neg_exemplars']

            neg_llm_det_inputs = inputs['neg_llm_det_inputs']
            neg_llm_det_inputs = {k: v.to(device) for k, v in neg_llm_det_inputs.items()}
            neg_llm_det_inputs['pixel_values'] = neg_llm_det_inputs['pixel_values'].to(torch.bfloat16)

            pos_llm_det_inputs['neg_token_type_ids'] = neg_llm_det_inputs['token_type_ids']
            pos_llm_det_inputs['neg_attention_mask'] = neg_llm_det_inputs['attention_mask']
            pos_llm_det_inputs['neg_pixel_mask'] = neg_llm_det_inputs['pixel_mask']
            pos_llm_det_inputs['neg_pixel_values'] = neg_llm_det_inputs['pixel_values']
            pos_llm_det_inputs['neg_input_ids'] = neg_llm_det_inputs['input_ids']
            pos_llm_det_inputs['use_neg'] = True

            # Forward pass
            outputs = model(**pos_llm_det_inputs)

            # Get prediction from density map (handle different output formats)
            if hasattr(outputs, 'pred_count') and outputs.pred_count is not None:
                density_map = outputs.density_map
                pred_count = outputs.pred_count.item()
            else:
                # CountEX returns density_map_pred, no pred_count
                density_map = outputs.density_map_pred
                pred_count = density_map.sum().item()
            gt_cnt = float(gt_count)

            # Calculate error
            cnt_err = abs(pred_count - gt_cnt)
            all_count_errors.append(abs(gt_cnt - pred_count))
            eval_mae += cnt_err
            eval_rmse += cnt_err ** 2
            total_samples += 1

            # Store result
            result = {
                'step': step,
                'pred_count': pred_count,
                'gt_count': gt_cnt,
                'error': cnt_err,
                'pos_caption': pos_caption[0] if isinstance(pos_caption, list) else pos_caption,
                'neg_caption': neg_caption[0] if isinstance(neg_caption, list) else neg_caption,
            }
            if 'category' in inputs:
                result['category'] = inputs['category']
            results.append(result)

            # Update progress bar
            current_mae = eval_mae / total_samples
            current_rmse = (eval_rmse / total_samples) ** 0.5
            pbar.set_postfix({
                'Pred': f'{pred_count:.1f}',
                'GT': f'{gt_cnt:.0f}',
                'MAE': f'{current_mae:.3f}',
                'RMSE': f'{current_rmse:.3f}',
            })

            # Save visualization (max_vis <= 0 means save all)
            if save_dir is not None and (max_vis <= 0 or step < max_vis):
                log_var = getattr(outputs, 'density_log_var', None)
                save_visualization(
                    inputs.get('image'),
                    density_map,
                    log_var,
                    pred_count,
                    gt_cnt,
                    pos_caption,
                    neg_caption,
                    save_dir,
                    step
                )

    # Calculate final metrics
    eval_mae = eval_mae / total_samples
    eval_rmse = (eval_rmse / total_samples) ** 0.5

    metrics = {
        f"{prefix}/mae": eval_mae,
        f"{prefix}/rmse": eval_rmse,
        f"{prefix}/total_samples": total_samples,
    }

    # verify the mae with list
    mae_with_list = np.mean(all_count_errors)
    rmse_with_list = np.sqrt(np.mean(np.array(all_count_errors) ** 2))
    print(f"MAE with list: {mae_with_list:.4f}, RMSE with list: {rmse_with_list:.4f}")
    print(f"MAE with list: {mae_with_list:.4f}, RMSE with list: {rmse_with_list:.4f}")

    return metrics, results


def save_visualization(image, density_map, log_var, pred_count, gt_count, pos_caption, neg_caption, save_dir, step):
    """Save density map visualization as separate files: original, density, overlay."""
    if image is None:
        return

    # Create subdirectories
    orig_dir = os.path.join(save_dir, "original")
    density_dir = os.path.join(save_dir, "density")
    overlay_dir = os.path.join(save_dir, "overlay")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(density_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    err = abs(pred_count - gt_count)
    # Captions are wrapped as [[string]] in the dataset, unwrap to string
    pos_cap = pos_caption
    while isinstance(pos_cap, list):
        pos_cap = pos_cap[0]
    neg_cap = neg_caption
    while isinstance(neg_cap, list):
        neg_cap = neg_cap[0]
    # Sanitize captions for filename: replace spaces/special chars, truncate
    def sanitize(s):
        return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in s)
    base_name = f"step_{step:04d}_pos_{sanitize(pos_cap)}_neg_{sanitize(neg_cap)}_pred_{pred_count:.1f}_gt_{gt_count:.0f}_err_{err:.1f}"

    # Convert image to PIL if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = image

    img_w, img_h = pil_image.size

    # 1. Save original image
    pil_image.convert("RGB").save(os.path.join(orig_dir, f"{base_name}.jpg"), quality=95)

    # 2. Save density map as a clean image (no matplotlib axes/borders)
    density = density_map.squeeze().cpu().float().numpy()
    # Resize density to original image size
    density_resized = np.array(Image.fromarray(density).resize((img_w, img_h), Image.BILINEAR))
    # Normalize for colormap
    d_min, d_max = density_resized.min(), density_resized.max()
    if d_max > d_min:
        density_norm = (density_resized - d_min) / (d_max - d_min)
    else:
        density_norm = np.zeros_like(density_resized)
    # Apply jet colormap -> PIL image
    cmap = plt.cm.jet
    density_rgb = (cmap(density_norm)[:, :, :3] * 255).astype(np.uint8)
    density_pil = Image.fromarray(density_rgb)
    density_pil.save(os.path.join(density_dir, f"{base_name}.jpg"), quality=95)

    # 3. Save overlay (density blended on original)
    overlay = Image.blend(pil_image.convert("RGB"), density_pil, alpha=0.5)
    overlay.save(os.path.join(overlay_dir, f"{base_name}.jpg"), quality=95)


def main():
    # Parse arguments
    parser = HfArgumentParser((EvalArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load processor
    if args.backbone_size == 'tiny':
        model_id = "fushh7/llmdet_swin_tiny_hf"
    elif args.backbone_size == 'base':
        model_id = "fushh7/llmdet_swin_base_hf"
    elif args.backbone_size == 'large':
        model_id = "fushh7/llmdet_swin_large_hf"
    else:
        raise ValueError(f"Unknown backbone size: {args.backbone_size}")

    llmdet_processor = GroundingDinoProcessor.from_pretrained(model_id)

    # Load model
    print(f"Loading model from {args.ckpt_path}")
    model = load_model(args)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    model.eval()

    # Load datasets
    _, val_dataset, test_dataset, _ = build_dataset(args)

    all_metrics = {
        "model": args.model,
        "ckpt_path": args.ckpt_path,
        "data_split": args.data_split,
    }

    # Evaluate on test set
    if test_dataset is not None:
        print("\nEvaluating on test dataset...")
        test_save_dir = os.path.join(args.output_dir, "test_vis") if args.save_visualizations else None
        if test_save_dir:
            os.makedirs(test_save_dir, exist_ok=True)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator()
        )

        test_metrics, test_results = evaluate_dataset(
            model, test_dataloader, device,
            save_dir=test_save_dir,
            max_vis=args.max_vis_samples,
            prefix="test"
        )

        all_metrics.update(test_metrics)
        print(f"\nTest Results: MAE={test_metrics['test/mae']:.4f}, RMSE={test_metrics['test/rmse']:.4f}")

        # Save detailed results
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)

    # Evaluate on validation set
    if val_dataset is not None:
        print("\nEvaluating on validation dataset...")
        val_save_dir = os.path.join(args.output_dir, "val_vis") if args.save_visualizations else None
        if val_save_dir:
            os.makedirs(val_save_dir, exist_ok=True)

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator()
        )

        val_metrics, val_results = evaluate_dataset(
            model, val_dataloader, device,
            save_dir=val_save_dir,
            max_vis=args.max_vis_samples,
            prefix="val"
        )

        all_metrics.update(val_metrics)
        print(f"\nVal Results: MAE={val_metrics['val/mae']:.4f}, RMSE={val_metrics['val/rmse']:.4f}")

        # Save detailed results
        with open(os.path.join(args.output_dir, "val_results.json"), "w") as f:
            json.dump(val_results, f, indent=2)

    # Save all metrics
    with open(os.path.join(args.output_dir, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nEvaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
