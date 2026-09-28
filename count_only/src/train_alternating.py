# coding=utf-8
"""
Training script for alternating training between density head and detection decoder.

GAN-like mutual learning:
- Phase 1 (density): Freeze decoder, train density head with pseudo_density_loss
- Phase 2 (detection): Freeze density head, train decoder with detection_alignment_loss

Usage:
    python train_alternating.py --model countex_with_detection_prior \
        --use_alternating_training True --density_phase_length 100 --detection_phase_length 10
"""

import os
import torch
import wandb
from dataclasses import dataclass, field
from transformers import (
    GroundingDinoProcessor,
    TrainingArguments,
    HfArgumentParser,
)
from hf_model.CountEX import CountEX
from hf_model.CountEXWithDetectionPrior import CountEXWithDetectionPrior
from hf_model.CountEXStage2 import CountEXStage2
from alternating_trainer import AlternatingTrainer
from utils import collator, build_dataset


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model: str = field(
        default="countex_with_detection_prior",
        metadata={"help": "Model type (only countex_with_detection_prior supported)"}
    )
    backbone_size: str = field(
        default="tiny",
        metadata={"help": "Backbone size: tiny, base, or large"}
    )
    pretrained_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model (optional, for Stage 2 from Stage 1 checkpoint)"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
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
        metadata={"help": "Path to weakly-supervised training dataset (optional)"}
    )
    data_split: str = field(
        default="all",
        metadata={"help": "Data split mode: 'all' or specific category for cross-validation"}
    )
    use_weakly_supervised_training: bool = field(
        default=False,
        metadata={"help": "Whether to use weakly-supervised dataset in training"}
    )
    weakly_supervised_sample_num: int = field(
        default=4000,
        metadata={"help": "Number of samples to use from weakly-supervised dataset"}
    )
    train_data_ratio: float = field(
        default=1.0,
        metadata={"help": "Ratio of training data to use (0.0-1.0), randomly sampled"}
    )


@dataclass
class LossArguments:
    """Arguments for loss configuration."""
    count_loss_type: str = field(
        default="interval_huber",
        metadata={"help": "Count loss type: interval_huber, ae, mse, relative_huber"}
    )
    count_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for count loss"}
    )
    use_uncertainty_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use uncertainty-weighted loss"}
    )
    use_neg_prob: float = field(
        default=0.0,
        metadata={"help": "Probability of using negative prompt during training (default 0 for detection prior)"}
    )
    # Density phase loss
    use_pseudo_density_loss: bool = field(
        default=True,
        metadata={"help": "Whether to use MSE loss between raw_density and detection_prior (density phase)"}
    )
    pseudo_density_weight: float = field(
        default=1000.0,
        metadata={"help": "Weight for pseudo density MSE loss"}
    )
    # Alternating training arguments
    use_alternating_training: bool = field(
        default=True,
        metadata={"help": "Whether to use alternating training between density and detection"}
    )
    density_phase_length: int = field(
        default=100,
        metadata={"help": "Number of steps for density training phase"}
    )
    detection_phase_length: int = field(
        default=10,
        metadata={"help": "Number of steps for detection training phase"}
    )
    detection_alignment_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for detection alignment loss (detection phase)"}
    )
    detection_count_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for detection count loss (detection phase)"}
    )
    # Mean Teacher settings
    use_mean_teacher: bool = field(
        default=False,
        metadata={"help": "Whether to use Mean Teacher (EMA) for consistency regularization"}
    )
    consistency_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for Mean Teacher consistency loss (MSE between student and teacher density)"}
    )
    ema_decay: float = field(
        default=0.999,
        metadata={"help": "EMA decay rate for Mean Teacher"}
    )
    # Count noise (for robustness ablation)
    count_noise_ratio: float = field(
        default=0.0,
        metadata={"help": "Multiplicative noise ratio for gt count. E.g., 0.1 means c' = c * (1 + uniform(-0.1, 0.1)). 0 means no noise."}
    )
    # Rebuttal: matched-inference model selection
    save_best_det: bool = field(
        default=False,
        metadata={"help": "Also save best_val_det_model, selected by the DETECTION-head val MAE (for decoder-only controls)"}
    )
    det_eval_threshold: float = field(
        default=0.42,
        metadata={"help": "Confidence threshold used for the detection-head val metric"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended training arguments."""
    wandb_project: str = field(
        default="countex_alternating",
        metadata={"help": "Wandb project name"}
    )
    wandb_name: str = field(
        default=None,
        metadata={"help": "Wandb run name"}
    )
    save_qualitative_results: bool = field(
        default=False,
        metadata={"help": "Whether to save qualitative results during evaluation"}
    )


def main():
    # Parse arguments
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        LossArguments,
        CustomTrainingArguments,
    ))
    model_args, data_args, loss_args, training_args = parser.parse_args_into_dataclasses()

    # Setup wandb
    if training_args.local_rank <= 0:
        config_dict = {
            "model": model_args.model,
            "backbone_size": model_args.backbone_size,
            "count_loss_type": loss_args.count_loss_type,
            "use_neg_prob": loss_args.use_neg_prob,
            "use_pseudo_density_loss": loss_args.use_pseudo_density_loss,
            "pseudo_density_weight": loss_args.pseudo_density_weight,
            "use_alternating_training": loss_args.use_alternating_training,
            "density_phase_length": loss_args.density_phase_length,
            "detection_phase_length": loss_args.detection_phase_length,
            "detection_alignment_weight": loss_args.detection_alignment_weight,
            "detection_count_loss_weight": loss_args.detection_count_loss_weight,
            "use_weakly_supervised_training": data_args.use_weakly_supervised_training,
            "weakly_supervised_sample_num": data_args.weakly_supervised_sample_num,
            "train_data_ratio": data_args.train_data_ratio,
            "use_mean_teacher": loss_args.use_mean_teacher,
            "consistency_weight": loss_args.consistency_weight,
            "ema_decay": loss_args.ema_decay,
            "count_noise_ratio": loss_args.count_noise_ratio,
        }
        wandb.init(
            project=training_args.wandb_project,
            name=training_args.wandb_name or f"alternating_{data_args.data_split}",
            config=config_dict
        )

    # Load processor
    if model_args.backbone_size == 'tiny':
        model_id = "fushh7/llmdet_swin_tiny_hf"
    elif model_args.backbone_size == 'base':
        model_id = "fushh7/llmdet_swin_base_hf"
    elif model_args.backbone_size == 'large':
        model_id = "fushh7/llmdet_swin_large_hf"
    else:
        raise ValueError(f"Unknown backbone size: {model_args.backbone_size}")

    llmdet_processor = GroundingDinoProcessor.from_pretrained(model_id)

    # Load model
    model_class_map = {
        'countex_stage2': CountEXStage2,
    }
    ModelClass = model_class_map.get(model_args.model, CountEXStage2)
    print(f"Using model class: {ModelClass.__name__}")

    if model_args.pretrained_path:
        print(f"LLLLLLLLLLLLLLLLL Loading model from {model_args.pretrained_path}")
        model = ModelClass.from_pretrained(model_args.pretrained_path)
    else:
        print(f"Initializing model from {model_id}")
        model = ModelClass.from_pretrained(model_id)

    # Check for NaN in model weights
    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append((name, torch.isnan(param).sum().item(), param.numel()))
    if nan_params:
        print("WARNING: Found NaN in model parameters!")
        for name, nan_count, total in nan_params[:10]:
            print(f"  {name}: {nan_count}/{total} NaN values")
    else:
        print("Model weights OK: No NaN values found")

    # Unfreeze decoder for alternating training (so optimizer includes these params)
    if loss_args.use_alternating_training:
        print("\nUnfreezing decoder for alternating training...")
        for p in model.model.decoder.parameters():
            p.requires_grad = True
        for p in model.bbox_embed.parameters():
            p.requires_grad = True
        for p in model.class_embed.parameters():
            p.requires_grad = True

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Verify frozen/trainable modules
    print("\nVerifying module status:")
    module_checks = [
        ("model.backbone", model.model.backbone),
        ("model.encoder", model.model.encoder),
        ("model.decoder", model.model.decoder),
        ("bbox_embed", model.bbox_embed),
        ("class_embed", model.class_embed),
        ("density_head", model.density_head),
    ]
    for name, module in module_checks:
        num_params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        is_frozen = trainable == 0
        status = "FROZEN" if is_frozen else f"TRAINABLE ({trainable:,} params)"
        print(f"  - {name}: {status}")

    # List all trainable parameter groups
    print("\nTrainable parameters by group:")
    trainable_groups = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            group = name.split('.')[0]
            if group not in trainable_groups:
                trainable_groups[group] = 0
            trainable_groups[group] += param.numel()
    for group, count in sorted(trainable_groups.items(), key=lambda x: -x[1]):
        print(f"  - {group}: {count:,} params")

    # Convert to bfloat16
    model = model.to(torch.bfloat16)

    # Build datasets
    train_dataset, val_dataset, test_dataset, weakly_supervised_dataset = build_dataset(data_args)

    # Subsample training data if ratio < 1.0
    if data_args.train_data_ratio < 1.0:
        from torch.utils.data import Subset
        import random
        total = len(train_dataset)
        n_samples = max(1, int(total * data_args.train_data_ratio))
        indices = random.sample(range(total), n_samples)
        train_dataset = Subset(train_dataset, indices)
        print(f"Subsampled training data: {n_samples}/{total} ({data_args.train_data_ratio*100:.1f}%)")

    # Combine datasets if weakly supervised training is enabled
    if data_args.use_weakly_supervised_training and weakly_supervised_dataset is not None:
        from torch.utils.data import ConcatDataset, Subset
        import random
        if data_args.weakly_supervised_sample_num > 0 and len(weakly_supervised_dataset) > data_args.weakly_supervised_sample_num:
            indices = random.sample(range(len(weakly_supervised_dataset)), data_args.weakly_supervised_sample_num)
            weakly_supervised_dataset = Subset(weakly_supervised_dataset, indices)
            print(f"Sampled {data_args.weakly_supervised_sample_num} from weakly supervised dataset")
        train_dataset = ConcatDataset([train_dataset, weakly_supervised_dataset])
        print(f"Combined dataset size: {len(train_dataset)}")
    else:
        print(f"Training with only main dataset, size: {len(train_dataset)}")

    # Create Mean Teacher (CountEX) from pretrained checkpoint if enabled
    teacher_model = None
    if loss_args.use_mean_teacher and model_args.pretrained_path:
        print(f"\nCreating CountEX teacher from {model_args.pretrained_path}")
        teacher_model = CountEX.from_pretrained(model_args.pretrained_path)
        teacher_model = teacher_model.to(torch.bfloat16)
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"  Teacher (CountEX): {teacher_params:,} params")

    # Create AlternatingTrainer
    trainer = AlternatingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator(),
        llmdet_processor=llmdet_processor,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        output_dir=training_args.output_dir,
        # Loss settings
        count_loss_type=loss_args.count_loss_type,
        count_loss_weight=loss_args.count_loss_weight,
        use_uncertainty_loss=loss_args.use_uncertainty_loss,
        use_neg_prob=loss_args.use_neg_prob,
        use_pseudo_density_loss=loss_args.use_pseudo_density_loss,
        pseudo_density_weight=loss_args.pseudo_density_weight,
        save_qualitative_results=training_args.save_qualitative_results,
        # Alternating training settings
        use_alternating_training=loss_args.use_alternating_training,
        density_phase_length=loss_args.density_phase_length,
        detection_phase_length=loss_args.detection_phase_length,
        detection_alignment_weight=loss_args.detection_alignment_weight,
        detection_count_loss_weight=loss_args.detection_count_loss_weight,
        # Mean Teacher settings
        use_mean_teacher=loss_args.use_mean_teacher,
        consistency_weight=loss_args.consistency_weight,
        ema_decay=loss_args.ema_decay,
        teacher_model=teacher_model,
        count_noise_ratio=loss_args.count_noise_ratio,
        save_best_det=loss_args.save_best_det,
        det_eval_threshold=loss_args.det_eval_threshold,
    )

    print("\n" + "=" * 60)
    print("Alternating Training Configuration")
    print("=" * 60)
    print(f"  Density phase: {loss_args.density_phase_length} steps (pseudo_density_loss, weight={loss_args.pseudo_density_weight})")
    print(f"  Detection phase: {loss_args.detection_phase_length} steps (detection_alignment_loss, weight={loss_args.detection_alignment_weight})")
    print("=" * 60 + "\n")

    # Train
    print("Starting alternating training...")
    trainer.train()

    # Final evaluation
    print("\nFinal evaluation on test set:")
    trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    # Save final model
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    print(f"\nTraining completed! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()