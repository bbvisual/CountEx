import os
import sys
sys.path.append('GroundingDINO')
import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    TrainingArguments, 
    GroundingDinoProcessor, 
    get_cosine_schedule_with_warmup,
    HfArgumentParser
)
from accelerate.utils import set_seed


from hf_model import CountEX
import wandb
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, concatenate_datasets
from criterion import SetCriterion
from utils import collator, rank0_print, build_dataset
from trainer import FineGrainedCountingTrainer
from datetime import datetime

@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model: str = field(
        default="llmdet",
        metadata={"help": "Model size to use: tiny, base, or large"}
    )
    backbone_size: str = field(
        default="tiny",
        metadata={"help": "Backbone size: tiny, base, or large"}
    )
    model_ckpt_path: str = field(
        default="None",
        metadata={"help": "Model checkpoint path"}
    )
    
@dataclass
class DataTrainingArguments:
    """Arguments for data training configuration."""
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
    data_split: str = field(
        default="all",
        metadata={"help": "Data split mode: 'all' (use all data) or specific category (FOO, FUN, OFF, OTR, HOU) for cross-validation"}
    )
    num_epochs: int = field(
        default=5,
        metadata={"help": "Number of training epochs"}
    )
    save_qualitative_results: bool = field(
        default=False,
        metadata={"help": "Whether to save qualitative results during evaluation"}
    )
    use_weakly_supervised_training: bool = field(
        default=False,
        metadata={"help": "Whether to use weakly-supervised training"}
    )
    weakly_supervised_sample_num: int = field(
        default=1000,
        metadata={"help": "Number of weakly-supervised samples"}
    )
    weakly_supervised_loss: str = field(
        default='interval_huber',
        metadata={"help": "Weakly-supervised loss: 'interval_huber' or 'ae'"}
    )
    use_contrastive_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use contrastive loss"}
    )
    contrastive_loss_weight: float = field(
        default=0.01,
        metadata={"help": "Weight for contrastive loss"}
    )
    density_loss_weight: float = field(
        default=200,
        metadata={"help": "Weight for density loss"}
    )
    use_neg_prob: float = field(
        default=1.0,
        metadata={"help": "Probability of not using negative captions during training"}
    )
    fusion_scale: float = field(
        default=0.25,
        metadata={"help": "Fusion scale"}
    )

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    seed = training_args.seed
    
    # Load datasets
    train_dataset, val_dataset, test_dataset, weakly_supervised_data = build_dataset(data_args)
    if data_args.use_weakly_supervised_training:
        if data_args.weakly_supervised_sample_num < len(weakly_supervised_data):
            import torch, torch.distributed as dist
            import numpy as np
            import json
            assert dist.is_initialized()
            rank = dist.get_rank()
            if rank == 0:
                rng = np.random.default_rng(training_args.seed)
                N = data_args.weakly_supervised_sample_num
                assert N <= len(weakly_supervised_data)
                arr = rng.choice(len(weakly_supervised_data), size=N, replace=False).astype('int64')
                idx = torch.from_numpy(arr)

                indices = idx.tolist()
                if not os.path.exists(training_args.output_dir):
                    os.makedirs(training_args.output_dir)
                indices_save_path = os.path.join(training_args.output_dir, "weakly_supervised_indices.json")
                with open(indices_save_path, 'w') as f:
                    json.dump(indices, f)
            else:
                import time
                time.sleep(1)
            indices_save_path = os.path.join(training_args.output_dir, "weakly_supervised_indices.json")
            # print("indices_save_path: ", indices_save_path)
            with open(indices_save_path, 'r') as f:
                indices = json.load(f)
            weakly_supervised_data = weakly_supervised_data.select(indices)
            dist.barrier()
            #weakly_supervised_data = weakly_supervised_data.shuffle(seed=training_args.seed).select(range(data_args.weakly_supervised_sample_num))
        else:
            weakly_supervised_data = weakly_supervised_data.shuffle(seed=training_args.seed)
        train_dataset = concatenate_datasets([train_dataset, weakly_supervised_data])

    if model_args.model_ckpt_path != "None":
        rank0_print(f"Loading model from {model_args.model_ckpt_path}...")
        model_id = model_args.model_ckpt_path
    else:
        if model_args.backbone_size == 'tiny':
            model_id = "fushh7/llmdet_swin_tiny_hf"
        elif model_args.backbone_size == 'base':
            model_id = "fushh7/llmdet_swin_base_hf"
        elif model_args.backbone_size == 'large':
            model_id = "fushh7/llmdet_swin_large_hf"
        else:
            raise ValueError(f"Unknown backbone size: {model_args.backbone_size}")

    if model_args.model == 'countex':
        model = CountEX.from_pretrained(model_id)
    else:
        raise ValueError(f"Unknown model: {model_args.model}")
    llmdet_processor = GroundingDinoProcessor.from_pretrained("fushh7/llmdet_swin_tiny_hf")
    rank0_print(f"Model loaded for {model_args.model}")
    
    # Initialize criterion
    criterion = SetCriterion()

    # Disable certain parameters for training
    params_to_disable = [
        "model.encoder_output_class_embed.bias",
        "model.encoder_output_bbox_embed.layers.2.bias",
        "model.encoder_output_bbox_embed.layers.2.weight",
        "model.encoder_output_bbox_embed.layers.1.bias",
        "model.encoder_output_bbox_embed.layers.1.weight",
        "model.encoder_output_bbox_embed.layers.0.bias",
        "model.encoder_output_bbox_embed.layers.0.weight",
        "model.enc_output_norm.bias",
        "model.enc_output_norm.weight",
        "model.enc_output.bias",
        "model.enc_output.weight"
    ]

    for name, param in model.named_parameters():
        if name in params_to_disable:
            param.requires_grad = False

    # Update training arguments with data args    
    training_args.num_train_epochs = data_args.num_epochs
    training_args.evaluation_strategy = "steps"
    training_args.save_strategy = "steps"
    training_args.load_best_model_at_end = False
    training_args.remove_unused_columns = False
    training_args.dataloader_pin_memory = False
    training_args.report_to = ["wandb"]

    output_dir = training_args.output_dir
    # Initialize trainer
    trainer = FineGrainedCountingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator(),
        criterion=criterion,
        llmdet_processor=llmdet_processor,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        save_qualitative_results=data_args.save_qualitative_results,
        output_dir=output_dir,
        weakly_supervised_loss=data_args.weakly_supervised_loss,
        use_contrastive_loss=data_args.use_contrastive_loss,
        density_loss_weight=data_args.density_loss_weight,
        use_neg_prob=data_args.use_neg_prob,
    )
    set_seed(seed, device_specific=True)

    # Train the model
    rank0_print("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Final evaluation
    trainer.evaluate()

    # Save the final model
    trainer.save_model()
    rank0_print(f"Model saved to {trainer.args.output_dir}")

if __name__ == "__main__":
    main() 