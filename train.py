"""Train NeuroStat on a pair-format dataset.

Example::

    python train.py \\
        --model_name_or_path Qwen/Qwen2-0.5B \\
        --train_data_path data/train.json \\
        --train_data_format MIRAGE \\
        --output_dir ./ckpt/neurostat \\
        --num_train_epochs 5 \\
        --train_batch_size 8
"""

import argparse
import inspect
import json
import os
import random
import shutil

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments

from neurostat import build_neurostat_detector
from neurostat.data import (
    BinaryTextDataset,
    SequenceClassificationCollator,
    build_binary_classification_samples,
)


# ===================== Custom Trainer =====================

class NeuroStatTrainer(Trainer):
    """Override ``_save`` to de-duplicate tied CausalLM weights.

    ``safetensors`` rejects state dicts containing tensors that share the same
    underlying storage. We de-duplicate and write via ``torch.save`` so that
    checkpoints round-trip reliably.
    """

    def _save(self, output_dir=None, state_dict=None, **kwargs):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        full_state = model_to_save.state_dict()

        dedup_state = {}
        seen_ptrs = {}
        for name, tensor in full_state.items():
            ptr = tensor.data_ptr()
            if ptr in seen_ptrs:
                continue  # skip duplicate (e.g. lm_head.weight == embed_tokens.weight)
            seen_ptrs[ptr] = name
            dedup_state[name] = tensor

        torch.save(dedup_state, os.path.join(output_dir, "pytorch_model.bin"))

        tokenizer = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Always mirror neurostat_config.json into every checkpoint directory
        src_config = os.path.join(self.args.output_dir, "neurostat_config.json")
        dst_config = os.path.join(output_dir, "neurostat_config.json")
        if os.path.exists(src_config) and not os.path.exists(dst_config):
            shutil.copy2(src_config, dst_config)


# ===================== CLI =====================

parser = argparse.ArgumentParser(description="Train NeuroStat (TF + TB dual-branch)")

# --- model ---
parser.add_argument("--model_name_or_path", type=str, required=True,
                    help="HuggingFace repo id or local path of the CausalLM backbone.")
parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument("--freeze_backbone", type=int, default=0,
                    help="1 = freeze CausalLM backbone, 0 = full fine-tune (default).")

# --- data ---
parser.add_argument("--train_data_path", type=str, required=True)
parser.add_argument("--train_data_format", type=str, default="MIRAGE",
                    choices=["MIRAGE", "pair"])
parser.add_argument("--eval_data_path", type=str, default=None)
parser.add_argument("--eval_data_format", type=str, default="MIRAGE",
                    choices=["MIRAGE", "pair"])

# --- output ---
parser.add_argument("--output_dir", type=str, default="./ckpt/neurostat")

# --- training hyper-params ---
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_train_epochs", type=int, default=5)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--logging_steps", type=int, default=10)
parser.add_argument("--save_total_limit", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--precision", type=str, default="auto",
                    choices=["auto", "fp32", "fp16", "bf16"])
parser.add_argument("--report_to", type=str, default="none")
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--disable_eval_during_training", action="store_true")
parser.add_argument("--skip_final_eval", action="store_true")
parser.add_argument("--resume_from_checkpoint", type=str, default=None)

# --- architecture dims ---
parser.add_argument("--tf_dim", type=int, default=64,
                    help="Output dim of the TF branch.")
parser.add_argument("--tb_dim", type=int, default=64,
                    help="Output dim of the TB branch.")
parser.add_argument("--cls_hidden_dim", type=int, default=128)
parser.add_argument("--cls_dropout", type=float, default=0.1)

# --- loss weights ---
parser.add_argument("--lambda_supcon", type=float, default=0.1,
                    help="Weight of the supervised contrastive loss.")
parser.add_argument("--supcon_temperature", type=float, default=0.07)
parser.add_argument("--lambda_orth", type=float, default=1e-3,
                    help="Weight of the orthogonal penalty between TF and TB.")


# ===================== Utilities =====================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_precision(precision: str, use_cpu: bool):
    if use_cpu or not torch.cuda.is_available():
        return False, False
    if precision == "fp32":
        return False, False
    if precision == "fp16":
        return True, False
    if precision == "bf16":
        return False, True
    supports_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    return (not supports_bf16), supports_bf16


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0, pos_label=1,
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_training_args(args, run_name: str, fp16: bool, bf16: bool):
    supported_params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": args.output_dir,
        "run_name": run_name,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "fp16": fp16,
        "bf16": bf16,
        "dataloader_num_workers": 4,
        "report_to": args.report_to,
        "remove_unused_columns": False,
    }

    if args.disable_eval_during_training:
        if "eval_strategy" in supported_params:
            kwargs["eval_strategy"] = "no"
        elif "evaluation_strategy" in supported_params:
            kwargs["evaluation_strategy"] = "no"
        kwargs["load_best_model_at_end"] = False
    else:
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "f1"
        kwargs["greater_is_better"] = True
        if "eval_strategy" in supported_params:
            kwargs["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in supported_params:
            kwargs["evaluation_strategy"] = "epoch"

    if "save_safetensors" in supported_params:
        kwargs["save_safetensors"] = False

    if args.use_cpu:
        if "use_cpu" in supported_params:
            kwargs["use_cpu"] = True
        elif "no_cuda" in supported_params:
            kwargs["no_cuda"] = True

    return TrainingArguments(**kwargs)


# ===================== Main =====================

def main(args):
    set_seed(args.seed)
    if not args.use_cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Pass --use_cpu if you really want CPU training."
        )
    fp16, bf16 = resolve_precision(args.precision, args.use_cpu)

    # ---- build model ----
    model, tokenizer, resolved_model_path = build_neurostat_detector(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        freeze_backbone=bool(args.freeze_backbone),
        tf_dim=args.tf_dim,
        tb_dim=args.tb_dim,
        cls_hidden_dim=args.cls_hidden_dim,
        cls_dropout=args.cls_dropout,
        supcon_temperature=args.supcon_temperature,
        lambda_supcon=args.lambda_supcon,
        lambda_orth=args.lambda_orth,
        max_length=args.max_length,
    )

    if args.gradient_checkpointing:
        if hasattr(model.backbone, "gradient_checkpointing_enable"):
            model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone.config, "use_cache"):
            model.backbone.config.use_cache = False

    # ---- data ----
    train_samples = build_binary_classification_samples(
        data_path=args.train_data_path,
        data_format=args.train_data_format,
        shuffle=True,
        seed=args.seed,
    )
    train_dataset = BinaryTextDataset(train_samples)

    eval_dataset = None
    if args.eval_data_path and (not args.disable_eval_during_training or not args.skip_final_eval):
        eval_samples = build_binary_classification_samples(
            data_path=args.eval_data_path,
            data_format=args.eval_data_format,
            shuffle=False,
            seed=args.seed,
        )
        eval_dataset = BinaryTextDataset(eval_samples)

    collator = SequenceClassificationCollator(tokenizer=tokenizer, max_length=args.max_length)

    # ---- run name ----
    model_name = os.path.basename(resolved_model_path.rstrip("/"))
    train_name = os.path.splitext(os.path.basename(args.train_data_path))[0]
    run_name = f"neurostat_{model_name}_{train_name}"

    training_args = build_training_args(args, run_name, fp16, bf16)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    trainer = NeuroStatTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=None if args.disable_eval_during_training else compute_metrics,
    )

    print("=" * 60)
    print("NeuroStat Training")
    print(f"  Backbone:     {resolved_model_path}")
    print(f"  Frozen:       {bool(args.freeze_backbone)}")
    print(f"  TF dim:       {args.tf_dim}   TB dim: {args.tb_dim}")
    print(f"  SupCon:       lambda={args.lambda_supcon}, tau={args.supcon_temperature}")
    print(f"  Orth penalty: lambda={args.lambda_orth}")
    print(f"  Trainable:    {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    print(f"  Train samples: {len(train_dataset)} | Eval: {len(eval_dataset) if eval_dataset else 0}")
    print(f"  Output dir:   {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "neurostat_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)

    # persist config after final save
    with open(os.path.join(args.output_dir, "neurostat_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if not args.skip_final_eval and eval_dataset is not None:
        print(trainer.evaluate(eval_dataset=eval_dataset))


if __name__ == "__main__":
    main(parser.parse_args())
