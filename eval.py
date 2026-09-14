"""Evaluate a trained NeuroStat checkpoint on a pair-format dataset.

Example::

    python eval.py \\
        --model_name_or_path ./ckpt/neurostat \\
        --eval_data_path data/test.json \\
        --eval_data_format MIRAGE \\
        --save_path ./results/neurostat \\
        --save_file test.json

Output JSON contains AUROC / AUPR / MCC / Balanced Acc / TPR@FPR=5%% as well
as the raw per-sample scores for both classes.
"""

import argparse
import json
import os

import accelerate
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from neurostat import build_neurostat_detector
from neurostat.data import PairTextDataset
from neurostat.metrics import (
    AUROC,
    AUPR,
    Balanced_Accuracy,
    MCC,
    TPR_at_FPR5,
)


parser = argparse.ArgumentParser(description="Evaluate NeuroStat")
parser.add_argument("--model_name_or_path", type=str, required=True,
                    help="Path to a trained NeuroStat checkpoint (must contain "
                         "neurostat_config.json).")
parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument("--eval_data_path", type=str, required=True)
parser.add_argument("--eval_data_format", type=str, default="MIRAGE",
                    choices=["MIRAGE", "pair"])
parser.add_argument("--save_path", type=str, default="./results/neurostat")
parser.add_argument("--save_file", type=str, default="eval.json")
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--skip_best_threshold", action="store_true",
                    help="Skip the exhaustive best-threshold search to speed up evaluation.")


class NeuroStatWrapper(nn.Module):
    """Thin wrapper that rebuilds NeuroStat with its stored config, then loads
    the trained weights.
    """

    def __init__(self, checkpoint_path: str, cache_dir: str | None = None,
                 max_length: int = 512):
        super().__init__()
        config_path = os.path.join(checkpoint_path, "neurostat_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"neurostat_config.json not found in {checkpoint_path}. "
                "Is this a NeuroStat checkpoint?"
            )
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.model, self.tokenizer, _ = build_neurostat_detector(
            model_name_or_path=cfg["model_name_or_path"],
            cache_dir=cache_dir or cfg.get("cache_dir"),
            freeze_backbone=True,
            tf_dim=cfg.get("tf_dim", 64),
            tb_dim=cfg.get("tb_dim", 64),
            cls_hidden_dim=cfg.get("cls_hidden_dim", 128),
            cls_dropout=cfg.get("cls_dropout", 0.1),
            max_length=max_length,
        )

        weight_files = [
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            os.path.join(checkpoint_path, "model.safetensors"),
        ]
        loaded = False
        for wf in weight_files:
            if not os.path.exists(wf):
                continue
            if wf.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                except ImportError:
                    continue
                state = load_file(wf)
            else:
                state = torch.load(wf, map_location="cpu")
            self.model.load_state_dict(state, strict=False)
            loaded = True
            break

        if not loaded:
            print(f"[WARNING] No weight file found in {checkpoint_path}; "
                  "classification heads will use random init.")

        self.max_length = max_length

    def forward(self, input_texts: list[str]) -> list[float]:
        return self.model.predict_scores(input_texts)


def compute_accuracy(neg_list, pos_list, threshold):
    neg_correct = sum(score <= threshold for score in neg_list)
    pos_correct = sum(score > threshold for score in pos_list)
    total = len(neg_list) + len(pos_list)
    return float((neg_correct + pos_correct) / total) if total > 0 else 0.0


if __name__ == "__main__":
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"--threshold must be within [0, 1], got {args.threshold}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    accelerator = accelerate.Accelerator(cpu=args.use_cpu)
    if accelerator.is_main_process:
        print(f"Running on device: {accelerator.device} | use_cpu={args.use_cpu}")

    model = NeuroStatWrapper(
        checkpoint_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        max_length=args.max_length,
    )
    model.eval()

    dataset = PairTextDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=PairTextDataset.collate_fn,
    )

    model, data_loader = accelerator.prepare(model, data_loader)

    local_original_scores = []
    local_rewritten_scores = []

    for item in tqdm(
        data_loader,
        total=len(data_loader),
        desc=f"Evaluating NeuroStat on {os.path.basename(args.eval_data_path)}",
        disable=not accelerator.is_main_process,
    ):
        local_original_scores.extend(model(item["original"]))
        local_rewritten_scores.extend(model(item["rewritten"]))

    accelerator.wait_for_everyone()
    all_original_scores = accelerator.gather_for_metrics(
        torch.tensor(local_original_scores, device=accelerator.device)
    ).cpu().tolist()
    all_rewritten_scores = accelerator.gather_for_metrics(
        torch.tensor(local_rewritten_scores, device=accelerator.device)
    ).cpu().tolist()

    if not accelerator.is_main_process:
        exit(0)

    fpr, tpr, eval_auroc = AUROC(all_original_scores, all_rewritten_scores)
    prec, recall, eval_aupr = AUPR(all_original_scores, all_rewritten_scores)
    tpr_at_5 = TPR_at_FPR5(all_original_scores, all_rewritten_scores)

    orig_t = torch.tensor(all_original_scores)
    rew_t = torch.tensor(all_rewritten_scores)

    fixed_mcc = MCC(all_original_scores, all_rewritten_scores, threshold=args.threshold)
    fixed_bacc = Balanced_Accuracy(all_original_scores, all_rewritten_scores, threshold=args.threshold)
    fixed_acc = compute_accuracy(all_original_scores, all_rewritten_scores, args.threshold)

    if args.skip_best_threshold:
        best_mcc = fixed_mcc
        best_bacc = fixed_bacc
        best_threshold = args.threshold
        print("[INFO] Skipped best-threshold search (--skip_best_threshold).")
    else:
        best_mcc = 0.0
        best_bacc = 0.0
        best_threshold = args.threshold
        all_scores = all_original_scores + all_rewritten_scores
        for threshold in tqdm(all_scores, total=len(all_scores),
                              desc="Searching best threshold"):
            mcc = MCC(all_original_scores, all_rewritten_scores, threshold=threshold)
            bacc = Balanced_Accuracy(all_original_scores, all_rewritten_scores, threshold=threshold)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold
            if bacc > best_bacc:
                best_bacc = bacc

    print(f"Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f}")
    print(
        f"Threshold={args.threshold:.4f} -> ACC: {fixed_acc:.4f} | "
        f"MCC: {fixed_mcc:.4f} | Balanced Acc: {fixed_bacc:.4f}"
    )
    print(
        f"Best threshold={best_threshold:.4f} -> MCC: {best_mcc:.4f} | "
        f"Balanced Acc: {best_bacc:.4f}"
    )

    result = {
        "method": "neurostat",
        "model_name_or_path": args.model_name_or_path,
        "eval_dataset": os.path.splitext(os.path.basename(args.eval_data_path))[0],
        "eval_batch_size": args.eval_batch_size,
        "threshold": args.threshold,
        "original_score_mean": float(orig_t.mean()),
        "original_score_std": float(orig_t.std()),
        "rewritten_score_mean": float(rew_t.mean()),
        "rewritten_score_std": float(rew_t.std()),
        "AUROC": eval_auroc,
        "AUPR": eval_aupr,
        "BEST_THRESHOLD": best_threshold,
        "BEST_MCC": best_mcc,
        "BEST_BALANCED_ACCURACY": best_bacc,
        "THRESHOLD_ACC": fixed_acc,
        "THRESHOLD_MCC": fixed_mcc,
        "THRESHOLD_BALANCED_ACCURACY": fixed_bacc,
        "TPR_AT_FPR_5%": tpr_at_5,
        "original_scores": all_original_scores,
        "rewritten_scores": all_rewritten_scores,
        "fpr": fpr,
        "tpr": tpr,
        "precision": prec,
        "recall": recall,
    }

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, args.save_file), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"[OK] Saved results to {os.path.join(args.save_path, args.save_file)}")
