from __future__ import annotations

import argparse
import os

import numpy as np

from run.run import train_ours
from utils.utils import get_scores

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CGAD PyG refactor without DGL")
    parser.add_argument("--expid", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="inj_cora")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Directory containing <dataset>.pt/.json/.txt")
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "max", "min", "weighted_sum"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--subgraph_size", type=int, default=4)
    parser.add_argument("--auc_test_rounds", type=int, default=100)
    parser.add_argument("--num_community", type=int, default=3)
    parser.add_argument("--neg_sample_method", type=str, default="bias", choices=["bias", "even", "random"])
    parser.add_argument("--num_negs", type=int, default=3)
    parser.add_argument("--strategy", type=str, default="most-relevant", choices=["random", "most-relevant", "least-relevant"])
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of node-level contrastive loss/score")
    parser.add_argument("--loss_fun", type=str, default="rnce", choices=["rnce"])
    parser.add_argument("--lam", type=float, default=0.5, help="Negative-pair weight in RNCE")
    parser.add_argument("--T", type=float, default=1.0, help="Contrastive temperature")
    parser.add_argument("--q", type=float, default=0.3, help="RNCE q parameter")
    parser.add_argument("--restart_prob", type=float, default=0.9, help="RWR restart probability")
    parser.add_argument("--rwr_max_steps", type=int, default=None, help="Max walk steps per seed; default=max(20, subgraph_size*10)")
    return parser


def apply_dataset_defaults(args):
    """Keep the original dataset-specific defaults while allowing CLI overrides."""
    if args.dataset == "inj_cora":
        args.num_community = 10
    elif args.dataset == "books":
        args.num_community = 10
    elif args.dataset == "disney":
        args.num_community = 3
    elif args.dataset == "reddit":
        args.num_epoch = min(args.num_epoch, 10)
        args.auc_test_rounds = min(args.auc_test_rounds, 10)
        args.num_community = 3
    return args


def main() -> None:
    args = apply_dataset_defaults(build_parser().parse_args())
    labels, scores = train_ours(args)
    k = int(np.sum(labels))
    auc, ap, recall = get_scores(labels, scores, k)
    print("\nFinal metric")
    print(f"AUC: {auc:.6f}")
    print(f"AUPRC/AP: {ap:.6f}")
    print(f"Recall@K: {recall:.6f}")


if __name__ == "__main__":
    main()
