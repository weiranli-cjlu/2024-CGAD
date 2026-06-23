import argparse
import csv
import os
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# Limit OpenMP/MKL oversubscription before importing heavy numeric libraries.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from run.run import train_ours
from utils.utils import get_scores

warnings.filterwarnings("ignore", category=UserWarning)


def _expand_path(path_like):
    if path_like is None:
        return None
    return Path(os.path.expanduser(str(path_like))).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="CGAD CPU-optimized PyG/NumPy version")

    parser.add_argument("--expid", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat")

    # Output/cache paths.
    parser.add_argument("--train_dir", type=str, default="./runs")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--results_csv", type=str, default="runs/cgad_results.csv")
    parser.add_argument("--force_preprocess", action="store_true")

    # Community preprocessing.
    parser.add_argument("--community_method", type=str, default="louvain", choices=["louvain", "greedy", "components"])
    parser.add_argument("--max_communities", type=int, default=0, help="0 means keep all generated communities")
    parser.add_argument("--num_community", type=int, default=3, help="kept for compatibility")

    # CPU optimisation controls.
    parser.add_argument("--torch_threads", type=int, default=1, help="limit torch CPU threads to avoid CPU oversubscription")
    parser.add_argument("--subgraph_cache_rounds", type=int, default=16, help="number of RWR subgraph rounds generated/cached once")
    parser.add_argument("--disable_subgraph_cache", action="store_true", help="do not save/load RWR subgraph bank on disk")
    parser.add_argument("--force_subgraph_cache", action="store_true", help="regenerate cached RWR subgraph bank")
    parser.add_argument("--cap_test_rounds_to_cache", action="store_true", help="run at most subgraph_cache_rounds test rounds")
    parser.add_argument("--rwr_restart_prob", type=float, default=0.9)
    parser.add_argument("--rwr_max_retries", type=int, default=10)

    # Model/training.
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "max", "min", "weighted_sum"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--subgraph_size", type=int, default=4)
    parser.add_argument("--auc_test_rounds", type=int, default=100)
    parser.add_argument("--neg_sample_method", type=str, default="bias", choices=["bias", "even", "random"])
    parser.add_argument("--num_negs", type=int, default=3)
    parser.add_argument("--strategy", type=str, default="most-relevant", choices=["random", "most-relevant", "least-relevant"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--loss_fun", type=str, default="rnce")
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=0.3)

    parser.add_argument("--grid_search", action="store_true", help="Run original-style small grid search")
    return parser.parse_args()


def dataset_default_grid(args):
    # Keep the original hyper-parameter convention, but make grid search optional.
    lrs1 = [1e-3, 5e-4]
    lrs2 = [1e-2, 5e-3]
    bs1 = [512, 1024]
    bs2 = [32, 64]
    ems1 = [64, 128]
    ems2 = [12, 16]
    ems3 = [32, 48]

    if args.dataset in ["inj_cora"]:
        return lrs1, bs1, ems1
    if args.dataset in ["books", "book"]:
        return lrs2, bs1, ems2
    if args.dataset in ["disney"]:
        return lrs2, bs2, ems2
    if args.dataset in ["reddit"]:
        args.num_epoch = min(args.num_epoch, 10)
        args.auc_test_rounds = min(args.auc_test_rounds, 10)
        return lrs1, bs1, ems3
    return [args.lr], [args.batch_size], [args.embedding_dim]


def append_result_csv(path, row):
    if path is None:
        return
    path = _expand_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.cache_dir is None:
        args.cache_dir = str(_expand_path(args.train_dir) / "cgad_preprocess")

    if args.grid_search:
        lrs, batch_sizes, embedding_dims = dataset_default_grid(args)
    else:
        lrs, batch_sizes, embedding_dims = [args.lr], [args.batch_size], [args.embedding_dim]

    results = []
    for lr in lrs:
        for batch_size in batch_sizes:
            for embedding_dim in embedding_dims:
                cur_args = deepcopy(args)
                cur_args.lr = lr
                cur_args.batch_size = batch_size
                cur_args.embedding_dim = embedding_dim

                print("\n==============================")
                print(
                    f"dataset={cur_args.dataset}, lr={cur_args.lr}, "
                    f"batch_size={cur_args.batch_size}, embedding_dim={cur_args.embedding_dim}, "
                    f"subgraph_cache_rounds={cur_args.subgraph_cache_rounds}"
                )
                ano_label, ano_score_final = train_ours(cur_args)
                k = int(np.sum(ano_label))
                auc, auprc, recall = get_scores(ano_label, ano_score_final, k)
                print(f"AUC={auc:.4f}, AUPRC={auprc:.4f}, Recall@K={recall:.4f}")
                results.append([auc, auprc, recall])

                append_result_csv(
                    cur_args.results_csv,
                    {
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "dataset": cur_args.dataset,
                        "runs": cur_args.runs,
                        "num_epoch": cur_args.num_epoch,
                        "batch_size": cur_args.batch_size,
                        "embedding_dim": cur_args.embedding_dim,
                        "lr": cur_args.lr,
                        "auc": f"{auc:.6f}",
                        "auprc": f"{auprc:.6f}",
                        "recall_at_k": f"{recall:.6f}",
                    },
                )

    final_metric = np.mean(results, axis=0)
    print("Final mean [AUC, AUPRC, Recall@K]:", final_metric)


if __name__ == "__main__":
    main()
