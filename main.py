import argparse
import csv
import math
import os
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

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


CSV_FIELDS = [
    "datetime",
    "dataset",
    "runs",
    "auc",
    "auprc",
    "num_epoch",
    "epochs_trained",
    "best_epoch",
    "recall_at_k",
    "auc_mean",
    "auc_std",
    "auc_var",
    "auc_max",
    "auprc_mean",
    "auprc_std",
    "auprc_var",
    "auprc_max",
    "recall_mean",
    "recall_std",
    "recall_var",
    "recall_max",
    "lr",
    "batch_size",
    "embedding_dim",
    "patience",
    "weight_decay",
    "subgraph_size",
    "auc_test_rounds",
    "neg_sample_method",
    "num_negs",
    "strategy",
    "alpha",
    "lam",
    "T",
    "q",
    "seed",
    "seeds",
    "community_method",
    "max_communities",
    "rwr_restart_prob",
    "subgraph_cache_rounds",
    "data_dir",
    "train_dir",
]


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
    parser.add_argument("--tqdm", action="store_true")
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


def _finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _stat(values: Iterable[float]) -> Dict[str, float]:
    arr = _finite_array(values)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "var": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "var": float(np.var(arr, ddof=0)),
        "max": float(np.max(arr)),
    }


def _fmt_float(value: float, digits: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _fmt_metric(values: Iterable[float]) -> str:
    """Format metrics as mean±std(max) in percentage, e.g. 90.21±2.33(91.00)."""
    s = _stat(values)
    if not all(math.isfinite(s[k]) for k in ("mean", "std", "max")):
        return "nan±nan(nan)"
    return f"{s['mean'] * 100:.2f}±{s['std'] * 100:.2f}({s['max'] * 100:.2f})"


def _fmt_epoch(values: Iterable[float]) -> str:
    arr = _finite_array(values)
    if arr.size == 0:
        return "nan±nan(nan)"
    return f"{np.mean(arr):.2f}±{np.std(arr, ddof=0):.2f}({np.max(arr):.0f})"


def append_result_csv(path, row):
    if path is None:
        return
    path = _expand_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # If an old result CSV exists with a different header, migrate it once so
    # newly appended rows keep correct column alignment.
    if path.exists() and path.stat().st_size > 0:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            old_rows = list(reader)
        if old_fields != CSV_FIELDS:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                for old_row in old_rows:
                    writer.writerow({field: old_row.get(field, "") for field in CSV_FIELDS})

    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _evaluate_run_scores(ano_label: np.ndarray, run_scores: List[np.ndarray]):
    metrics = []
    k = int(np.sum(ano_label))
    for run_idx, score in enumerate(run_scores, start=1):
        auc, auprc, recall = get_scores(ano_label, score, k)
        metrics.append({"run": run_idx, "auc": auc, "auprc": auprc, "recall_at_k": recall})
    return metrics


def _build_summary_row(args, run_metrics, run_infos):
    auc_values = [item["auc"] for item in run_metrics]
    auprc_values = [item["auprc"] for item in run_metrics]
    recall_values = [item["recall_at_k"] for item in run_metrics]
    epoch_values = [item.get("epochs_trained", float("nan")) for item in run_infos]
    best_epoch_values = [item.get("best_epoch", float("nan")) + 1 for item in run_infos]
    seeds = [str(item.get("seed", "")) for item in run_infos]

    auc_stat = _stat(auc_values)
    auprc_stat = _stat(auprc_values)
    recall_stat = _stat(recall_values)

    return {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": args.dataset,
        "runs": int(getattr(args, "runs", 1)),
        "num_epoch": int(getattr(args, "num_epoch", 100)),
        "epochs_trained": _fmt_epoch(epoch_values),
        "best_epoch": _fmt_epoch(best_epoch_values),
        "auc": _fmt_metric(auc_values),
        "auprc": _fmt_metric(auprc_values),
        "recall_at_k": _fmt_metric(recall_values),
        "auc_mean": _fmt_float(auc_stat["mean"]),
        "auc_std": _fmt_float(auc_stat["std"]),
        "auc_var": _fmt_float(auc_stat["var"]),
        "auc_max": _fmt_float(auc_stat["max"]),
        "auprc_mean": _fmt_float(auprc_stat["mean"]),
        "auprc_std": _fmt_float(auprc_stat["std"]),
        "auprc_var": _fmt_float(auprc_stat["var"]),
        "auprc_max": _fmt_float(auprc_stat["max"]),
        "recall_mean": _fmt_float(recall_stat["mean"]),
        "recall_std": _fmt_float(recall_stat["std"]),
        "recall_var": _fmt_float(recall_stat["var"]),
        "recall_max": _fmt_float(recall_stat["max"]),
        "lr": getattr(args, "lr", ""),
        "batch_size": getattr(args, "batch_size", ""),
        "embedding_dim": getattr(args, "embedding_dim", ""),
        "patience": getattr(args, "patience", ""),
        "weight_decay": getattr(args, "weight_decay", ""),
        "subgraph_size": getattr(args, "subgraph_size", ""),
        "auc_test_rounds": getattr(args, "auc_test_rounds", ""),
        "neg_sample_method": getattr(args, "neg_sample_method", ""),
        "num_negs": getattr(args, "num_negs", ""),
        "strategy": getattr(args, "strategy", ""),
        "alpha": getattr(args, "alpha", ""),
        "lam": getattr(args, "lam", ""),
        "T": getattr(args, "T", ""),
        "q": getattr(args, "q", ""),
        "seed": getattr(args, "seed", ""),
        "seeds": ";".join(seeds),
        "community_method": getattr(args, "community_method", ""),
        "max_communities": getattr(args, "max_communities", ""),
        "rwr_restart_prob": getattr(args, "rwr_restart_prob", ""),
        "subgraph_cache_rounds": getattr(args, "subgraph_cache_rounds", ""),
        "data_dir": getattr(args, "data_dir", ""),
        "train_dir": getattr(args, "train_dir", ""),
    }


def main():
    args = parse_args()
    if args.cache_dir is None:
        args.cache_dir = str(_expand_path(args.train_dir) / "cgad_preprocess")

    if args.grid_search:
        lrs, batch_sizes, embedding_dims = dataset_default_grid(args)
    else:
        lrs, batch_sizes, embedding_dims = [args.lr], [args.batch_size], [args.embedding_dim]

    summary_rows = []
    for lr in lrs:
        for batch_size in batch_sizes:
            for embedding_dim in embedding_dims:
                cur_args = deepcopy(args)
                cur_args.lr = lr
                cur_args.batch_size = batch_size
                cur_args.embedding_dim = embedding_dim
                cur_args.return_run_scores = True

                print("\n==============================")
                print(
                    f"dataset={cur_args.dataset}, runs={cur_args.runs}, lr={cur_args.lr}, "
                    f"batch_size={cur_args.batch_size}, embedding_dim={cur_args.embedding_dim}, "
                    f"subgraph_cache_rounds={cur_args.subgraph_cache_rounds}"
                )

                ano_label, _, run_scores, run_infos = train_ours(cur_args)
                run_metrics = _evaluate_run_scores(ano_label, run_scores)
                row = _build_summary_row(cur_args, run_metrics, run_infos)
                append_result_csv(cur_args.results_csv, row)
                summary_rows.append(row)

                print(
                    f"Summary: AUC={row['auc']}, AUPRC={row['auprc']}, "
                    f"Recall@K={row['recall_at_k']}, epochs={row['epochs_trained']}"
                )
                print(f"Saved result to: {_expand_path(cur_args.results_csv)}")

    if summary_rows:
        print("\nFinal summaries:")
        for row in summary_rows:
            print(
                f"dataset={row['dataset']}, lr={row['lr']}, batch_size={row['batch_size']}, "
                f"embedding_dim={row['embedding_dim']}, AUC={row['auc']}, AUPRC={row['auprc']}"
            )


if __name__ == "__main__":
    main()
