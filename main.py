import argparse
import csv
import os
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

from run.run import train_ours
from utils.utils import get_scores

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


def _expand_path(path_like):
    if path_like is None:
        return None
    return Path(os.path.expanduser(str(path_like))).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="CGAD PyG + .mat refactor")

    parser.add_argument("--expid", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat")

    # Training outputs.  If --cache_dir is omitted, preprocessing cache is saved under
    # <train_dir>/cgad_preprocess by utils.resolve_preprocess_paths.
    parser.add_argument("--train_dir", type=str, default="./runs")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--force_preprocess", action="store_true")

    # CSV summary and optional y_true/y_score export.
    parser.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="CSV file used to append experiment summaries. Default: <train_dir>/cgad_results.csv",
    )
    parser.add_argument(
        "--save_score_run",
        type=int,
        default=-1,
        help="1-based run id whose y_true/y_score will be saved. <=0 means do not save.",
    )
    parser.add_argument(
        "--score_save_dir",
        type=str,
        default=None,
        help="Directory for y_true/y_score CSV files. Default: <train_dir>/y_true_y_score",
    )

    parser.add_argument("--community_method", type=str, default="louvain", choices=["louvain", "greedy", "components"])
    parser.add_argument("--max_communities", type=int, default=0, help="0 means keep all generated communities")
    parser.add_argument("--readout", type=str, default="avg")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--subgraph_size", type=int, default=4)
    parser.add_argument("--auc_test_rounds", type=int, default=100)
    parser.add_argument("--num_community", type=int, default=3)  # kept for compatibility
    parser.add_argument("--neg_sample_method", type=str, default="bias", choices=["bias", "even", "random"])
    parser.add_argument("--num_negs", type=int, default=3)
    parser.add_argument("--strategy", type=str, default="most-relevant", choices=["random", "most-relevant", "least-relevant"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--loss_fun", type=str, default="rnce")
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=0.3)
    parser.add_argument("--grid_search", action="store_true", help="Run original-style small grid search")
    return parser


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
    if args.dataset in ["books"]:
        return lrs2, bs1, ems2
    if args.dataset in ["disney"]:
        return lrs2, bs2, ems2
    if args.dataset in ["reddit"]:
        args.num_epoch = min(args.num_epoch, 10)
        args.auc_test_rounds = min(args.auc_test_rounds, 10)
        return lrs1, bs1, ems3
    return [args.lr], [args.batch_size], [args.embedding_dim]


def _metric_summary(values, percent=True, decimals=2):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "nan±nan(nan)"
    if percent:
        arr = arr * 100.0
    return f"{np.mean(arr):.{decimals}f}±{np.std(arr):.{decimals}f}({np.max(arr):.{decimals}f})"


def _append_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _save_y_true_y_score(args, y_true, y_score, current_time, tag):
    if args.save_score_run <= 0:
        return ""

    save_dir = _expand_path(args.score_save_dir) if args.score_save_dir else _expand_path(args.train_dir) / "y_true_y_score"
    save_dir.mkdir(parents=True, exist_ok=True)
    safe_time = current_time.replace("-", "").replace(":", "").replace(" ", "_")
    filename = f"{args.dataset}_{tag}_run{args.save_score_run}_y_true_y_score_{safe_time}.csv"
    save_path = save_dir / filename

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "y_true", "y_score"])
        for node_id, (label, score) in enumerate(zip(y_true, y_score)):
            writer.writerow([node_id, int(label), float(score)])
    return str(save_path)


def _build_result_row(args, current_time, auc_values, auprc_values, recall_values, run_infos, saved_score_path):
    epochs_trained = [info.get("epochs_trained", np.nan) for info in run_infos]
    best_epochs = [info.get("best_epoch", np.nan) for info in run_infos]
    best_losses = [info.get("best_loss", np.nan) for info in run_infos]

    return {
        "datetime": current_time,
        "dataset": args.dataset,
        "runs": args.runs,
        "auc": _metric_summary(auc_values, percent=True, decimals=2),
        "auprc": _metric_summary(auprc_values, percent=True, decimals=2),
        "expid": args.expid,
        "seed": args.seed,
        "num_epoch": args.num_epoch,
        "epochs_trained": _metric_summary(epochs_trained, percent=False, decimals=2),
        "best_epoch": _metric_summary(best_epochs, percent=False, decimals=2),
        "best_loss": _metric_summary(best_losses, percent=False, decimals=6),
        "recall_at_k": _metric_summary(recall_values, percent=True, decimals=2),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "patience": args.patience,
        "subgraph_size": args.subgraph_size,
        "auc_test_rounds": args.auc_test_rounds,
        "readout": args.readout,
        "community_method": args.community_method,
        "max_communities": args.max_communities,
        "neg_sample_method": args.neg_sample_method,
        "num_negs": args.num_negs,
        "strategy": args.strategy,
        "alpha": args.alpha,
        "loss_fun": args.loss_fun,
        "lam": args.lam,
        "T": args.T,
        "q": args.q,
        "data_dir": str(_expand_path(args.data_dir)),
        "cache_dir": "" if args.cache_dir is None else str(_expand_path(args.cache_dir)),
        "saved_score_path": saved_score_path,
    }


def main():
    args = parse_args().parse_args()
    args.train_dir = str(_expand_path(args.train_dir))
    Path(args.train_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        args.cache_dir = str(_expand_path(args.cache_dir))

    results_csv = _expand_path(args.results_csv) if args.results_csv else Path(args.train_dir) / "cgad_results.csv"

    if args.grid_search:
        lrs, batch_sizes, embedding_dims = dataset_default_grid(args)
    else:
        lrs, batch_sizes, embedding_dims = [args.lr], [args.batch_size], [args.embedding_dim]

    rows = []
    for lr in lrs:
        for batch_size in batch_sizes:
            for embedding_dim in embedding_dims:
                run_args = deepcopy(args)
                run_args.lr = lr
                run_args.batch_size = batch_size
                run_args.embedding_dim = embedding_dim
                run_args.return_run_scores = True

                y_true, y_score_mean, run_score_list, run_infos = train_ours(run_args)
                if not run_score_list:
                    run_score_list = [y_score_mean]

                k = int(np.sum(y_true))
                auc_values, auprc_values, recall_values = [], [], []
                for y_score in run_score_list:
                    auc_value, auprc_value, recall_value = get_scores(y_true, y_score, k)
                    auc_values.append(auc_value)
                    auprc_values.append(auprc_value)
                    recall_values.append(recall_value)

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                tag = f"lr{lr:g}_bs{batch_size}_emb{embedding_dim}"
                saved_score_path = ""
                if 1 <= run_args.save_score_run <= len(run_score_list):
                    saved_score_path = _save_y_true_y_score(
                        run_args,
                        y_true,
                        run_score_list[run_args.save_score_run - 1],
                        current_time,
                        tag,
                    )

                row = _build_result_row(
                    run_args,
                    current_time,
                    auc_values,
                    auprc_values,
                    recall_values,
                    run_infos,
                    saved_score_path,
                )
                _append_csv(results_csv, row)
                rows.append(row)

    print("\nTraining finished. Summary:")
    for row in rows:
        print(
            f"[{row['datetime']}] dataset={row['dataset']} runs={row['runs']} "
            f"lr={row['lr']} batch_size={row['batch_size']} emb={row['embedding_dim']} "
            f"AUC={row['auc']} AUPRC={row['auprc']} Recall@K={row['recall_at_k']} "
            f"epochs={row['epochs_trained']}"
        )
        if row["saved_score_path"]:
            print(f"Saved y_true/y_score: {row['saved_score_path']}")
    print(f"CSV summary saved to: {results_csv}")


if __name__ == "__main__":
    main()
