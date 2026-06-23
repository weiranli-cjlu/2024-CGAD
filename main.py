import argparse
import os
import warnings

import numpy as np

from run.run import train_ours
from utils.utils import get_scores

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="CGAD PyG + .mat refactor")
    parser.add_argument("--expid", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat")
    parser.add_argument("--cache_dir", type=str, default="~/datasets/GAD/mat/cgad_preprocess")
    parser.add_argument("--force_preprocess", action="store_true")
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
    if args.dataset in ["books"]:
        return lrs2, bs1, ems2
    if args.dataset in ["disney"]:
        return lrs2, bs2, ems2
    if args.dataset in ["reddit"]:
        args.num_epoch = min(args.num_epoch, 10)
        args.auc_test_rounds = min(args.auc_test_rounds, 10)
        return lrs1, bs1, ems3
    return [args.lr], [args.batch_size], [args.embedding_dim]


def main():
    args = parse_args()
    ave_results = []

    if args.grid_search:
        lrs, batch_sizes, embedding_dims = dataset_default_grid(args)
    else:
        lrs, batch_sizes, embedding_dims = [args.lr], [args.batch_size], [args.embedding_dim]

    for lr in lrs:
        for batch_size in batch_sizes:
            for embedding_dim in embedding_dims:
                args.lr = lr
                args.batch_size = batch_size
                args.embedding_dim = embedding_dim
                print("\n==============================")
                print(f"lr={args.lr}, batch_size={args.batch_size}, embedding_dim={args.embedding_dim}")
                ano_label, ano_score_final = train_ours(args)
                k = int(np.sum(ano_label))
                auc, ap, recall = get_scores(ano_label, ano_score_final, k)
                print(f"AUC={auc:.4f}, AUPRC={ap:.4f}, Recall@K={recall:.4f}")
                ave_results.append([auc, ap, recall])

    final_metric = np.mean(ave_results, axis=0)
    print("Final mean [AUC, AUPRC, Recall@K]:", final_metric)


if __name__ == "__main__":
    main()
