import math
import os
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_scipy_sparse_matrix

from model.model import Model
from utils.utils import (
    RemoveIsolated,
    build_neighbor_lists,
    generate_rwr_subgraph,
    get_negs,
    get_one_sample,
    load_mat_data,
    load_or_generate_preprocess,
    normalize_adj,
    preprocess_features,
)


def Rnce_loss(logits, lam, q):
    exps = torch.exp(logits)
    pos = -(exps[:, 0]) ** q / q
    neg = ((lam * exps.sum(1)) ** q) / q
    return pos.mean() + neg.mean()


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _build_community_helpers(nodecom, nb_nodes):
    com_size = dict(Counter(nodecom))
    communities = sorted(com_size)

    com_size_ratio = []
    for com in communities:
        other_sums = nb_nodes - com_size[com]
        if other_sums <= 0:
            seqs = [1.0 for item in communities if item != com]
        else:
            seqs = [com_size[item] / other_sums for item in communities if item != com]
        com_size_ratio.append(seqs)

    comnode = []
    for item in communities:
        each_com_node = [nd for nd in range(len(nodecom)) if nodecom[nd] == item]
        comnode.append(each_com_node)
    return communities, com_size_ratio, comnode


def _iter_community_batches(comnode: List[List[int]], batch_num: int):
    """Yield balanced node batches without rebuilding repeated slice code."""
    com_batch_sizes = [max(len(item) // batch_num, 1) for item in comnode]
    for item in comnode:
        random.shuffle(item)

    for batch_idx in range(batch_num):
        is_final_batch = batch_idx == batch_num - 1
        if not is_final_batch:
            idx_nested = [
                comnode[j][batch_idx * com_batch_sizes[j] : (batch_idx + 1) * com_batch_sizes[j]]
                for j in range(len(comnode))
            ]
        else:
            idx_nested = [comnode[j][batch_idx * com_batch_sizes[j] :] for j in range(len(comnode))]
        idx = sum(idx_nested, [])
        if idx:
            yield batch_idx, is_final_batch, idx


def _build_batch_tensors(
    adj_base: torch.Tensor,
    features_base: torch.Tensor,
    subgraph_tensor: torch.Tensor,
    idx: List[int],
    mask_adj_base: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized subgraph tensor construction.

    Replaces the original Python loop + torch.cat pattern:
        for i in idx: adj[:, subgraphs[i], :][:, :, subgraphs[i]] ...
    This reduces CPU overhead and avoids many tiny tensor slices per batch.
    """
    device = features_base.device
    idx_tensor = torch.as_tensor(idx, dtype=torch.long, device=device)
    nodes = subgraph_tensor.index_select(0, idx_tensor)  # [B, S]
    bf = features_base[nodes]  # [B, S, F]
    ba = adj_base[nodes.unsqueeze(2), nodes.unsqueeze(1)]  # [B, S, S]
    bf_mask = bf.clone()
    bf_mask[:, -1, :] = 0.0
    ba_mask = mask_adj_base.expand(len(idx), -1, -1)
    return ba, bf, bf_mask, ba_mask


def _prepare_epoch_subgraphs(neighbor_lists, subgraph_size, nb_nodes, coef, strategy, device):
    subgraphs = generate_rwr_subgraph(neighbor_lists, subgraph_size)
    subgraph_tensor = torch.as_tensor(subgraphs, dtype=torch.long, device=device)
    all_samples = get_one_sample(subgraph_size, nb_nodes, coef, subgraphs, strategy)
    return subgraph_tensor, all_samples


def train_ours(args):
    """Train CGAD and return anomaly labels/scores.

    Printing inside the training and testing loops is intentionally removed.
    main.py prints only the final summary and appends the CSV result row.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = load_mat_data(args.dataset, args.data_dir)
    if data.has_isolated_nodes():
        data = RemoveIsolated(data)

    nodecom, coef, _, _ = load_or_generate_preprocess(data, args)

    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    features_sp = sp.lil_matrix(data.x.detach().cpu().numpy())
    y = data.y.bool()
    ano_label = np.asarray(y.detach().cpu().numpy()).astype(int)

    features = preprocess_features(features_sp)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    adj_norm = normalize_adj(adj)
    adj_dense = (adj_norm + sp.eye(adj_norm.shape[0], dtype=np.float32)).toarray().astype(np.float32)
    features_base = torch.as_tensor(features, dtype=torch.float32, device=device)
    adj_base = torch.as_tensor(adj_dense, dtype=torch.float32, device=device)
    mask_adj_base = torch.eye(args.subgraph_size, device=device)

    # Build once, reuse for every epoch/test round to avoid repeated edge_index scans.
    neighbor_lists = build_neighbor_lists(data.edge_index, data.num_nodes)

    communities, com_size_ratio, comnode = _build_community_helpers(nodecom, nb_nodes)
    if len(communities) < 2 and args.neg_sample_method != "random":
        args.neg_sample_method = "random"

    final_scores = []
    run_infos = []
    seeds = [args.seed + i for i in range(args.runs)]
    batch_size = args.batch_size
    subgraph_size = args.subgraph_size
    batch_num = max(math.ceil(nb_nodes / batch_size), 1)

    for run in range(args.runs):
        seed = seeds[run]
        set_seed(seed)

        model = Model(ft_size, args.embedding_dim, "prelu", args.readout, args.T).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        cnt_wait = 0
        best = float("inf")
        best_t = 0
        epochs_trained = 0

        for epoch in range(args.num_epoch):
            model.train()
            total_loss = 0.0
            total_seen = 0
            epochs_trained = epoch + 1

            subgraph_tensor, all_samples = _prepare_epoch_subgraphs(
                neighbor_lists, subgraph_size, nb_nodes, coef, args.strategy, device
            )

            for _, _, idx in _iter_community_batches(comnode, batch_num):
                random.shuffle(idx)
                cur_batch_size = len(idx)
                optimiser.zero_grad(set_to_none=True)

                multi_neg_node = get_negs(
                    idx,
                    nodecom,
                    communities,
                    com_size_ratio,
                    args.num_negs,
                    args.neg_sample_method,
                )

                ba, bf, bf_mask, ba_mask = _build_batch_tensors(
                    adj_base,
                    features_base,
                    subgraph_tensor,
                    idx,
                    mask_adj_base,
                )
                sample_node = all_samples[idx]

                node_logits, sub_logits, _ = model(
                    bf_mask,
                    ba,
                    bf,
                    ba_mask,
                    multi_neg_node,
                    sample_node,
                )

                node_loss = Rnce_loss(node_logits, lam=args.lam, q=args.q)
                sub_loss = Rnce_loss(sub_logits, lam=args.lam, q=args.q)
                loss = args.alpha * node_loss + (1 - args.alpha) * sub_loss

                loss.backward()
                optimiser.step()

                loss_value = float(loss.detach().cpu().item())
                total_loss += loss_value * cur_batch_size
                total_seen += cur_batch_size

            mean_loss = total_loss / max(total_seen, 1)
            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                break

        model.eval()
        multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes), dtype=np.float32)

        for round_id in range(args.auc_test_rounds):
            subgraph_tensor, all_samples = _prepare_epoch_subgraphs(
                neighbor_lists, subgraph_size, nb_nodes, coef, args.strategy, device
            )

            for _, _, idx in _iter_community_batches(comnode, batch_num):
                multi_neg_node = get_negs(
                    idx,
                    nodecom,
                    communities,
                    com_size_ratio,
                    args.num_negs,
                    args.neg_sample_method,
                )

                ba, bf, bf_mask, ba_mask = _build_batch_tensors(
                    adj_base,
                    features_base,
                    subgraph_tensor,
                    idx,
                    mask_adj_base,
                )

                with torch.no_grad():
                    sample_node = all_samples[idx]
                    node_logits, sub_logits, _ = model(
                        bf_mask,
                        ba,
                        bf,
                        ba_mask,
                        multi_neg_node,
                        sample_node,
                    )

                    node_logits_np = node_logits.detach().cpu().numpy()
                    sub_logits_np = sub_logits.detach().cpu().numpy()
                    node_score = node_logits_np[:, 1:] - node_logits_np[:, [0]]
                    node_score = np.mean(node_score, axis=1) + np.std(node_score, axis=1)
                    sub_score = sub_logits_np[:, 1:] - sub_logits_np[:, [0]]
                    sub_score = np.mean(sub_score, axis=1) + np.std(sub_score, axis=1)
                    ano_score = args.alpha * node_score + (1 - args.alpha) * sub_score
                    multi_round_ano_score[round_id, idx] = ano_score

        ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
        ano_score_final = ano_score_final - np.min(ano_score_final)
        final_scores.append(ano_score_final)
        run_infos.append(
            {
                "run": run + 1,
                "seed": seed,
                "epochs_trained": epochs_trained,
                "best_epoch": best_t,
                "best_loss": best,
            }
        )

    ano_score_mean = np.mean(np.vstack(final_scores), axis=0)
    if getattr(args, "return_run_scores", False):
        return ano_label, ano_score_mean, final_scores, run_infos
    return ano_label, ano_score_mean
