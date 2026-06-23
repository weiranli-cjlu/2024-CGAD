"""Training and inference loop for DGL-free CGAD."""

from __future__ import annotations

import json
import math
import os
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

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
    normalize_adj,
    preprocess_features,
)


def rnce_loss(logits: torch.Tensor, lam: float, q: float) -> torch.Tensor:
    exps = torch.exp(logits)
    pos = -(exps[:, 0]) ** q / q
    neg = ((lam * exps.sum(dim=1)) ** q) / q
    return pos.mean() + neg.mean()


# Backward-compatible name used by some scripts.
Rnce_loss = rnce_loss


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_dataset_file(data_dir: str | os.PathLike[str], dataset: str, suffix: str) -> Path:
    path = Path(data_dir).expanduser() / f"{dataset}{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return path


def _load_inputs(args):
    data_dir = getattr(args, "data_dir", ".")
    data_path = _resolve_dataset_file(data_dir, args.dataset, ".pt")
    json_path = _resolve_dataset_file(data_dir, args.dataset, ".json")
    coef_path = _resolve_dataset_file(data_dir, args.dataset, ".txt")

    try:
        data = torch.load(data_path, weights_only=False)
    except TypeError:
        data = torch.load(data_path)

    if data.has_isolated_nodes():
        data = RemoveIsolated(data)

    adj_sparse = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    features_sparse = sp.lil_matrix(data.x.detach().cpu().numpy())
    labels = data.y.detach().cpu().numpy().astype(bool)

    features = preprocess_features(features_sparse)
    nb_nodes, ft_size = features.shape

    adj_norm = normalize_adj(adj_sparse)
    adj_dense = torch.as_tensor((adj_norm + sp.eye(adj_norm.shape[0])).todense(), dtype=torch.float32)
    features_dense = torch.as_tensor(np.asarray(features), dtype=torch.float32)

    with open(json_path, encoding="utf8") as f:
        nodecom = json.loads(f.readline())["com"]

    if len(nodecom) != nb_nodes:
        raise ValueError(
            f"Community file has {len(nodecom)} nodes, but graph has {nb_nodes} nodes after preprocessing."
        )

    with open(coef_path, "rb") as f:
        coef = pickle.load(f)

    communities = sorted(dict(Counter(nodecom)))
    com_size = dict(Counter(nodecom))
    comnode = {
        com: [node for node, node_com in enumerate(nodecom) if node_com == com]
        for com in communities
    }

    neighbors = build_neighbor_lists(data.edge_index, nb_nodes, undirected=True)

    return {
        "data": data,
        "labels": labels,
        "adj_dense": adj_dense,
        "features_dense": features_dense,
        "nb_nodes": nb_nodes,
        "ft_size": ft_size,
        "nodecom": nodecom,
        "communities": communities,
        "com_size": com_size,
        "comnode": comnode,
        "coef": coef,
        "neighbors": neighbors,
    }


def _iter_community_batches(comnode: dict[int, list[int]], communities: Sequence[int], batch_num: int):
    comnode_epoch = {com: nodes.copy() for com, nodes in comnode.items()}
    for nodes in comnode_epoch.values():
        random.shuffle(nodes)

    com_batch_sizes = {
        com: max(1, len(comnode_epoch[com]) // batch_num) if len(comnode_epoch[com]) > 0 else 0
        for com in communities
    }

    for batch_idx in range(batch_num):
        is_final_batch = batch_idx == batch_num - 1
        idx: list[int] = []
        for com in communities:
            step = com_batch_sizes[com]
            if step == 0:
                continue
            start = batch_idx * step
            end = None if is_final_batch else (batch_idx + 1) * step
            idx.extend(comnode_epoch[com][start:end])
        if idx:
            random.shuffle(idx)
            yield idx


def _build_batch_tensors(
    idx: Sequence[int],
    subgraphs: Sequence[Sequence[int]],
    adj_dense: torch.Tensor,
    features_dense: torch.Tensor,
    subgraph_size: int,
    ft_size: int,
    device: torch.device,
):
    batch_nodes = torch.as_tensor([subgraphs[i] for i in idx], dtype=torch.long, device=device)
    features_device = features_dense.to(device)
    adj_device = adj_dense.to(device)

    bf = features_device[batch_nodes]
    rows = batch_nodes.unsqueeze(2).expand(-1, subgraph_size, subgraph_size)
    cols = batch_nodes.unsqueeze(1).expand(-1, subgraph_size, subgraph_size)
    ba = adj_device[rows, cols]

    added_feat_zero_row = torch.zeros((len(idx), 1, ft_size), device=device)
    bf_mask = torch.cat((bf[:, :-1, :], added_feat_zero_row), dim=1)
    ba_mask = torch.eye(subgraph_size, device=device).expand(len(idx), subgraph_size, subgraph_size)
    return bf_mask, ba, bf, ba_mask


def _sample_subgraphs_and_positive(args, state):
    subgraphs = generate_rwr_subgraph(
        state["neighbors"],
        args.subgraph_size,
        restart_prob=getattr(args, "restart_prob", 0.9),
        max_steps=getattr(args, "rwr_max_steps", None),
    )
    all_samples = get_one_sample(
        args.subgraph_size,
        state["nb_nodes"],
        state["coef"],
        subgraphs,
        args.strategy,
    )
    return subgraphs, all_samples


def train_ours(args):
    print(f"Dataset: {args.dataset}", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    state = _load_inputs(args)

    batch_size = int(args.batch_size)
    nb_nodes = int(state["nb_nodes"])
    ft_size = int(state["ft_size"])
    batch_num = max(1, math.ceil(nb_nodes / batch_size))

    final_score = None
    for run in range(int(args.runs)):
        seed = int(getattr(args, "seed", 1)) + run
        set_seed(seed)
        print(f"--- Train run {run + 1}/{args.runs}, seed={seed} ---", flush=True)

        model = Model(ft_size, args.embedding_dim, "prelu", args.readout, args.T).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best = float("inf")
        cnt_wait = 0

        for epoch in range(int(args.num_epoch)):
            model.train()
            subgraphs, all_samples = _sample_subgraphs_and_positive(args, state)
            epoch_loss = 0.0
            seen_nodes = 0

            for idx in _iter_community_batches(state["comnode"], state["communities"], batch_num):
                optimiser.zero_grad()
                multi_neg_node = get_negs(
                    idx,
                    state["nodecom"],
                    state["communities"],
                    state["com_size"],
                    args.num_negs,
                    args.neg_sample_method,
                )
                bf_mask, ba, bf, ba_mask = _build_batch_tensors(
                    idx,
                    subgraphs,
                    state["adj_dense"],
                    state["features_dense"],
                    args.subgraph_size,
                    ft_size,
                    device,
                )
                sample_node = all_samples[idx]
                node_logits, sub_logits, _ = model(bf_mask, ba, bf, ba_mask, multi_neg_node, sample_node)

                if args.loss_fun != "rnce":
                    raise ValueError(f"Unsupported loss_fun: {args.loss_fun}")
                node_loss = rnce_loss(node_logits, lam=args.lam, q=args.q)
                sub_loss = rnce_loss(sub_logits, lam=args.lam, q=args.q)
                loss = args.alpha * node_loss + (1.0 - args.alpha) * sub_loss
                loss.backward()
                optimiser.step()

                cur_batch_size = len(idx)
                epoch_loss += float(loss.detach().cpu()) * cur_batch_size
                seen_nodes += cur_batch_size

            mean_loss = epoch_loss / max(1, seen_nodes)
            print(f"Epoch:{epoch} Loss:{mean_loss:.8f}", flush=True)

            if mean_loss < best:
                best = mean_loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait >= int(args.patience):
                    print("Early stopping!", flush=True)
                    break

        final_score = _inference(args, state, model, batch_num, ft_size, device)

    return state["labels"], final_score


def _inference(args, state, model: Model, batch_num: int, ft_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    nb_nodes = state["nb_nodes"]
    multi_round_ano_score = np.zeros((int(args.auc_test_rounds), nb_nodes), dtype=float)
    print("--- Test now ---", flush=True)

    with torch.no_grad():
        for round_id in range(int(args.auc_test_rounds)):
            subgraphs, all_samples = _sample_subgraphs_and_positive(args, state)
            for idx in _iter_community_batches(state["comnode"], state["communities"], batch_num):
                multi_neg_node = get_negs(
                    idx,
                    state["nodecom"],
                    state["communities"],
                    state["com_size"],
                    args.num_negs,
                    args.neg_sample_method,
                )
                bf_mask, ba, bf, ba_mask = _build_batch_tensors(
                    idx,
                    subgraphs,
                    state["adj_dense"],
                    state["features_dense"],
                    args.subgraph_size,
                    ft_size,
                    device,
                )
                sample_node = all_samples[idx]
                node_logits, sub_logits, _ = model(bf_mask, ba, bf, ba_mask, multi_neg_node, sample_node)

                node_logits_np = node_logits.detach().cpu().numpy()
                sub_logits_np = sub_logits.detach().cpu().numpy()

                node_score = node_logits_np[:, 1:] - node_logits_np[:, [0]]
                node_score = np.mean(node_score, axis=1) + np.std(node_score, axis=1)

                sub_score = sub_logits_np[:, 1:] - sub_logits_np[:, [0]]
                sub_score = np.mean(sub_score, axis=1) + np.std(sub_score, axis=1)

                ano_score = args.alpha * node_score + (1.0 - args.alpha) * sub_score
                multi_round_ano_score[round_id, idx] = ano_score

    ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
    return ano_score_final - np.min(ano_score_final)
