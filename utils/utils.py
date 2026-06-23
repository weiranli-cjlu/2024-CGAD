"""Utility functions for CGAD without DGL.

This module replaces ``dgl.contrib.sampling.random_walk_with_restart`` with a
PyG edge_index based random-walk-with-restart sampler. The sampler runs on CPU
because subgraph sampling is stochastic control logic and does not benefit much
from moving many small random choices to GPU. Training tensors are still moved to
``args.device``.
"""

from __future__ import annotations

import heapq
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableSequence, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import remove_isolated_nodes, to_undirected


def preprocess_features(features: sp.spmatrix) -> np.matrix:
    """Row-normalize a scipy sparse feature matrix."""
    rowsum = np.asarray(features.sum(1)).reshape(-1)
    r_inv = np.zeros_like(rowsum, dtype=float)
    nonzero = rowsum != 0
    r_inv[nonzero] = np.power(rowsum[nonzero], -1)
    r_inv[~np.isfinite(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features).todense()


def normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    """Symmetrically normalize adjacency matrix: D^{-1/2} A D^{-1/2}."""
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(1)).reshape(-1)
    d_inv_sqrt = np.zeros_like(rowsum, dtype=float)
    nonzero = rowsum != 0
    d_inv_sqrt[nonzero] = np.power(rowsum[nonzero], -0.5)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def build_neighbor_lists(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    undirected: bool = True,
) -> List[List[int]]:
    """Build CPU adjacency lists from a PyG ``edge_index`` tensor.

    Parameters
    ----------
    edge_index:
        PyG COO graph connectivity with shape [2, num_edges].
    num_nodes:
        Number of nodes in the graph.
    undirected:
        Whether to symmetrize the graph before random walks. The original CGAD
        code converted scipy adjacency to a NetworkX graph and then a DGLGraph;
        for common attributed-network datasets this behaves as an undirected
        neighborhood sampler. Keeping this default improves compatibility.
    """
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

    edge_index = edge_index.detach().cpu().long()
    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    src, dst = edge_index.tolist()
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
            neighbors[u].append(v)

    # De-duplicate while preserving deterministic order under a fixed edge_index.
    for node in range(num_nodes):
        if len(neighbors[node]) > 1:
            neighbors[node] = list(dict.fromkeys(neighbors[node]))
    return neighbors


def _fallback_expand(
    seed: int,
    neighbors: Sequence[Sequence[int]],
    selected: MutableSequence[int],
    selected_set: set[int],
    target_size: int,
) -> None:
    """Fill a subgraph with 1-hop/2-hop neighbors if RWR under-samples."""
    for hop1 in neighbors[seed]:
        if hop1 != seed and hop1 not in selected_set:
            selected.append(hop1)
            selected_set.add(hop1)
            if len(selected) >= target_size:
                return

    for hop1 in neighbors[seed]:
        for hop2 in neighbors[hop1]:
            if hop2 != seed and hop2 not in selected_set:
                selected.append(hop2)
                selected_set.add(hop2)
                if len(selected) >= target_size:
                    return


def _single_rwr_subgraph(
    seed: int,
    neighbors: Sequence[Sequence[int]],
    subgraph_size: int,
    restart_prob: float,
    max_steps: int,
) -> List[int]:
    """Sample one fixed-length subgraph and put the target node at the end."""
    context_size = subgraph_size - 1
    if context_size <= 0:
        return [seed]

    selected: List[int] = []
    selected_set: set[int] = set()
    cur = seed

    for _ in range(max_steps):
        cur_neighbors = neighbors[cur]
        if (not cur_neighbors) or random.random() < restart_prob:
            cur = seed
        else:
            cur = random.choice(cur_neighbors)

        if cur != seed and cur not in selected_set:
            selected.append(cur)
            selected_set.add(cur)
            if len(selected) >= context_size:
                break

    if len(selected) < context_size:
        _fallback_expand(seed, neighbors, selected, selected_set, context_size)

    # Keep tensor shapes stable for isolated or tiny components. Repeating the
    # available context is preferable to failing; if there is no context, repeat
    # the target, matching the original code's padding behavior.
    if not selected:
        selected = [seed]
    while len(selected) < context_size:
        selected.append(selected[len(selected) % len(selected)])

    return selected[:context_size] + [seed]


def generate_rwr_subgraph(
    edge_index_or_neighbors: torch.Tensor | Sequence[Sequence[int]],
    num_nodes_or_subgraph_size: int,
    subgraph_size: int | None = None,
    *,
    restart_prob: float = 0.9,
    max_steps: int | None = None,
    undirected: bool = True,
) -> List[List[int]]:
    """Generate RWR subgraphs using PyG ``edge_index`` instead of DGL.

    Two calling styles are supported:

    1. ``generate_rwr_subgraph(edge_index, num_nodes, subgraph_size)``
    2. ``generate_rwr_subgraph(neighbor_lists, subgraph_size)``

    Returns
    -------
    list[list[int]]
        One subgraph per target node. Each subgraph has length ``subgraph_size``;
        the target node is always the final element.
    """
    if subgraph_size is None:
        neighbors = edge_index_or_neighbors  # type: ignore[assignment]
        subgraph_size = int(num_nodes_or_subgraph_size)
        num_nodes = len(neighbors)  # type: ignore[arg-type]
    else:
        edge_index = edge_index_or_neighbors  # type: ignore[assignment]
        num_nodes = int(num_nodes_or_subgraph_size)
        neighbors = build_neighbor_lists(edge_index, num_nodes, undirected=undirected)  # type: ignore[arg-type]

    if subgraph_size < 1:
        raise ValueError("subgraph_size must be at least 1")

    if max_steps is None:
        max_steps = max(20, subgraph_size * 10)

    return [
        _single_rwr_subgraph(
            seed=node,
            neighbors=neighbors,  # type: ignore[arg-type]
            subgraph_size=subgraph_size,
            restart_prob=restart_prob,
            max_steps=max_steps,
        )
        for node in range(num_nodes)
    ]


def recall_at_k(y_true: Sequence[int], scores: Sequence[float], k: int) -> float:
    """Compute recall@k for anomaly scores."""
    y_true_arr = np.asarray(y_true).astype(bool)
    scores_arr = np.asarray(scores)
    k = int(max(1, min(k, len(scores_arr))))
    positives = int(y_true_arr.sum())
    if positives == 0:
        return 0.0
    topk = np.argsort(scores_arr)[-k:]
    return float(y_true_arr[topk].sum() / positives)


def get_scores(actual: Sequence[int], score: Sequence[float], k: int) -> tuple[float, float, float]:
    """Return ROC-AUC, AUPRC/AP, and recall@k without relying on PyGOD."""
    actual_arr = np.asarray(actual).astype(int)
    score_arr = np.asarray(score, dtype=float)
    if len(np.unique(actual_arr)) < 2:
        auc = 0.0
    else:
        auc = float(roc_auc_score(actual_arr, score_arr))
    ap = float(average_precision_score(actual_arr, score_arr))
    rec = recall_at_k(actual_arr, score_arr, k)
    return auc, ap, rec


def get_one_sample(
    subgraph_size: int,
    nb_nodes: int,
    coef: Sequence[Sequence[float]],
    subgraphs: Sequence[Sequence[int]],
    strategy: str,
) -> np.ndarray:
    """Select one positive neighbor position inside every sampled subgraph."""
    all_samples: List[int] = []
    for nd in range(nb_nodes):
        candidates = list(dict.fromkeys(subgraphs[nd]))
        candidates = [node for node in candidates if node != nd]
        if not candidates:
            candidates = [nd]

        if strategy == "random":
            chosen = random.choice(candidates)
        elif strategy == "most-relevant":
            values = [coef[nd][neb] for neb in candidates]
            chosen = candidates[int(np.argmax(values))]
        elif strategy == "least-relevant":
            values = [coef[nd][neb] for neb in candidates]
            chosen = candidates[int(np.argmin(values))]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        all_samples.append(list(subgraphs[nd]).index(chosen))
    return np.asarray(all_samples, dtype=np.int64)


def RemoveIsolated(data):
    """Remove isolated nodes and keep node-level tensors aligned."""
    num_nodes = data.num_nodes
    edge_attr = getattr(data, "edge_attr", None)
    data.edge_index, data.edge_attr, mask = remove_isolated_nodes(
        data.edge_index, edge_attr, num_nodes=num_nodes
    )
    data.num_nodes = int(mask.sum())

    for key, item in data:
        if bool(re.search("edge", key)):
            continue
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            data[key] = item[mask]
    return data


def _sample_from_positions(positions: Sequence[int], forbidden: int | None = None) -> int:
    valid = [pos for pos in positions if pos != forbidden]
    if not valid:
        return forbidden if forbidden is not None else 0
    return random.choice(valid)


def get_negs(
    idx: Sequence[int],
    nodecom: Sequence[int],
    communities: Sequence[int],
    com_size: Mapping[int, int],
    num_negs: int,
    neg_sample_method: str,
) -> np.ndarray:
    """Sample negative nodes as local batch indices.

    ``multi_neg_node`` has shape [num_negs, batch_size], matching the original
    CGAD model indexing logic.
    """
    batch_size = len(idx)
    if batch_size == 0:
        return np.empty((num_negs, 0), dtype=np.int64)

    all_positions = list(range(batch_size))
    if batch_size == 1:
        return np.zeros((num_negs, 1), dtype=np.int64)

    if neg_sample_method == "random":
        negs = [
            [_sample_from_positions(all_positions, forbidden=pos) for pos in all_positions]
            for _ in range(num_negs)
        ]
        return np.asarray(negs, dtype=np.int64)

    positions_by_com: Dict[int, List[int]] = defaultdict(list)
    for local_pos, node in enumerate(idx):
        positions_by_com[int(nodecom[node])].append(local_pos)

    negs_per_node: List[List[int]] = []
    for local_pos, node in enumerate(idx):
        own_com = int(nodecom[node])
        available_coms = [
            com for com in communities
            if com != own_com and len(positions_by_com.get(int(com), [])) > 0
        ]
        if not available_coms:
            negs_per_node.append([
                _sample_from_positions(all_positions, forbidden=local_pos)
                for _ in range(num_negs)
            ])
            continue

        if neg_sample_method == "bias":
            weights = np.asarray([com_size[int(com)] for com in available_coms], dtype=float)
            weights = weights / weights.sum()
        elif neg_sample_method == "even":
            weights = np.ones(len(available_coms), dtype=float) / len(available_coms)
        else:
            raise ValueError(f"Unknown neg_sample_method: {neg_sample_method}")

        sampled = []
        for _ in range(num_negs):
            chosen_com = int(random.choices(list(available_coms), weights=weights.tolist(), k=1)[0])
            sampled.append(random.choice(positions_by_com[chosen_com]))
        negs_per_node.append(sampled)

    return np.asarray(negs_per_node, dtype=np.int64).T
