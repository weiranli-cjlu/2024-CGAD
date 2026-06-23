import json
import os
import pickle
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from scipy.sparse import issparse
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes


MAT_ADJ_KEYS = ("Network", "network", "A", "adj", "Adj", "adjacency", "edges")
MAT_FEAT_KEYS = ("Attributes", "attributes", "X", "x", "features", "Features", "attrb", "attr")
MAT_LABEL_KEYS = ("Label", "label", "labels", "y", "Y", "gnd", "Class", "class")


def _first_existing_key(mat: Dict, keys: Sequence[str]):
    for key in keys:
        if key in mat:
            return key
    return None


def _to_numpy_or_sparse(value):
    if issparse(value):
        return value
    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return arr


def _as_csr_matrix(value) -> sp.csr_matrix:
    value = _to_numpy_or_sparse(value)
    if issparse(value):
        return value.tocsr()
    return sp.csr_matrix(np.asarray(value))


def _as_dense_float_array(value) -> np.ndarray:
    value = _to_numpy_or_sparse(value)
    if issparse(value):
        value = value.toarray()
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    return value


def _as_label_array(value, num_nodes: int) -> np.ndarray:
    value = _to_numpy_or_sparse(value)
    if issparse(value):
        value = value.toarray()
    value = np.asarray(value).squeeze()
    if value.ndim > 1:
        # For one-hot labels, use the positive/anomaly column if possible.
        if value.shape[0] == num_nodes:
            value = np.argmax(value, axis=1)
        else:
            value = value.reshape(-1)
    value = value.astype(np.int64)
    if value.shape[0] != num_nodes:
        raise ValueError(f"Label length {value.shape[0]} does not match num_nodes {num_nodes}.")
    # Some binary anomaly datasets use {-1, 1}; make them {0, 1}.
    uniq = set(np.unique(value).tolist())
    if uniq.issubset({-1, 1}):
        value = (value > 0).astype(np.int64)
    return value


def load_mat_data(dataset: str, data_dir: str = "~/datasets/GAD/mat") -> Data:
    """Load common GAD .mat datasets as a PyG Data object.

    Supported key aliases:
    - adjacency: Network/A/adj/adjacency
    - feature: Attributes/X/features/attrb
    - label: Label/y/gnd/Class
    """
    data_dir = Path(os.path.expanduser(data_dir))
    mat_path = data_dir / f"{dataset}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {mat_path}")

    mat = sio.loadmat(mat_path)
    adj_key = _first_existing_key(mat, MAT_ADJ_KEYS)
    feat_key = _first_existing_key(mat, MAT_FEAT_KEYS)
    label_key = _first_existing_key(mat, MAT_LABEL_KEYS)

    if adj_key is None:
        raise KeyError(f"No adjacency key found in {mat_path}. Tried: {MAT_ADJ_KEYS}")
    if feat_key is None:
        raise KeyError(f"No feature key found in {mat_path}. Tried: {MAT_FEAT_KEYS}")
    if label_key is None:
        raise KeyError(f"No label key found in {mat_path}. Tried: {MAT_LABEL_KEYS}")

    adj = _as_csr_matrix(mat[adj_key])
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()

    x = _as_dense_float_array(mat[feat_key])
    y = _as_label_array(mat[label_key], adj.shape[0])

    if x.shape[0] != adj.shape[0]:
        # Some .mat files store attributes transposed.
        if x.shape[1] == adj.shape[0]:
            x = x.T
        else:
            raise ValueError(
                f"Feature shape {x.shape} does not match adjacency shape {adj.shape}."
            )

    row, col = adj.nonzero()
    edge_index = torch.as_tensor(np.vstack([row, col]), dtype=torch.long)
    data = Data(
        x=torch.as_tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.as_tensor(y, dtype=torch.long),
    )
    data.num_nodes = int(adj.shape[0])
    return data


def preprocess_features(features):
    """Row-normalize feature matrix and return dense matrix."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def build_neighbor_lists(edge_index: torch.Tensor, num_nodes: int, undirected: bool = True) -> List[List[int]]:
    """Build CPU adjacency lists from PyG edge_index."""
    edge_index = edge_index.detach().cpu().long()
    neighbors = [set() for _ in range(num_nodes)]
    rows = edge_index[0].tolist()
    cols = edge_index[1].tolist()
    for src, dst in zip(rows, cols):
        if src == dst:
            continue
        neighbors[src].add(dst)
        if undirected:
            neighbors[dst].add(src)
    return [sorted(list(item)) for item in neighbors]


def _rwr_unique_trace(seed: int, neighbors: List[List[int]], restart_prob: float, max_steps: int) -> List[int]:
    cur = seed
    trace = [seed]
    for _ in range(max_steps):
        if random.random() < restart_prob or not neighbors[cur]:
            cur = seed
        else:
            cur = random.choice(neighbors[cur])
        trace.append(cur)
    # preserve order while removing duplicates
    seen = set()
    unique = []
    for node in trace:
        if node not in seen:
            seen.add(node)
            unique.append(node)
    return unique


def generate_rwr_subgraph(edge_index_or_neighbors, subgraph_size: int, num_nodes: int = None):
    """Generate RWR subgraphs using PyG edge_index instead of DGL.

    The returned subgraph keeps the original CGAD convention:
    the target node is placed at the final position of each subgraph.
    """
    if isinstance(edge_index_or_neighbors, torch.Tensor):
        if num_nodes is None:
            num_nodes = int(edge_index_or_neighbors.max().item()) + 1
        neighbors = build_neighbor_lists(edge_index_or_neighbors, num_nodes)
    else:
        neighbors = edge_index_or_neighbors
        num_nodes = len(neighbors)

    reduced_size = subgraph_size - 1
    subgraphs = []
    for node in range(num_nodes):
        unique_nodes = _rwr_unique_trace(
            node,
            neighbors,
            restart_prob=1.0,
            max_steps=max(subgraph_size * 3, 1),
        )
        retry_time = 0
        while len(unique_nodes) < reduced_size:
            unique_nodes = _rwr_unique_trace(
                node,
                neighbors,
                restart_prob=0.9,
                max_steps=max(subgraph_size * 5, 1),
            )
            retry_time += 1
            if len(unique_nodes) <= 2 and retry_time > 10:
                break

        candidate = [v for v in unique_nodes if v != node]
        if not candidate:
            candidate = [node]
        while len(candidate) < reduced_size:
            candidate.extend(candidate)
        candidate = candidate[:reduced_size]
        candidate.append(node)
        subgraphs.append(candidate)
    return subgraphs


def generate_coef(features: np.ndarray) -> np.ndarray:
    """Generate feature cosine-similarity matrix used by CGAD sample selection."""
    if sp.issparse(features):
        features = features.toarray()
    return cosine_similarity(np.asarray(features, dtype=np.float32))


def _largest_remainder_partition(nodecom: List[int], max_communities: int) -> List[int]:
    """Merge tiny surplus communities into [0, max_communities-1] ids."""
    if max_communities <= 0:
        return nodecom
    uniq = sorted(set(nodecom))
    if len(uniq) <= max_communities:
        remap = {cid: i for i, cid in enumerate(uniq)}
        return [remap[cid] for cid in nodecom]

    counts = Counter(nodecom)
    keep = [cid for cid, _ in counts.most_common(max_communities)]
    keep_set = set(keep)
    remap = {cid: i for i, cid in enumerate(keep)}
    merged = []
    for idx, cid in enumerate(nodecom):
        if cid in keep_set:
            merged.append(remap[cid])
        else:
            # Assign remaining communities to the nearest-size bucket deterministically.
            merged.append(idx % max_communities)
    return merged


def generate_community(edge_index: torch.Tensor, num_nodes: int, method: str = "louvain", seed: int = 1,
                       max_communities: int = 0) -> List[int]:
    """Generate community id for every node, replacing the original external json."""
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index.detach().cpu().long().t().tolist())

    communities = None
    if method == "louvain" and hasattr(nx.community, "louvain_communities"):
        communities = nx.community.louvain_communities(graph, seed=seed)
    elif method == "greedy":
        communities = nx.community.greedy_modularity_communities(graph)
    elif method == "components":
        communities = list(nx.connected_components(graph))
    else:
        # Safe fallback for older networkx versions.
        try:
            communities = nx.community.greedy_modularity_communities(graph)
        except Exception:
            communities = list(nx.connected_components(graph))

    nodecom = [0] * num_nodes
    for cid, nodes in enumerate(communities):
        for node in nodes:
            nodecom[int(node)] = int(cid)

    nodecom = _largest_remainder_partition(nodecom, max_communities)
    uniq = sorted(set(nodecom))
    remap = {cid: i for i, cid in enumerate(uniq)}
    return [remap[cid] for cid in nodecom]


def load_or_generate_preprocess(data: Data, args):
    """Load cached community/coef if available, otherwise generate them automatically."""
    cache_dir = Path(os.path.expanduser(args.cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    community_path = cache_dir / f"{args.dataset}.json"
    coef_path = cache_dir / f"{args.dataset}.pkl"

    if args.force_preprocess or not community_path.exists():
        nodecom = generate_community(
            data.edge_index,
            data.num_nodes,
            method=args.community_method,
            seed=args.seed,
            max_communities=args.max_communities,
        )
        with open(community_path, "w", encoding="utf8") as f:
            json.dump({"com": nodecom}, f)
    else:
        with open(community_path, encoding="utf8") as f:
            nodecom = json.load(f)["com"]

    if args.force_preprocess or not coef_path.exists():
        coef = generate_coef(data.x.detach().cpu().numpy())
        with open(coef_path, "wb") as f:
            pickle.dump(coef, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(coef_path, "rb") as f:
            coef = pickle.load(f)

    return nodecom, coef, community_path, coef_path


def get_scores(actual, score, k=None):
    actual = np.asarray(actual).astype(int)
    score = np.asarray(score)
    auc = metrics.roc_auc_score(actual, score)
    ap = metrics.average_precision_score(actual, score)
    if k is None:
        k = int(actual.sum())
    k = max(int(k), 1)
    order = np.argsort(-score)[:k]
    rec = float(actual[order].sum() / max(actual.sum(), 1))
    return auc, ap, rec


def get_one_sample(subgraph_size, nb_nodes, coef, subgraphs, strategy):
    all_samples = []
    for nd in range(nb_nodes):
        neighbors = list(set(subgraphs[nd]))
        if nd in neighbors:
            neighbors.remove(nd)
        if not neighbors:
            neighbors = [nd]

        if strategy == "random":
            chosen = random.sample(neighbors, 1)[0]
        elif strategy == "least-relevant":
            coefs = [coef[nd][neb] for neb in neighbors]
            chosen = neighbors[int(np.argmin(coefs))]
        else:
            coefs = [coef[nd][neb] for neb in neighbors]
            chosen = neighbors[int(np.argmax(coefs))]
        all_samples.append(subgraphs[nd].index(chosen))
    return np.asarray(all_samples)


def RemoveIsolated(data):
    num_nodes = data.num_nodes
    edge_index, edge_attr, mask = remove_isolated_nodes(data.edge_index, getattr(data, "edge_attr", None), num_nodes)
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    for key, item in list(data):
        if bool(re.search("edge", key)):
            continue
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            data[key] = item[mask]
    data.num_nodes = int(mask.sum())
    return data


def get_negs(idx, nodecom, communities, Com_size_ratio, num_negs, neg_sample_method):
    if len(idx) <= 1:
        return np.zeros((num_negs, len(idx)), dtype=np.int64)

    if neg_sample_method == "random" or len(communities) <= 1:
        neg_node = list(range(len(idx)))
        negs = []
        for _ in range(num_negs):
            each_negs = random.choices(neg_node, k=len(idx))
            for i, value in enumerate(each_negs):
                if value == i:
                    each_negs[i] = (value + 1) % len(idx)
            negs.append(each_negs)
        return np.asarray(negs, dtype=np.int64)

    mapper = {node: i for i, node in enumerate(idx)}
    com_to_pos = []
    for com in communities:
        nodes = [mapper[nd] for nd in idx if nodecom[nd] == com]
        com_to_pos.append(nodes)

    negs = []
    for nd in idx:
        nd_com = nodecom[nd]
        candidate_coms = [com for com in communities if com != nd_com and len(com_to_pos[com]) > 0]
        if not candidate_coms:
            candidates = [i for i in range(len(idx)) if i != mapper[nd]]
            negs.append(random.choices(candidates, k=num_negs))
            continue

        if neg_sample_method == "bias":
            weights = []
            for com in candidate_coms:
                original_candidates = [c for c in communities if c != nd_com]
                pos = original_candidates.index(com)
                weights.append(Com_size_ratio[nd_com][pos])
        else:  # even
            weights = [1.0] * len(candidate_coms)

        selected_coms = random.choices(candidate_coms, weights=weights, k=num_negs)
        selected = []
        for com in selected_coms:
            selected.append(random.choice(com_to_pos[com]))
        negs.append(selected)
    return np.asarray(negs, dtype=np.int64).T
