import json
import os
import pickle
import random
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from scipy.sparse import issparse
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes

MAT_ADJ_KEYS = ("Network", "network", "A", "adj", "Adj", "adjacency", "edges")
MAT_FEAT_KEYS = ("Attributes", "attributes", "X", "x", "features", "Features", "attrb", "attr")
MAT_LABEL_KEYS = ("Label", "label", "labels", "y", "Y", "gnd", "Class", "class")

# In-process caches reduce repeated disk IO during grid search / repeated train_ours calls.
_DATA_CACHE: Dict[Tuple[str, str, float], Data] = {}
_PREPROCESS_CACHE: Dict[Tuple[str, str, str, int, int, float, float], Tuple[List[int], np.ndarray, Path, Path]] = {}


def _expand_path(path_like) -> Path:
    return Path(os.path.expanduser(str(path_like))).resolve()


def resolve_preprocess_paths(args) -> Tuple[Path, Path, Path]:
    """Resolve preprocess cache location.

    If --cache_dir is not explicitly set, generated files are placed under
    <train_dir>/cgad_preprocess instead of the dataset directory:
        <train_dir>/cgad_preprocess/<dataset>.json
        <train_dir>/cgad_preprocess/<dataset>.pkl
    """
    if getattr(args, "cache_dir", None):
        cache_dir = _expand_path(args.cache_dir)
    else:
        train_dir = _expand_path(getattr(args, "train_dir", "./runs"))
        cache_dir = train_dir / "cgad_preprocess"
    community_path = cache_dir / f"{args.dataset}.json"
    coef_path = cache_dir / f"{args.dataset}.pkl"
    return cache_dir, community_path, coef_path


def _atomic_json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf8", dir=path.parent, delete=False) as f:
        json.dump(obj, f, ensure_ascii=False)
        tmp_name = f.name
    os.replace(tmp_name, path)


def _atomic_pickle_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_name = f.name
    os.replace(tmp_name, path)


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
    data_dir = _expand_path(data_dir)
    mat_path = data_dir / f"{dataset}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {mat_path}")

    mtime = mat_path.stat().st_mtime
    cache_key = (dataset, str(data_dir), mtime)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

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
            raise ValueError(f"Feature shape {x.shape} does not match adjacency shape {adj.shape}.")

    row, col = adj.nonzero()
    edge_index = torch.as_tensor(np.vstack([row, col]), dtype=torch.long)
    data = Data(
        x=torch.as_tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.as_tensor(y, dtype=torch.long),
    )
    data.num_nodes = int(adj.shape[0])
    _DATA_CACHE[cache_key] = data
    return data


def preprocess_features(features):
    """Row-normalize feature matrix and return dense float32 matrix."""
    rowsum = np.array(features.sum(1), dtype=np.float32)
    r_inv = np.zeros_like(rowsum, dtype=np.float32).flatten()
    nz = rowsum.flatten() != 0
    r_inv[nz] = 1.0 / rowsum.flatten()[nz]
    r_inv[np.isinf(r_inv)] = 0.0
    r_inv[np.isnan(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense(), dtype=np.float32)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj, dtype=np.float32)
    rowsum = np.array(adj.sum(1), dtype=np.float32).flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    nz = rowsum != 0
    d_inv_sqrt[nz] = np.power(rowsum[nz], -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def build_neighbor_lists(edge_index: torch.Tensor, num_nodes: int, undirected: bool = True) -> List[List[int]]:
    """Build CPU adjacency lists from PyG edge_index once and reuse them."""
    edge_np = edge_index.detach().cpu().long().numpy()
    neighbors = [set() for _ in range(num_nodes)]
    for src, dst in zip(edge_np[0], edge_np[1]):
        src = int(src)
        dst = int(dst)
        if src == dst:
            continue
        neighbors[src].add(dst)
        if undirected:
            neighbors[dst].add(src)
    return [sorted(item) for item in neighbors]


def _rwr_unique_trace(seed: int, neighbors: List[List[int]], restart_prob: float, max_steps: int) -> List[int]:
    cur = seed
    trace = [seed]
    for _ in range(max_steps):
        if random.random() < restart_prob or not neighbors[cur]:
            cur = seed
        else:
            cur = random.choice(neighbors[cur])
        trace.append(cur)

    # Preserve order while removing duplicates.
    seen = set()
    unique = []
    for node in trace:
        if node not in seen:
            seen.add(node)
            unique.append(node)
    return unique


def generate_rwr_subgraph(edge_index_or_neighbors, subgraph_size: int, num_nodes: int = None):
    """Generate RWR subgraphs using PyG edge_index / cached neighbor lists instead of DGL.

    The returned subgraph keeps the original CGAD convention: the target node is
    placed at the final position of each subgraph.
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
    features = np.asarray(features, dtype=np.float32)
    coef = cosine_similarity(features)
    return np.asarray(coef, dtype=np.float32)


def _largest_remainder_partition(nodecom: List[int], max_communities: int) -> List[int]:
    """Merge surplus communities into [0, max_communities - 1] ids."""
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
            # Assign remaining communities deterministically.
            merged.append(idx % max_communities)
    return merged


def generate_community(
    edge_index: torch.Tensor,
    num_nodes: int,
    method: str = "louvain",
    seed: int = 1,
    max_communities: int = 0,
) -> List[int]:
    """Generate community id for every node, replacing the original external json."""
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index.detach().cpu().long().t().tolist())

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
    """Load cached community/coef if available, otherwise generate them automatically.

    Cache is kept in memory after the first load/generation, so grid search does
    not repeatedly read the same .json/.pkl files from disk.
    """
    cache_dir, community_path, coef_path = resolve_preprocess_paths(args)
    cache_dir.mkdir(parents=True, exist_ok=True)

    community_mtime = community_path.stat().st_mtime if community_path.exists() else -1.0
    coef_mtime = coef_path.stat().st_mtime if coef_path.exists() else -1.0
    cache_key = (
        args.dataset,
        str(cache_dir),
        getattr(args, "community_method", "louvain"),
        int(getattr(args, "max_communities", 0)),
        int(getattr(args, "seed", 1)),
        community_mtime,
        coef_mtime,
    )
    if not getattr(args, "force_preprocess", False) and cache_key in _PREPROCESS_CACHE:
        return _PREPROCESS_CACHE[cache_key]

    if getattr(args, "force_preprocess", False) or not community_path.exists():
        nodecom = generate_community(
            data.edge_index,
            data.num_nodes,
            method=getattr(args, "community_method", "louvain"),
            seed=getattr(args, "seed", 1),
            max_communities=getattr(args, "max_communities", 0),
        )
        _atomic_json_dump({"com": nodecom}, community_path)
    else:
        with open(community_path, encoding="utf8") as f:
            nodecom = json.load(f)["com"]

    if getattr(args, "force_preprocess", False) or not coef_path.exists():
        coef = generate_coef(data.x.detach().cpu().numpy())
        _atomic_pickle_dump(coef, coef_path)
    else:
        with open(coef_path, "rb") as f:
            coef = pickle.load(f)
    coef = np.asarray(coef, dtype=np.float32)

    # Refresh mtime after possible generation.
    community_mtime = community_path.stat().st_mtime if community_path.exists() else -1.0
    coef_mtime = coef_path.stat().st_mtime if coef_path.exists() else -1.0
    cache_key = (
        args.dataset,
        str(cache_dir),
        getattr(args, "community_method", "louvain"),
        int(getattr(args, "max_communities", 0)),
        int(getattr(args, "seed", 1)),
        community_mtime,
        coef_mtime,
    )
    _PREPROCESS_CACHE[cache_key] = (nodecom, coef, community_path, coef_path)
    return nodecom, coef, community_path, coef_path


def get_scores(actual, score, k=None):
    """Return ROC-AUC, PR-AUC and Recall@K.

    AUPRC is intentionally computed by scikit-learn's
    precision_recall_curve + auc, rather than average_precision_score.
    """
    actual = np.asarray(actual).astype(int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    if actual.shape[0] != score.shape[0]:
        raise ValueError(f"actual length {actual.shape[0]} != score length {score.shape[0]}")

    positives = int(actual.sum())
    has_two_classes = len(np.unique(actual)) == 2
    auc_value = float(roc_auc_score(actual, score)) if has_two_classes else float("nan")

    if positives > 0:
        precision, recall, _ = precision_recall_curve(actual, score)
        auprc_value = float(sk_auc(recall, precision))
    else:
        auprc_value = float("nan")

    if k is None:
        k = positives
    k = max(int(k), 1)
    if positives > 0:
        order = np.argsort(-score)[:k]
        rec = float(actual[order].sum() / positives)
    else:
        rec = float("nan")
    return auc_value, auprc_value, rec


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
    return np.asarray(all_samples, dtype=np.int64)


def RemoveIsolated(data):
    num_nodes = data.num_nodes
    edge_index, edge_attr, mask = remove_isolated_nodes(
        data.edge_index,
        getattr(data, "edge_attr", None),
        num_nodes,
    )
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
            original_candidates = [c for c in communities if c != nd_com]
            for com in candidate_coms:
                pos = original_candidates.index(com)
                weights.append(Com_size_ratio[nd_com][pos])
        else:  # even
            weights = [1.0] * len(candidate_coms)

        selected_coms = random.choices(candidate_coms, weights=weights, k=num_negs)
        selected = [random.choice(com_to_pos[com]) for com in selected_coms]
        negs.append(selected)
    return np.asarray(negs, dtype=np.int64).T
