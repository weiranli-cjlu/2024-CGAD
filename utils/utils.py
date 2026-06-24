import json
import os
import pickle
import random
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from scipy.sparse import issparse
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes

MAT_ADJ_KEYS = ("Network", "network", "A", "adj", "Adj", "adjacency", "edges")
MAT_FEAT_KEYS = ("Attributes", "attributes", "X", "x", "features", "Features", "attrb", "attr")
MAT_LABEL_KEYS = ("Label", "label", "labels", "y", "Y", "gnd", "Class", "class")

_DATA_CACHE: Dict[Tuple[str, str, float], Data] = {}
_PREPROCESS_CACHE: Dict[Tuple, Tuple[List[int], Path]] = {}


def _expand_path(path_like) -> Path:
    return Path(os.path.expanduser(str(path_like))).resolve()


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
    arr = np.asarray(value)
    # Accept two-column edge list as well as dense adjacency.
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] != arr.shape[1]:
        n = int(arr.max()) + 1
        row, col = arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)
        return sp.csr_matrix((np.ones_like(row, dtype=np.float32), (row, col)), shape=(n, n))
    return sp.csr_matrix(arr)


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
        if value.shape[0] == num_nodes:
            value = np.argmax(value, axis=1)
        else:
            value = value.reshape(-1)
    value = value.astype(np.int64)
    if value.shape[0] != num_nodes:
        raise ValueError(f"Label length {value.shape[0]} does not match num_nodes {num_nodes}.")
    uniq = set(np.unique(value).tolist())
    if uniq.issubset({-1, 1}):
        value = (value > 0).astype(np.int64)
    return value


def load_mat_data(dataset: str, data_dir: str = "~/datasets/GAD/mat") -> Data:
    """Load common graph anomaly .mat datasets as a PyG Data object."""
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


def load_graph_data(dataset: str, data_dir: str = "~/datasets/GAD/mat") -> Data:
    """Load .mat first; fall back to the original .pt format for compatibility."""
    data_dir_path = _expand_path(data_dir)
    mat_path = data_dir_path / f"{dataset}.mat"
    if mat_path.exists():
        return load_mat_data(dataset, str(data_dir_path))

    candidates = [Path(f"{dataset}.pt"), data_dir_path / f"{dataset}.pt"]
    for pt_path in candidates:
        if pt_path.exists():
            data = torch.load(pt_path, map_location="cpu")
            if not hasattr(data, "num_nodes") or data.num_nodes is None:
                data.num_nodes = data.x.size(0)
            return data
    raise FileNotFoundError(
        f"Cannot find {dataset}.mat under {data_dir_path}, nor {dataset}.pt in current/data directory."
    )


def preprocess_features(features):
    """Row-normalize feature matrix and return dense float32 matrix."""
    rowsum = np.array(features.sum(1), dtype=np.float32).flatten()
    r_inv = np.zeros_like(rowsum, dtype=np.float32)
    nz = rowsum != 0
    r_inv[nz] = 1.0 / rowsum[nz]
    r_inv[np.isinf(r_inv)] = 0.0
    r_inv[np.isnan(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense(), dtype=np.float32)


def l2_normalize_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return features / norm


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


def build_neighbor_lists(edge_index: torch.Tensor, num_nodes: int, undirected: bool = True) -> List[np.ndarray]:
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
    return [np.asarray(sorted(item), dtype=np.int64) for item in neighbors]


def _rwr_one_node(
    seed: int,
    neighbors: List[np.ndarray],
    subgraph_size: int,
    rng: np.random.Generator,
    restart_prob: float = 0.9,
    max_steps: Optional[int] = None,
    max_retries: int = 10,
) -> List[int]:
    """Generate one CGAD subgraph with the target node at the last position."""
    reduced_size = subgraph_size - 1
    if reduced_size <= 0:
        return [seed]
    if max_steps is None:
        max_steps = max(subgraph_size * 5, 8)

    best = [seed]
    for _ in range(max_retries + 1):
        cur = seed
        trace = [seed]
        for _ in range(max_steps):
            neigh = neighbors[cur]
            if len(neigh) == 0 or rng.random() < restart_prob:
                cur = seed
            else:
                cur = int(neigh[rng.integers(0, len(neigh))])
            trace.append(cur)
        seen = set()
        unique = []
        for node in trace:
            if node not in seen:
                seen.add(node)
                unique.append(int(node))
        if len(unique) > len(best):
            best = unique
        if len(unique) >= reduced_size:
            best = unique
            break

    candidate = [v for v in best if v != seed]
    if not candidate:
        candidate = [seed]
    while len(candidate) < reduced_size:
        candidate.extend(candidate)
    candidate = candidate[:reduced_size]
    candidate.append(seed)
    return candidate


def generate_rwr_subgraph(
    edge_index_or_neighbors,
    subgraph_size: int,
    num_nodes: Optional[int] = None,
    seed: Optional[int] = None,
    restart_prob: float = 0.9,
    max_retries: int = 10,
):
    """Generate RWR subgraphs using PyG edge_index or cached neighbor lists.

    This replaces dgl.contrib.sampling.random_walk_with_restart and avoids DGL.
    """
    if isinstance(edge_index_or_neighbors, torch.Tensor):
        if num_nodes is None:
            num_nodes = int(edge_index_or_neighbors.max().item()) + 1
        neighbors = build_neighbor_lists(edge_index_or_neighbors, num_nodes)
    else:
        neighbors = edge_index_or_neighbors
        num_nodes = len(neighbors)
    rng = np.random.default_rng(seed)
    return [
        _rwr_one_node(i, neighbors, subgraph_size, rng, restart_prob=restart_prob, max_retries=max_retries)
        for i in range(num_nodes)
    ]


def choose_sample_indices(subgraphs: np.ndarray, feature_l2: np.ndarray, strategy: str) -> np.ndarray:
    """Choose positive neighbor position for every node.

    This computes cosine scores only inside each small RWR subgraph, instead of
    materialising an N x N cosine-similarity matrix.
    """
    subgraphs = np.asarray(subgraphs, dtype=np.int64)
    nb_nodes = subgraphs.shape[0]
    all_samples = np.empty(nb_nodes, dtype=np.int64)
    if strategy == "random":
        for nd in range(nb_nodes):
            candidates = [pos for pos, node in enumerate(subgraphs[nd]) if int(node) != nd]
            if not candidates:
                candidates = [subgraphs.shape[1] - 1]
            all_samples[nd] = random.choice(candidates)
        return all_samples

    for nd in range(nb_nodes):
        row = subgraphs[nd]
        positions = [pos for pos, node in enumerate(row) if int(node) != nd]
        if not positions:
            all_samples[nd] = subgraphs.shape[1] - 1
            continue
        cand_nodes = row[positions]
        sims = feature_l2[cand_nodes] @ feature_l2[nd]
        best_pos = int(np.argmin(sims) if strategy == "least-relevant" else np.argmax(sims))
        all_samples[nd] = positions[best_pos]
    return all_samples


def prepare_subgraph_bank(
    neighbor_lists: List[np.ndarray],
    subgraph_size: int,
    feature_l2: np.ndarray,
    strategy: str,
    rounds: int,
    seed: int,
    cache_path: Optional[Path] = None,
    force: bool = False,
    restart_prob: float = 0.9,
    max_retries: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate/load a bank of RWR subgraphs and corresponding sample positions.

    Returns:
        subgraph_bank: [R, N, S] int64
        sample_bank:   [R, N] int64
    """
    rounds = max(int(rounds), 1)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            pack = np.load(cache_path)
            subgraph_bank = pack["subgraph_bank"].astype(np.int64, copy=False)
            sample_bank = pack["sample_bank"].astype(np.int64, copy=False)
            if subgraph_bank.shape[0] >= rounds and subgraph_bank.shape[2] == subgraph_size:
                return subgraph_bank[:rounds], sample_bank[:rounds]

    subgraphs_all = []
    samples_all = []
    for r in range(rounds):
        subgraphs = np.asarray(
            generate_rwr_subgraph(
                neighbor_lists,
                subgraph_size,
                seed=seed + r,
                restart_prob=restart_prob,
                max_retries=max_retries,
            ),
            dtype=np.int64,
        )
        samples = choose_sample_indices(subgraphs, feature_l2, strategy)
        subgraphs_all.append(subgraphs.astype(np.int32))
        samples_all.append(samples.astype(np.int16 if subgraph_size < 32767 else np.int32))

    subgraph_bank = np.stack(subgraphs_all, axis=0)
    sample_bank = np.stack(samples_all, axis=0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, subgraph_bank=subgraph_bank, sample_bank=sample_bank)
    return subgraph_bank.astype(np.int64), sample_bank.astype(np.int64)


def resolve_preprocess_paths(args) -> Tuple[Path, Path]:
    if getattr(args, "cache_dir", None):
        cache_dir = _expand_path(args.cache_dir)
    else:
        train_dir = _expand_path(getattr(args, "train_dir", "./runs"))
        cache_dir = train_dir / "cgad_preprocess"
    community_path = cache_dir / f"{args.dataset}.json"
    return cache_dir, community_path


def _largest_remainder_partition(nodecom: List[int], max_communities: int) -> List[int]:
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
            merged.append(idx % max_communities)
    return merged


def generate_community(
    edge_index: torch.Tensor,
    num_nodes: int,
    method: str = "louvain",
    seed: int = 1,
    max_communities: int = 0,
) -> List[int]:
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
    """Load cached community labels or generate them once.

    Unlike the original implementation, this function no longer computes/stores
    a dense N x N cosine matrix by default; sample selection is computed locally
    inside each small subgraph.
    """
    cache_dir, community_path = resolve_preprocess_paths(args)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Backward compatibility: prefer existing ./dataset.json when cache does not exist.
    local_json = Path(f"{args.dataset}.json")
    if not community_path.exists() and local_json.exists() and not getattr(args, "force_preprocess", False):
        community_path = local_json

    community_mtime = community_path.stat().st_mtime if community_path.exists() else -1.0
    cache_key = (
        args.dataset,
        str(community_path),
        getattr(args, "community_method", "louvain"),
        int(getattr(args, "max_communities", 0)),
        int(getattr(args, "seed", 1)),
        community_mtime,
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
            text = f.read().strip()
            nodecom = json.loads(text)["com"]

    # Ensure community ids are contiguous, because the sampler uses them as array ids.
    uniq = sorted(set(nodecom))
    remap = {cid: i for i, cid in enumerate(uniq)}
    nodecom = [remap[cid] for cid in nodecom]
    result = (nodecom, community_path)
    _PREPROCESS_CACHE[cache_key] = result
    return result


def get_scores(actual, score, k=None):
    """Return ROC-AUC, PR-AUC and Recall@K.

    AUPRC is computed by sklearn.precision_recall_curve + sklearn.auc.
    """
    actual = np.asarray(actual).astype(int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    if actual.shape[0] != score.shape[0]:
        raise ValueError(f"actual length {actual.shape[0]} != score length {score.shape[0]}")
    positives = int(actual.sum())
    has_two_classes = len(np.unique(actual)) == 2
    score = np.nan_to_num(score)
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


def get_one_sample(subgraph_size, nb_nodes, coef, subgraphs, strategy):
    """Backward-compatible wrapper.

    Prefer choose_sample_indices(subgraphs, feature_l2, strategy) in new code.
    """
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


def get_negs_fast(
    idx: Sequence[int],
    nodecom: Sequence[int],
    num_negs: int,
    neg_sample_method: str = "bias",
) -> np.ndarray:
    """Batch-aware negative sampler.

    Returns negative sample *positions inside the current batch*, shape [K, B].
    It avoids repeatedly building Python dicts/lists for every community in the
    whole graph, and handles empty communities in small batches.
    """
    idx = np.asarray(idx, dtype=np.int64)
    batch_size = idx.shape[0]
    if batch_size <= 1:
        return np.zeros((num_negs, batch_size), dtype=np.int64)

    if neg_sample_method == "random":
        negs = np.random.randint(0, batch_size, size=(num_negs, batch_size), dtype=np.int64)
        cols = np.arange(batch_size)
        same = negs == cols[None, :]
        negs[same] = (negs[same] + 1) % batch_size
        return negs

    nodecom_np = np.asarray(nodecom, dtype=np.int64)
    batch_com = nodecom_np[idx]
    max_com = int(batch_com.max()) if batch_com.size else 0
    com_to_pos = [np.flatnonzero(batch_com == c) for c in range(max_com + 1)]
    non_empty = np.asarray([c for c, pos in enumerate(com_to_pos) if len(pos) > 0], dtype=np.int64)
    negs = np.empty((num_negs, batch_size), dtype=np.int64)

    for col in range(batch_size):
        cur_com = int(batch_com[col])
        cand_coms = non_empty[non_empty != cur_com]
        if cand_coms.size == 0:
            candidates = np.delete(np.arange(batch_size, dtype=np.int64), col)
            negs[:, col] = np.random.choice(candidates, size=num_negs, replace=True)
            continue
        if neg_sample_method == "bias":
            weights = np.asarray([len(com_to_pos[int(c)]) for c in cand_coms], dtype=np.float64)
            weights = weights / weights.sum()
        else:
            weights = None
        selected_coms = np.random.choice(cand_coms, size=num_negs, replace=True, p=weights)
        for k, com in enumerate(selected_coms):
            choices = com_to_pos[int(com)]
            negs[k, col] = int(choices[np.random.randint(0, len(choices))])
    return negs


def get_negs(idx, nodecom, communities, Com_size_ratio, num_negs, neg_sample_method):
    """Backward-compatible name used by older run.py."""
    return get_negs_fast(idx, nodecom, num_negs, neg_sample_method)
