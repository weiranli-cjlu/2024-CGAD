import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_scipy_sparse_matrix

try:
    from model.model import Model
except Exception:  # compatibility with the original flat import style
    from model import Model

try:
    from utils.utils import (
        RemoveIsolated,
        build_neighbor_lists,
        get_negs_fast,
        l2_normalize_features,
        load_graph_data,
        load_or_generate_preprocess,
        normalize_adj,
        prepare_subgraph_bank,
        preprocess_features,
    )
except Exception:
    from utils import (
        RemoveIsolated,
        build_neighbor_lists,
        get_negs_fast,
        l2_normalize_features,
        load_graph_data,
        load_or_generate_preprocess,
        normalize_adj,
        prepare_subgraph_bank,
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


def _build_community_helpers(nodecom: Sequence[int], nb_nodes: int):
    nodecom = [int(x) for x in nodecom]
    com_size = dict(Counter(nodecom))
    communities = sorted(com_size)
    comnode = []
    for item in communities:
        each_com_node = [nd for nd in range(len(nodecom)) if nodecom[nd] == item]
        comnode.append(each_com_node)
    return communities, comnode


def _iter_community_batches(comnode: List[List[int]], batch_num: int):
    """Yield balanced node batches.

    This keeps the original community-balanced batching idea but avoids repeated
    slice boilerplate and guards against zero-sized community slices.
    """
    comnode_work = [list(item) for item in comnode]
    com_batch_sizes = [max(len(item) // batch_num, 1) for item in comnode_work]
    for item in comnode_work:
        random.shuffle(item)
    for batch_idx in range(batch_num):
        is_final_batch = batch_idx == batch_num - 1
        if not is_final_batch:
            idx_nested = [
                comnode_work[j][batch_idx * com_batch_sizes[j] : (batch_idx + 1) * com_batch_sizes[j]]
                for j in range(len(comnode_work))
            ]
        else:
            idx_nested = [comnode_work[j][batch_idx * com_batch_sizes[j] :] for j in range(len(comnode_work))]
        idx = sum(idx_nested, [])
        if idx:
            yield batch_idx, is_final_batch, idx


def _build_batch_tensors(
    adj_base: torch.Tensor,
    features_base: torch.Tensor,
    subgraph_tensor: torch.Tensor,
    idx: Sequence[int],
    mask_adj_base: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized subgraph tensor construction.

    Replaces the original per-node loop:
        for i in idx:
            cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
            cur_feat = features[:, subgraphs[i], :]
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


def _resolve_cache_dir(args) -> Path:
    if getattr(args, "cache_dir", None):
        return Path(os.path.expanduser(str(args.cache_dir))).resolve()
    train_dir = Path(os.path.expanduser(str(getattr(args, "train_dir", "./runs")))).resolve()
    return train_dir / "cgad_preprocess"


def _select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA is not available; fallback to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested if torch.cuda.is_available() or not requested.startswith("cuda") else "cpu")


def _prepare_static_tensors(data, args, device):
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    features_sp = sp.lil_matrix(data.x.detach().cpu().numpy())
    features_np = preprocess_features(features_sp)
    nb_nodes, ft_size = features_np.shape

    adj_norm = normalize_adj(adj)
    # This keeps the original dense subgraph adjacency semantics. The expensive
    # dense matrix is built once, then indexed on device per batch.
    adj_dense = (adj_norm + sp.eye(adj_norm.shape[0], dtype=np.float32)).toarray().astype(np.float32)

    features_base = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    adj_base = torch.as_tensor(adj_dense, dtype=torch.float32, device=device)
    mask_adj_base = torch.eye(args.subgraph_size, device=device)
    return adj, features_np, features_base, adj_base, mask_adj_base, nb_nodes, ft_size


def train_ours(args):
    """Train CGAD and return anomaly labels/scores.

    CPU optimizations implemented here:
    1. DGL RWR is replaced by cached PyG/NumPy neighbor-list RWR.
    2. RWR subgraphs are generated as a small bank once and reused.
    3. Sample selection computes local cosine only inside each subgraph, avoiding
       an N x N cosine matrix.
    4. Batch subgraph tensors are built by advanced indexing instead of Python
       loops + torch.cat.
    5. Negative sampling is batch-aware and vectorized for the random path.
    """
    torch_threads = int(getattr(args, "torch_threads", 1))
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(max(1, min(torch_threads, 4)))
        except RuntimeError:
            pass

    device = _select_device(getattr(args, "device", "cuda:0"))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = load_graph_data(args.dataset, getattr(args, "data_dir", "~/datasets/GAD/mat"))
    if data.has_isolated_nodes():
        data = RemoveIsolated(data)

    y = data.y.bool()
    ano_label = np.asarray(y.detach().cpu().numpy()).astype(int)

    nodecom, community_path = load_or_generate_preprocess(data, args)
    adj, features_np, features_base, adj_base, mask_adj_base, nb_nodes, ft_size = _prepare_static_tensors(
        data, args, device
    )
    feature_l2 = l2_normalize_features(features_np)

    # Build once and cache a small bank of subgraphs. This removes the largest
    # source of CPU pressure in the original code.
    neighbor_lists = build_neighbor_lists(data.edge_index, data.num_nodes)
    cache_dir = _resolve_cache_dir(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bank_rounds = int(getattr(args, "subgraph_cache_rounds", 16))
    required_rounds = max(bank_rounds, min(int(getattr(args, "auc_test_rounds", 100)), bank_rounds))
    subgraph_cache_path = None
    if not getattr(args, "disable_subgraph_cache", False):
        subgraph_cache_path = cache_dir / (
            f"{args.dataset}_rwr_s{args.subgraph_size}_r{required_rounds}_"
            f"seed{args.seed}_rp{getattr(args, 'rwr_restart_prob', 0.9)}_"
            f"{getattr(args, 'strategy', 'most-relevant')}.npz"
        )
    subgraph_bank, sample_bank = prepare_subgraph_bank(
        neighbor_lists=neighbor_lists,
        subgraph_size=args.subgraph_size,
        feature_l2=feature_l2,
        strategy=getattr(args, "strategy", "most-relevant"),
        rounds=required_rounds,
        seed=int(getattr(args, "seed", 1)),
        cache_path=subgraph_cache_path,
        force=bool(getattr(args, "force_subgraph_cache", False) or getattr(args, "force_preprocess", False)),
        restart_prob=float(getattr(args, "rwr_restart_prob", 0.9)),
        max_retries=int(getattr(args, "rwr_max_retries", 10)),
    )
    subgraph_bank_t = [torch.as_tensor(x, dtype=torch.long, device=device) for x in subgraph_bank]

    communities, comnode = _build_community_helpers(nodecom, nb_nodes)
    if len(communities) < 2 and getattr(args, "neg_sample_method", "bias") != "random":
        args.neg_sample_method = "random"

    final_scores = []
    run_infos = []
    seeds = [int(getattr(args, "seed", 1)) + i for i in range(int(getattr(args, "runs", 1)))]
    batch_size = int(getattr(args, "batch_size", 256))
    batch_num = max(math.ceil(nb_nodes / batch_size), 1)

    for run in range(int(getattr(args, "runs", 1))):
        seed = seeds[run]
        set_seed(seed)
        model = Model(ft_size, args.embedding_dim, "prelu", args.readout, args.T).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        cnt_wait = 0
        best = float("inf")
        best_t = 0
        epochs_trained = 0

        for epoch in range(int(getattr(args, "num_epoch", 100))):
            model.train()
            total_loss = 0.0
            total_seen = 0
            epochs_trained = epoch + 1
            bank_id = epoch % len(subgraph_bank_t)
            subgraph_tensor = subgraph_bank_t[bank_id]
            all_samples = sample_bank[bank_id]

            for _, _, idx in _iter_community_batches(comnode, batch_num):
                random.shuffle(idx)
                cur_batch_size = len(idx)
                optimiser.zero_grad(set_to_none=True)

                multi_neg_node = get_negs_fast(
                    idx,
                    nodecom,
                    int(getattr(args, "num_negs", 3)),
                    getattr(args, "neg_sample_method", "bias"),
                )
                ba, bf, bf_mask, ba_mask = _build_batch_tensors(
                    adj_base, features_base, subgraph_tensor, idx, mask_adj_base
                )
                sample_node = all_samples[np.asarray(idx, dtype=np.int64)]
                node_logits, sub_logits, _ = model(bf_mask, ba, bf, ba_mask, multi_neg_node, sample_node)

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
            if cnt_wait == int(getattr(args, "patience", 20)):
                break

        model.eval()
        test_rounds = int(getattr(args, "auc_test_rounds", 100))
        if getattr(args, "cap_test_rounds_to_cache", False):
            test_rounds = min(test_rounds, len(subgraph_bank_t))
        multi_round_ano_score = np.zeros((test_rounds, nb_nodes), dtype=np.float32)

        with torch.no_grad():
            for round_id in range(test_rounds):
                bank_id = round_id % len(subgraph_bank_t)
                subgraph_tensor = subgraph_bank_t[bank_id]
                all_samples = sample_bank[bank_id]

                for _, _, idx in _iter_community_batches(comnode, batch_num):
                    multi_neg_node = get_negs_fast(
                        idx,
                        nodecom,
                        int(getattr(args, "num_negs", 3)),
                        getattr(args, "neg_sample_method", "bias"),
                    )
                    ba, bf, bf_mask, ba_mask = _build_batch_tensors(
                        adj_base, features_base, subgraph_tensor, idx, mask_adj_base
                    )
                    sample_node = all_samples[np.asarray(idx, dtype=np.int64)]
                    node_logits, sub_logits, _ = model(bf_mask, ba, bf, ba_mask, multi_neg_node, sample_node)

                    node_logits_np = node_logits.detach().cpu().numpy()
                    sub_logits_np = sub_logits.detach().cpu().numpy()
                    node_score = node_logits_np[:, 1:] - node_logits_np[:, [0]]
                    node_score = np.mean(node_score, axis=1) + np.std(node_score, axis=1)
                    sub_score = sub_logits_np[:, 1:] - sub_logits_np[:, [0]]
                    sub_score = np.mean(sub_score, axis=1) + np.std(sub_score, axis=1)
                    ano_score = args.alpha * node_score + (1 - args.alpha) * sub_score
                    multi_round_ano_score[round_id, idx] = ano_score.astype(np.float32)

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
                "community_path": str(community_path),
                "subgraph_cache_path": str(subgraph_cache_path) if subgraph_cache_path else None,
            }
        )

    ano_score_mean = np.mean(np.vstack(final_scores), axis=0)
    if getattr(args, "return_run_scores", False):
        return ano_label, ano_score_mean, final_scores, run_infos
    return ano_label, ano_score_mean
