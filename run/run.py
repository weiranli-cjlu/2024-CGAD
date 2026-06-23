import math
import os
import random
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_scipy_sparse_matrix

from model.model import Model
from utils.utils import (
    RemoveIsolated,
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


def train_ours(args):
    print(f"Dataset: {args.dataset}", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = load_mat_data(args.dataset, args.data_dir)
    if data.has_isolated_nodes():
        data = RemoveIsolated(data)

    nodecom, coef, community_path, coef_path = load_or_generate_preprocess(data, args)
    print(f"Community cache: {community_path}", flush=True)
    print(f"Coef cache: {coef_path}", flush=True)

    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    features_sp = sp.lil_matrix(data.x.detach().cpu().numpy())
    y = data.y.bool()
    ano_label = np.asarray(y.detach().cpu().numpy()).astype(int)

    features = preprocess_features(features_sp)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    adj_norm = normalize_adj(adj)
    adj_dense = (adj_norm + sp.eye(adj_norm.shape[0])).todense()
    features_tensor = torch.FloatTensor(np.asarray(features)[np.newaxis]).to(device)
    adj_tensor = torch.FloatTensor(np.asarray(adj_dense)[np.newaxis]).to(device)

    neighbor_lists = None
    # build once, reuse for every epoch/test round
    from utils.utils import build_neighbor_lists
    neighbor_lists = build_neighbor_lists(data.edge_index, data.num_nodes)

    communities, com_size_ratio, comnode = _build_community_helpers(nodecom, nb_nodes)
    if len(communities) < 2 and args.neg_sample_method != "random":
        print("Only one community generated; fallback neg_sample_method to random.", flush=True)
        args.neg_sample_method = "random"

    final_scores = []
    seeds = [args.seed + i for i in range(args.runs)]
    batch_size = args.batch_size
    subgraph_size = args.subgraph_size
    batch_num = max(math.ceil(nb_nodes / batch_size), 1)

    for run in range(args.runs):
        seed = seeds[run]
        set_seed(seed)
        print(f"---Train run {run + 1}/{args.runs}, seed={seed}---", flush=True)

        model = Model(ft_size, args.embedding_dim, "prelu", args.readout, args.T).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        cnt_wait = 0
        best = float("inf")
        best_t = 0

        for epoch in range(args.num_epoch):
            model.train()
            total_loss = 0.0
            subgraphs = generate_rwr_subgraph(neighbor_lists, subgraph_size)
            all_samples = get_one_sample(subgraph_size, nb_nodes, coef, subgraphs, args.strategy)

            com_batch_sizes = [max(len(item) // batch_num, 1) for item in comnode]
            for item in comnode:
                random.shuffle(item)

            last_loss = 0.0
            last_batch_size = 0
            for batch_idx in range(batch_num):
                optimiser.zero_grad()
                is_final_batch = batch_idx == batch_num - 1
                if not is_final_batch:
                    idx_nested = [
                        comnode[j][batch_idx * com_batch_sizes[j] : (batch_idx + 1) * com_batch_sizes[j]]
                        for j in range(len(communities))
                    ]
                else:
                    idx_nested = [
                        comnode[j][batch_idx * com_batch_sizes[j] :]
                        for j in range(len(communities))
                    ]
                idx = sum(idx_nested, [])
                if not idx:
                    continue
                random.shuffle(idx)
                cur_batch_size = len(idx)

                multi_neg_node = get_negs(
                    idx, nodecom, communities, com_size_ratio, args.num_negs, args.neg_sample_method
                )

                ba = []
                bf = []
                for i in idx:
                    cur_adj = adj_tensor[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features_tensor[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)
                ba = torch.cat(ba)
                bf = torch.cat(bf)

                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size), device=device)
                bf_mask = torch.cat((bf[:, :-1, :], added_feat_zero_row), dim=1)
                mask_adj = torch.eye(subgraph_size, device=device)
                ba_mask = mask_adj.expand(cur_batch_size, subgraph_size, subgraph_size)

                sample_node = all_samples[idx]
                node_logits, sub_logits, _ = model(
                    bf_mask, ba, bf, ba_mask, multi_neg_node, sample_node
                )
                node_loss = Rnce_loss(node_logits, lam=args.lam, q=args.q)
                sub_loss = Rnce_loss(sub_logits, lam=args.lam, q=args.q)
                loss = args.alpha * node_loss + (1 - args.alpha) * sub_loss
                loss.backward()
                optimiser.step()

                last_loss = float(loss.detach().cpu().item())
                last_batch_size = cur_batch_size
                if not is_final_batch:
                    total_loss += last_loss * cur_batch_size

            mean_loss = (total_loss + last_loss * last_batch_size) / max(nb_nodes, 1)
            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
            else:
                cnt_wait += 1
            print(f"Epoch:{epoch} Loss:{mean_loss:.8f}", flush=True)
            if cnt_wait == args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_t}", flush=True)
                break

        print(f"---Test run {run + 1}/{args.runs}---", flush=True)
        model.eval()
        multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes), dtype=np.float32)

        for round_id in range(args.auc_test_rounds):
            subgraphs = generate_rwr_subgraph(neighbor_lists, subgraph_size)
            all_samples = get_one_sample(subgraph_size, nb_nodes, coef, subgraphs, args.strategy)
            com_batch_sizes = [max(len(item) // batch_num, 1) for item in comnode]
            for item in comnode:
                random.shuffle(item)

            for batch_idx in range(batch_num):
                is_final_batch = batch_idx == batch_num - 1
                if not is_final_batch:
                    idx_nested = [
                        comnode[j][batch_idx * com_batch_sizes[j] : (batch_idx + 1) * com_batch_sizes[j]]
                        for j in range(len(communities))
                    ]
                else:
                    idx_nested = [
                        comnode[j][batch_idx * com_batch_sizes[j] :]
                        for j in range(len(communities))
                    ]
                idx = sum(idx_nested, [])
                if not idx:
                    continue
                cur_batch_size = len(idx)
                multi_neg_node = get_negs(
                    idx, nodecom, communities, com_size_ratio, args.num_negs, args.neg_sample_method
                )

                ba = []
                bf = []
                for i in idx:
                    cur_adj = adj_tensor[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features_tensor[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)
                ba = torch.cat(ba)
                bf = torch.cat(bf)

                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size), device=device)
                bf_mask = torch.cat((bf[:, :-1, :], added_feat_zero_row), dim=1)
                mask_adj = torch.eye(subgraph_size, device=device)
                ba_mask = mask_adj.expand(cur_batch_size, subgraph_size, subgraph_size)

                with torch.no_grad():
                    sample_node = all_samples[idx]
                    node_logits, sub_logits, _ = model(
                        bf_mask, ba, bf, ba_mask, multi_neg_node, sample_node
                    )
                    node_score = (
                        node_logits[:, 1:].detach().cpu().numpy()
                        - node_logits[:, 0].unsqueeze(1).detach().cpu().numpy()
                    )
                    node_score = np.mean(node_score, axis=1) + np.std(node_score, axis=1)
                    sub_score = (
                        sub_logits[:, 1:].detach().cpu().numpy()
                        - sub_logits[:, 0].unsqueeze(1).detach().cpu().numpy()
                    )
                    sub_score = np.mean(sub_score, axis=1) + np.std(sub_score, axis=1)
                    ano_score = args.alpha * node_score + (1 - args.alpha) * sub_score
                    multi_round_ano_score[round_id, idx] = ano_score

        ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
        ano_score_final = ano_score_final - np.min(ano_score_final)
        final_scores.append(ano_score_final)

    ano_score_final = np.mean(np.vstack(final_scores), axis=0)
    return ano_label, ano_score_final
