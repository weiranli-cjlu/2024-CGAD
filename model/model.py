"""CGAD model modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """GCN layer used by CGAD/CoLA-style subgraph encoders."""

    def __init__(self, in_ft: int, out_ft: int, act: str | nn.Module, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == "prelu" else act
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ft))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, seq: torch.Tensor, adj: torch.Tensor, sparse: bool = False) -> torch.Tensor:
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)


class AvgReadout(nn.Module):
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.mean(seq, dim=1)


class MaxReadout(nn.Module):
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.max(seq, dim=1).values


class MinReadout(nn.Module):
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.min(seq, dim=1).values


class WSReadout(nn.Module):
    """Simple weighted-sum readout fallback for compatibility."""

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(torch.norm(seq, p=2, dim=-1), dim=1).unsqueeze(-1)
        return torch.sum(seq * weights, dim=1)


class Model(nn.Module):
    def __init__(self, n_in: int, n_h: int, activation: str, readout: str, T: float):
        super().__init__()
        self.read_mode = readout
        self.T = T
        self.gcn_node = GCN(n_in, n_h, activation)
        self.gcn_context = GCN(n_in, n_h, activation)

        if readout == "max":
            self.read = MaxReadout()
        elif readout == "min":
            self.read = MinReadout()
        elif readout == "avg":
            self.read = AvgReadout()
        elif readout == "weighted_sum":
            self.read = WSReadout()
        else:
            raise ValueError(f"Unknown readout mode: {readout}")

    def forward(
        self,
        bf_mask: torch.Tensor,
        ba: torch.Tensor,
        bf: torch.Tensor,
        ba_mask: torch.Tensor,
        multi_neg_node,
        sample_node,
        sparse: bool = False,
        msk=None,
        samp_bias1=None,
        samp_bias2=None,
    ):
        device = bf.device
        sample_node = torch.as_tensor(sample_node, dtype=torch.long, device=device)
        multi_neg_node = torch.as_tensor(multi_neg_node, dtype=torch.long, device=device)

        h_1 = self.gcn_node(bf, ba_mask, sparse)
        h_2 = self.gcn_context(bf_mask, ba, sparse)

        batch_ids = torch.arange(sample_node.numel(), device=device)
        target_node = F.normalize(h_1[:, -1, :], dim=1)

        pos_individual_neighbor = h_1[batch_ids, sample_node, :]
        pos_individual_neighbor = F.normalize(pos_individual_neighbor, dim=1)

        # Shape: [K, B, H]
        neg_individual_neighbor = h_1[multi_neg_node, sample_node.unsqueeze(0).expand_as(multi_neg_node), :]
        neg_individual_neighbor = F.normalize(neg_individual_neighbor, dim=2)

        pos_sub = self.read(h_2[:, :-1, :])
        neg_sub = pos_sub[multi_neg_node]

        node_pos = torch.einsum("nc,nc->n", target_node, pos_individual_neighbor).unsqueeze(-1)
        node_neg = torch.einsum("nc,knc->nk", target_node, neg_individual_neighbor)
        node_logits = torch.cat([node_pos, node_neg], dim=1) / self.T

        sub_pos = torch.einsum("nc,nc->n", target_node, pos_sub).unsqueeze(-1)
        sub_neg = torch.einsum("nc,knc->nk", target_node, neg_sub)
        sub_logits = torch.cat([sub_pos, sub_neg], dim=1) / self.T

        labels = torch.zeros(node_logits.shape[0], dtype=torch.long, device=device)
        return node_logits, sub_logits, labels
