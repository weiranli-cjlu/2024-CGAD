import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """GCN layer used by CGAD/CoLA-style subgraph encoders."""

    def __init__(self, in_ft, out_ft, act="prelu", bias=True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == "prelu" else act
        if bias:
            self.bias = nn.Parameter(torch.empty(out_ft))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)


class AvgReadout(nn.Module):
    def forward(self, seq):
        return torch.mean(seq, dim=1)


class MaxReadout(nn.Module):
    def forward(self, seq):
        return torch.max(seq, dim=1).values


class MinReadout(nn.Module):
    def forward(self, seq):
        return torch.min(seq, dim=1).values


class WSReadout(nn.Module):
    """Weighted-sum readout. Uses embedding norm as an attention score."""

    def forward(self, seq):
        weights = torch.softmax(torch.norm(seq, p=2, dim=-1, keepdim=True), dim=1)
        return torch.sum(seq * weights, dim=1)


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation="prelu", readout="avg", T=1.0):
        super().__init__()
        self.read_mode = readout
        self.T = T
        self.gcn_node = GCN(n_in, n_h, activation)
        self.gcn_context = GCN(n_in, n_h, activation)
        if readout == "max":
            self.read = MaxReadout()
        elif readout == "min":
            self.read = MinReadout()
        elif readout == "weighted_sum":
            self.read = WSReadout()
        else:
            self.read = AvgReadout()

    def forward(
        self,
        bf_mask,
        ba,
        bf,
        ba_mask,
        multi_neg_node,
        sample_node,
        sparse=False,
        msk=None,
        samp_bias1=None,
        samp_bias2=None,
    ):
        """Forward pass.

        multi_neg_node is a [K, B] tensor/array of negative sample positions inside
        the current batch. The original implementation used two Python list
        comprehensions with torch.stack; this version uses advanced indexing.
        """
        h_1 = self.gcn_node(bf, ba_mask, sparse)
        h_2 = self.gcn_context(bf_mask, ba, sparse)

        target_node = F.normalize(h_1[:, -1, :], dim=1)
        device = h_1.device
        batch_size = h_1.size(0)
        batch_index = torch.arange(batch_size, device=device)

        sample_node = torch.as_tensor(sample_node, dtype=torch.long, device=device)
        multi_neg_node = torch.as_tensor(multi_neg_node, dtype=torch.long, device=device)
        if multi_neg_node.dim() == 1:
            multi_neg_node = multi_neg_node.unsqueeze(0)

        pos_individual_neighbor = h_1[batch_index, sample_node, :]
        pos_individual_neighbor = F.normalize(pos_individual_neighbor, dim=1)

        # h_1[multi_neg_node] -> [K, B, S, H]; gather the positive-neighbor
        # position for every batch item without a Python loop.
        neg_h = h_1[multi_neg_node]
        neg_sample_index = sample_node.view(1, batch_size, 1, 1).expand(
            multi_neg_node.size(0), batch_size, 1, h_1.size(-1)
        )
        neg_individual_neighbor = torch.gather(neg_h, dim=2, index=neg_sample_index).squeeze(2)
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
