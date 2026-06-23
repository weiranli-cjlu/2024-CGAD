import torch
import torch.nn as nn


class GCN(nn.Module):
    """GCN layer used by CGAD/CoLA-style subgraph encoders."""

    def __init__(self, in_ft, out_ft, act="prelu", bias=True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == "prelu" else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

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
    """Weighted-sum readout. Falls back to mean-style uniform weights."""

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
        h_1 = self.gcn_node(bf, ba_mask, sparse)
        h_2 = self.gcn_context(bf_mask, ba, sparse)

        target_node = h_1[:, -1, :]
        target_node = nn.functional.normalize(target_node, dim=1)

        batch_index = torch.arange(len(sample_node), device=h_1.device)
        sample_node = torch.as_tensor(sample_node, dtype=torch.long, device=h_1.device)
        multi_neg_node = torch.as_tensor(multi_neg_node, dtype=torch.long, device=h_1.device)

        pos_individual_neighbor = h_1[batch_index, sample_node, :].squeeze()
        pos_individual_neighbor = nn.functional.normalize(pos_individual_neighbor, dim=1)

        neg_individual_neighbor = torch.stack(
            [h_1[node_index, sample_node, :].squeeze() for node_index in multi_neg_node]
        )
        neg_individual_neighbor = nn.functional.normalize(neg_individual_neighbor, dim=2)

        pos_sub = self.read(h_2[:, :-1, :])
        neg_sub = torch.stack([pos_sub[node_index] for node_index in multi_neg_node])

        node_pos = torch.einsum("nc,nc->n", target_node, pos_individual_neighbor).unsqueeze(-1)
        node_neg = torch.einsum("nc,knc->nk", target_node, neg_individual_neighbor)
        node_logits = torch.cat([node_pos, node_neg], dim=1) / self.T

        sub_pos = torch.einsum("nc,nc->n", target_node, pos_sub).unsqueeze(-1)
        sub_neg = torch.einsum("nc,knc->nk", target_node, neg_sub)
        sub_logits = torch.cat([sub_pos, sub_neg], dim=1) / self.T

        labels = torch.zeros(node_logits.shape[0], dtype=torch.long, device=h_1.device)
        return node_logits, sub_logits, labels
