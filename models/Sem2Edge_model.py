import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class SnapshotStructuralEncoder(nn.Module):
    """
    Per-snapshot structural encoder using Graph Transformer layers
    implemented with PyG TransformerConv.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()

        self.conv1 = TransformerConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
            beta=False,
        )

        self.conv2 = TransformerConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
            beta=False,
        )

        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.gelu(h)
        return h


class SnapshotSemanticEncoder(nn.Module):
    """
    Per-snapshot semantic encoder using an MLP projection.
    """

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class CrossStreamAttention(nn.Module):
    """
    Cross-stream attention between structural and semantic states.

    struct queries semantic:
        h_struct_enh = h_struct + Attn(h_struct, h_sem, h_sem)

    semantic queries structure:
        h_sem_enh = h_sem + Attn(h_sem, h_struct, h_struct)
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.attn_struct_to_sem = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_sem_to_struct = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_struct = nn.LayerNorm(dim)
        self.norm_sem = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, h_struct, h_sem):
        """
        h_struct: [N, D]
        h_sem:    [N, D]

        We treat each node as a sequence length of 1 for cross-stream attention.
        """
        q_struct = h_struct.unsqueeze(1)  # [N,1,D]
        q_sem = h_sem.unsqueeze(1)

        sem_ctx, _ = self.attn_struct_to_sem(q_struct, q_sem, q_sem)
        struct_ctx, _ = self.attn_sem_to_struct(q_sem, q_struct, q_struct)

        h_struct = self.norm_struct(h_struct + F.dropout(sem_ctx.squeeze(1), p=self.dropout, training=self.training))
        h_sem = self.norm_sem(h_sem + F.dropout(struct_ctx.squeeze(1), p=self.dropout, training=self.training))

        return h_struct, h_sem


class DualTemporalText2Edge(nn.Module):
    """
    Dual Temporal Graph Transformer-GRU model with cross-stream attention.

    Structural stream:
        Graph Transformer encoder per snapshot + GRU over time

    Semantic stream:
        MLP encoder per snapshot + GRU over time

    Cross-stream interaction:
        structural state attends to semantic state
        semantic state attends to structural state

    Fusion:
        gated fusion of the enhanced temporal states
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        out_channels=64,
        gat_heads=4,   # kept for trainer compatibility
        dropout=0.1,
    ):
        super().__init__()

        self.struct_encoder = SnapshotStructuralEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=gat_heads,
            dropout=dropout,
        )

        self.sem_encoder = SnapshotSemanticEncoder(
            in_channels=in_channels,
            out_channels=hidden_channels,
            dropout=dropout,
        )

        self.gru_struct = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            batch_first=True,
        )
        self.gru_sem = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            batch_first=True,
        )

        # NEW: cross-stream attention
        self.cross_stream = CrossStreamAttention(hidden_channels, dropout=dropout)

        self.struct_out = nn.Linear(hidden_channels, out_channels)
        self.sem_out = nn.Linear(hidden_channels, out_channels)

        self.gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_seq, edge_index_seq):
        """
        x_seq: list of length T, each tensor [N, F]
        edge_index_seq: list of length T, each tensor [2, E]
        """
        assert len(x_seq) == len(edge_index_seq), "x_seq and edge_index_seq must have same length"

        struct_seq = []
        sem_seq = []

        for x_t, ei_t in zip(x_seq, edge_index_seq):
            z_struct_t = self.struct_encoder(x_t, ei_t)
            z_sem_t = self.sem_encoder(x_t)

            struct_seq.append(z_struct_t)
            sem_seq.append(z_sem_t)

        # [T, N, H] -> [N, T, H]
        struct_seq = torch.stack(struct_seq, dim=0).transpose(0, 1).contiguous()
        sem_seq = torch.stack(sem_seq, dim=0).transpose(0, 1).contiguous()

        h_struct_seq, _ = self.gru_struct(struct_seq)
        h_sem_seq, _ = self.gru_sem(sem_seq)

        # final temporal states
        h_struct = h_struct_seq[:, -1, :]
        h_sem = h_sem_seq[:, -1, :]

        # NEW: cross-stream interaction
        h_struct, h_sem = self.cross_stream(h_struct, h_sem)

        h_struct = self.struct_out(h_struct)
        h_sem = self.sem_out(h_sem)

        gate_input = torch.cat([h_struct, h_sem], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))

        z = g * h_struct + (1.0 - g) * h_sem
        return z

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)