# graph_transformer.py
# GraphTransformerDLP with encoder-aware defaults and flexible decoder.
# UPDATED (safe init + safe fallback dims)

from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

# Prefer TransformerConv; fallback to GATv2Conv if unavailable
try:
    from torch_geometric.nn import TransformerConv as _Conv
    _CONV_NAME = "TransformerConv"
except Exception:
    from torch_geometric.nn import GATv2Conv as _Conv
    _CONV_NAME = "GATv2Conv"


class GraphTransformerDLP(nn.Module):
    """
    Lightweight graph-transformer stack for link prediction.

    Args:
        in_channels (int): Input feature dimension.
        hidden_channels (int): Hidden size.
        out_channels (int): Output node-embedding size (decoder space).
        num_layers (int): Number of graph attention/transformer layers.
        heads (Optional[int]): Attention heads; if None, chosen by encoder_hint.
        dropout (Optional[float]): Dropout prob; if None, chosen by encoder_hint.
        adapter (Optional[bool]): Use input adapter (LN->Linear->GELU->Dropout->Linear);
                                  if None, chosen by encoder_hint.
        residual (bool): Use residual connections + LayerNorm.
        decoder (str): 'bilinear' (default) | 'dot' | 'mlp'.
        encoder_hint (Optional[str]): 'bert' or 'llama' to auto-tune defaults.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        heads: Optional[int] = None,
        dropout: Optional[float] = None,
        adapter: Optional[bool] = None,
        residual: bool = True,
        decoder: str = "bilinear",
        encoder_hint: Optional[str] = None,
    ):
        super().__init__()

        enc = (encoder_hint or "").lower()

        # Encoder-aware defaults (can be overridden by explicit args)
        default_heads   = 4 if enc == "bert" else (2 if enc == "llama" else 4)
        default_dropout = 0.20 if enc == "bert" else (0.10 if enc == "llama" else 0.20)
        default_adapter = True if enc == "bert" else (False if enc == "llama" else True)

        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        self.heads = int(heads if heads is not None else default_heads)
        self.dropout = float(dropout if dropout is not None else default_dropout)
        self.use_adapter = default_adapter if adapter is None else bool(adapter)
        self.residual = bool(residual)
        self.decoder_type = str(decoder).lower()
        self.encoder_hint = enc

        # ---- Input adapter (helps BERT-type features; off by default for LLaMA via encoder_hint) ----
        if self.use_adapter:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.in_channels),
                nn.Linear(self.in_channels, self.in_channels),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.in_channels, self.in_channels),
            )
        else:
            self.adapter = nn.Identity()

        # Stem to hidden size
        self.stem = nn.Linear(self.in_channels, self.hidden_channels)

        # ---- Graph layers ----
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            if _CONV_NAME == "TransformerConv":
                conv = _Conv(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    heads=self.heads,
                    concat=False,           # keep dim = hidden_channels
                    dropout=self.dropout,
                    edge_dim=None,
                    bias=True,
                )
            else:  # GATv2Conv fallback (make sure output dim == hidden_channels)
                # Ensure hidden_channels divisible by heads; otherwise force heads=1
                out_per_head = self.hidden_channels // self.heads
                if out_per_head * self.heads != self.hidden_channels:
                    self.heads = 1
                    out_per_head = self.hidden_channels
                concat = False  # output dim == out_channels when concat=False

                conv = _Conv(
                    in_channels=self.hidden_channels,
                    out_channels=out_per_head,
                    heads=self.heads,
                    dropout=self.dropout,
                    add_self_loops=True,
                    bias=True,
                    concat=concat,
                )

            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(self.hidden_channels))

        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)

        # Final projection to node embedding space
        self.out_proj = nn.Linear(self.hidden_channels, self.out_channels)

        # ---- Decoders ----
        if self.decoder_type == "bilinear":
            self.bilinear = nn.Parameter(torch.empty(self.out_channels, self.out_channels))
            nn.init.xavier_uniform_(self.bilinear)
        elif self.decoder_type == "mlp":
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.out_channels * 2, self.out_channels),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.out_channels, 1),
            )
        elif self.decoder_type == "dot":
            # no parameters
            pass
        else:
            raise ValueError(f"decoder must be one of: bilinear|dot|mlp (got {decoder})")

        self.reset_parameters()

    def reset_parameters(self):
        # Stem/out projection
        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.zeros_(self.stem.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Adapter linear layers (if enabled)
        if isinstance(self.adapter, nn.Sequential):
            for m in self.adapter.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        # Decoder params
        if hasattr(self, "bilinear"):
            nn.init.xavier_uniform_(self.bilinear)
        if hasattr(self, "edge_mlp"):
            for m in self.edge_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_index_pos: Optional[torch.Tensor] = None,
    ):
        """
        x: (N, F) node features
        edge_index: (2, E) COO edges
        edge_index_pos: unused (kept for API compatibility with your trainer)
        returns: z (N, out_channels)
        """
        # Adapter (Identity if disabled)
        x = self.adapter(x)

        # Stem
        h = self.stem(x)

        # Graph stack
        for conv, ln in zip(self.layers, self.norms):
            h_next = conv(h, edge_index)
            h_next = self.act(h_next)
            h_next = self.drop(h_next)
            if self.residual:
                h = ln(h + h_next)
            else:
                h = ln(h_next)

        # Node embedding space
        z = self.out_proj(h)
        return z

    @torch.no_grad()
    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x, edge_index)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return logits for edges in edge_index suitable for BCEWithLogitsLoss."""
        src, dst = edge_index[0], edge_index[1]
        z_u, z_v = z[src], z[dst]

        if self.decoder_type == "bilinear":
            mid = torch.matmul(z_u, self.bilinear)
            logits = (mid * z_v).sum(dim=-1)
            return logits
        elif self.decoder_type == "mlp":
            return self.edge_mlp(torch.cat([z_u, z_v], dim=-1)).squeeze(-1)
        else:  # 'dot'
            return (z_u * z_v).sum(dim=-1)
