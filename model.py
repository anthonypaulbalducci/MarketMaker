"""
Temporal Fusion Transformer for sector rotation.

Same TFT architecture as the SPY version but with multi-output:
- Input: all tickers (SPY + 11 sectors) as variates
- Output: relative return predictions for all 11 sectors
- Variable Selection Network learns which sectors/features matter
- LSTM captures temporal patterns in sector momentum
- Attention finds which past weeks matter most

Reference: Lim et al., "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting", 2021.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.fc = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_output)
        self.glu = GatedLinearUnit(d_output, d_output)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_output)
        self.skip = nn.Linear(d_input, d_output) if d_input != d_output else None

    def forward(self, x):
        residual = self.skip(x) if self.skip is not None else x
        h = self.dropout(F.elu(self.fc2(F.elu(self.fc1(x)))))
        h = self.glu(h)
        return self.norm(h + residual)


class ScaledVariableSelectionNetwork(nn.Module):
    """Variate-level variable selection for sector ETFs."""

    def __init__(self, n_variates, n_features_per, d_model, dropout=0.1):
        super().__init__()
        self.n_variates = n_variates
        self.variate_projections = nn.Linear(n_features_per, d_model)
        self.weight_grn = GatedResidualNetwork(
            n_variates * d_model, d_model, n_variates, dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.variate_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

    def forward(self, x):
        """
        Args: x: (batch, time, n_variates, n_features_per)
        Returns: selected: (batch, time, d_model), weights: (batch, time, n_variates)
        """
        B, T, N, F = x.shape
        projected = self.variate_projections(x)      # (B, T, N, d_model)
        flat = projected.reshape(B, T, -1)           # (B, T, N*d_model)
        weights = self.softmax(self.weight_grn(flat)) # (B, T, N)

        transformed = self.variate_grn(projected.reshape(B * T * N, -1))
        transformed = transformed.reshape(B, T, N, -1)

        selected = (transformed * weights.unsqueeze(-1)).sum(dim=2)
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        B, T, D = q.shape
        Q = self.W_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out), attn


class TFTSectorRotation(nn.Module):
    """
    TFT for sector rotation.

    1. Variable Selection: which sectors/features matter at each timestep
    2. LSTM Encoder: temporal sector momentum patterns
    3. Self-attention: which past weeks matter for this week's prediction
    4. Per-sector output heads: predicted relative returns for each sector
    """

    def __init__(
        self,
        lookback_len: int,
        num_variates: int,
        n_features: int,
        n_sectors: int,
        d_model: int = 64,
        n_heads: int = 4,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        sector_indices: list = None,
    ):
        super().__init__()
        self.num_variates = num_variates
        self.n_sectors = n_sectors
        self.sector_indices = sector_indices or list(range(1, num_variates))

        # 1. Variable Selection
        self.vsn = ScaledVariableSelectionNetwork(
            num_variates, n_features, d_model, dropout)

        # 2. Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, lookback_len, d_model) * 0.02)

        # 3. LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0)

        # 4. Post-LSTM gate
        self.post_lstm_glu = GatedLinearUnit(d_model, d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # 5. Self-attention
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.post_attn_glu = GatedLinearUnit(d_model, d_model)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # 6. Feed-forward
        self.ff_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # 7. Per-sector output projection
        self.sector_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_sectors),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args: x: (batch, lookback_len, num_variates, n_features)
        Returns: (batch, n_sectors)
        """
        B, T, N, F = x.shape

        # Variable selection
        selected, var_weights = self.vsn(x)
        selected = selected + self.pos_encoding

        # LSTM
        lstm_out, _ = self.lstm(selected)
        gated = self.post_lstm_glu(lstm_out)
        temporal = self.post_lstm_norm(gated + selected)

        # Causal self-attention
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        mask = (mask == 0).unsqueeze(0).unsqueeze(0)
        attn_out, _ = self.attention(temporal, temporal, temporal, mask)
        attn_gated = self.post_attn_glu(attn_out)
        enriched = self.post_attn_norm(attn_gated + temporal)

        # Feed-forward
        ff_out = self.ff_grn(enriched)

        # Take last timestep → predict all sectors
        last_step = ff_out[:, -1, :]
        preds = self.sector_projection(last_step)  # (B, n_sectors)

        return preds

    def get_variable_importance(self, x):
        with torch.no_grad():
            _, var_weights = self.vsn(x)
        return var_weights


def build_model(cfg, spy_index=0, sector_indices=None, n_sectors=11):
    """Build TFT sector rotation model."""
    model = TFTSectorRotation(
        lookback_len=cfg.model.lookback_len,
        num_variates=cfg.model.num_variates,
        n_features=cfg.model.n_features,
        n_sectors=n_sectors,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        lstm_layers=cfg.model.lstm_layers,
        dropout=cfg.model.dropout,
        sector_indices=sector_indices,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    return model
