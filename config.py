"""
Configuration for weekly sector rotation model.

Key differences from black mamba:
- 11 sector ETFs + SPY (12 variates, not 51)
- Weekly frequency (not daily)
- Predicts relative returns for ALL sectors (not just SPY)
- Strategy: long top sectors, short bottom sectors
"""
import json
from dataclasses import dataclass, field
from pathlib import Path


_MODEL_PARAM_KEYS = {"d_model", "n_heads", "lstm_layers", "dropout", "lookback_len"}
_TRAIN_PARAM_KEYS = {"learning_rate", "weight_decay", "batch_size", "warmup_epochs"}


@dataclass
class DataConfig:
    """Data download and storage configuration."""
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    start_date: str = "2010-01-01"
    end_date: str = "2026-04-01"


@dataclass
class FeaturesConfig:
    """Feature engineering configuration."""
    features: tuple = (
        "returns",
        "log_returns",
        "weekly_range",
        "open_close_return",
        "log_volume",
        "relative_volume",
        "rsi_4",
        "price_vs_sma4",
        "realized_vol_4",
        "return_rank",
    )
    normalization: str = "zscore"
    max_missing_ratio: float = 0.05


@dataclass
class ModelConfig:
    """Temporal Fusion Transformer for sector rotation."""
    lookback_len: int = 12        # 12 weeks of history (~3 months)
    forecast_len: int = 1         # Predict 1 week ahead
    num_variates: int = 12        # 11 sectors + SPY (set dynamically)
    n_features: int = 10          # Features per variate (set dynamically)

    # TFT architecture
    d_model: int = 64
    n_heads: int = 4
    lstm_layers: int = 2
    dropout: float = 0.1

    # iTransformer compat (unused but avoids errors)
    n_layers: int = 3
    d_ff: int = 256
    activation: str = "gelu"
    norm_type: str = "pre"

    predict_spy_only: bool = False


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    patience: int = 15
    loss_fn: str = "mse"          # MSE for relative return prediction
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    seed: int = 42
    device: str = "auto"
    log_interval: int = 0

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio = 1 - train - val = 0.15

    checkpoint_dir: Path = Path("checkpoints")
    results_dir: Path = Path("results")


@dataclass
class Config:
    data: DataConfig = None
    features: FeaturesConfig = None
    model: ModelConfig = None
    train: TrainConfig = None
    auto_load_best_params: bool = True

    def __post_init__(self):
        self.data = self.data or DataConfig()
        self.features = self.features or FeaturesConfig()
        self.model = self.model or ModelConfig()
        self.train = self.train or TrainConfig()

        self.data.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.data.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.train.results_dir.mkdir(parents=True, exist_ok=True)

        if self.auto_load_best_params:
            self.load_best_params()

    def load_best_params(self, path: Path = None) -> bool:
        """Apply tuned hyperparameters from a best_params.json if present.

        Returns True if params were applied, False if the file was missing or
        unreadable. Unknown keys are ignored so the file stays forward-compatible
        with future search-space additions.
        """
        path = path or (self.train.results_dir / "best_params.json")
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text())
            params = payload.get("best_params", payload)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[config] Ignoring {path}: {e}")
            return False

        applied = {}
        for k, v in params.items():
            if k in _MODEL_PARAM_KEYS:
                setattr(self.model, k, v)
                applied[k] = v
            elif k in _TRAIN_PARAM_KEYS:
                setattr(self.train, k, v)
                applied[k] = v

        if applied:
            print(f"[config] Loaded tuned params from {path}: {applied}")
        return bool(applied)
