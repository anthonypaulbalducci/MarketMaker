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
from typing import Optional


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


# Default search locations for a tuned-hyperparameters JSON file.
# Checked in order; the first existing file is used.
DEFAULT_BEST_PARAMS_PATHS = [
    "best_params.json",
    "results/best_params.json",
    "tuning/best_params.json",
    "checkpoints/best_params.json",
]


@dataclass
class Config:
    data: DataConfig = None
    features: FeaturesConfig = None
    model: ModelConfig = None
    train: TrainConfig = None

    def __post_init__(self):
        self.data = self.data or DataConfig()
        self.features = self.features or FeaturesConfig()
        self.model = self.model or ModelConfig()
        self.train = self.train or TrainConfig()

        self.data.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.data.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.train.results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Tuned-hyperparameter management
    # ---------------------------------------------------------------

    def load_best_params(self, path: Optional[str] = None) -> bool:
        """
        Load tuned hyperparameters from a JSON file (typically written by tune.py).

        The file should be a flat dictionary mapping hyperparameter names to values,
        matching Optuna's `study.best_params` format. Example:

            {
              "d_model": 128,
              "n_heads": 8,
              "lstm_layers": 3,
              "dropout": 0.2,
              "learning_rate": 0.0005
            }

        Args:
            path: Explicit path to the JSON file. If None, searches the default
                  locations defined in DEFAULT_BEST_PARAMS_PATHS.

        Returns:
            True if at least one parameter was loaded and applied; False otherwise.
        """
        candidates = [path] if path else DEFAULT_BEST_PARAMS_PATHS
        loaded_from = None
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                loaded_from = Path(candidate)
                break

        if loaded_from is None:
            tried = [c for c in candidates if c]
            print(f"Warning: no tuned-params file found. Searched: {tried}")
            print("         Keeping config.py defaults.")
            return False

        try:
            params = json.loads(loaded_from.read_text())
        except json.JSONDecodeError as e:
            print(f"Warning: could not parse {loaded_from}: {e}")
            print("         Keeping config.py defaults.")
            return False

        print(f"Loading tuned hyperparameters from {loaded_from}:")
        applied = 0
        for key, value in params.items():
            matched = False
            for section_name, section in (("model", self.model), ("train", self.train)):
                if hasattr(section, key):
                    old_value = getattr(section, key)
                    setattr(section, key, value)
                    print(f"  {section_name}.{key}: {old_value} -> {value}")
                    matched = True
                    applied += 1
                    break
            if not matched:
                print(f"  Warning: '{key}' not found on model or train config; ignored.")

        if applied == 0:
            print("  (no recognized hyperparameters found in file)")
            return False
        return True

    def save_best_params(self, path: str = "best_params.json") -> None:
        """
        Save the current tunable hyperparameters as a JSON file.

        Useful for capturing values after a tune run, or for converting manual
        config edits into a portable params file that load_best_params() can read.
        """
        tunable = {
            # Architectural
            "d_model": self.model.d_model,
            "n_heads": self.model.n_heads,
            "lstm_layers": self.model.lstm_layers,
            "dropout": self.model.dropout,
            "lookback_len": self.model.lookback_len,
            # Training
            "learning_rate": self.train.learning_rate,
            "weight_decay": self.train.weight_decay,
            "batch_size": self.train.batch_size,
        }
        Path(path).write_text(json.dumps(tunable, indent=2))
        print(f"Saved {len(tunable)} hyperparameters to {path}")
