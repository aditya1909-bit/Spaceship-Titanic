from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feature_selection import FeatureSelectionConfig, make_feature_pipeline

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

@dataclass
class NNConfig:
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-3
    epochs: int = 80
    hidden_sizes: Tuple[int, ...] = (128, 64)
    dropout: float = 0.5
    val_size: float = 0.2
    random_state: int = 42


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits; apply sigmoid outside when needed
        return self.net(x).squeeze(-1)

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "spaceship-titanic"

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    y = train_df["Transported"].astype(int)

    return train_df, test_df, y


def build_feature_pipeline():
    config = FeatureSelectionConfig(
        numerical_features=[
            "Age",
            "RoomService",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck",
        ],
        categorical_features=[
            "HomePlanet",
            "CryoSleep",
            "Cabin",
            "Destination",
            "VIP",
        ],
        poly_degree=1,    # no polynomial expansion; NN handles nonlinearity
        k_best=768,       # keep the 512 most informative features for better generalization
    )

    columns_to_drop = ["PassengerId", "Name", "Transported"]
    pipe = make_feature_pipeline(config, columns_to_drop=columns_to_drop)
    return pipe


def make_loaders(X: np.ndarray, y: pd.Series, cfg: NNConfig) -> tuple[DataLoader, DataLoader]:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y.values,
        test_size=cfg.val_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    X_train_t = torch.from_numpy(X_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.float32))

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: NNConfig) -> tuple[nn.Module, float]:
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = 0.0
    best_state: dict | None = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                train_correct += (preds == yb).sum().item()
                train_total += yb.numel()
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        model.train()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc

def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Using device: {DEVICE}")

    train_df, test_df, y = load_raw_data()
    pipe = build_feature_pipeline()

    print("Fitting feature pipeline...")
    X_all = pipe.fit_transform(train_df, y)
    X_test = pipe.transform(test_df)

    print(f"Feature matrix shape: {X_all.shape}")

    nn_cfg = NNConfig()
    train_loader, val_loader = make_loaders(X_all, y, nn_cfg)

    input_dim = X_all.shape[1]
    model = MLP(input_dim=input_dim, hidden_sizes=nn_cfg.hidden_sizes, dropout=nn_cfg.dropout)

    print("Training neural network...")
    model, best_val_acc = train_model(model, train_loader, val_loader, nn_cfg)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    model.to(DEVICE)
    model.eval()
    X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= 0.5).astype(int)

    root = Path(__file__).resolve().parents[1]
    out_path = root / "spaceship-titanic" / "submission_nn.csv"
    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Transported": preds.astype(bool),
        }
    )
    submission.to_csv(out_path, index=False)
    print(f"Wrote neural-network submission to {out_path}")


if __name__ == "__main__":
    main()