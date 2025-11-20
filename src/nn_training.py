from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feature_selection import FeatureSelectionConfig, make_feature_pipeline, global_feature_engineering

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
    hidden_sizes: Tuple[int, ...] = (128, 64) # Unused by ResNet but kept for compatibility
    dropout: float = 0.3
    val_size: float = 0.2
    random_state: int = 42

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)

class ResNetTabular(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "spaceship-titanic"

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    train_df, test_df = global_feature_engineering(train_df, test_df)
    y = train_df["Transported"].astype(int)

    return train_df, test_df, y

def build_feature_pipeline():
    config = FeatureSelectionConfig(
        numerical_features=[
            "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", 
            "TotalSpending", "GroupSize", "FamilySize", "Num", "CabinRegion",
        ],
        categorical_features=[
            "HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "IsSolo",
        ],
        poly_degree=1,
        k_best=0, 
    )
    columns_to_drop = ["PassengerId", "Name", "Transported"]
    pipe = make_feature_pipeline(config, columns_to_drop=columns_to_drop)
    return pipe

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        val_loss += loss.item() * xb.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.numel()
    return val_loss / len(loader.dataset), correct / total if total > 0 else 0.0

def run_training_fold(model, train_loader, val_loader, cfg):
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, best_val_acc

def run_kfold(X_all, y_all, nn_cfg):
    """Runs 5-Fold CV and returns the list of trained models and mean accuracy."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_models = []
    input_dim = X_all.shape[1]

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), start=1):
        print(f"\n--- Fold {fold_idx} ---")
        
        X_tr = X_all.iloc[train_idx]
        X_va = X_all.iloc[val_idx]
        y_tr = y_all.iloc[train_idx]
        y_va = y_all.iloc[val_idx]

        # Prepare Tensors
        X_tr_t = torch.from_numpy(X_tr.values).float()
        X_va_t = torch.from_numpy(X_va.values).float()
        y_tr_t = torch.from_numpy(y_tr.values.astype(np.float32))
        y_va_t = torch.from_numpy(y_va.values.astype(np.float32))

        train_ds = TensorDataset(X_tr_t, y_tr_t)
        val_ds = TensorDataset(X_va_t, y_va_t)
        
        train_loader = DataLoader(train_ds, batch_size=nn_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=nn_cfg.batch_size, shuffle=False)

        model = ResNetTabular(input_dim=input_dim, hidden_dim=256, num_blocks=3, dropout=nn_cfg.dropout)
        model, best_acc = run_training_fold(model, train_loader, val_loader, nn_cfg)
        
        print(f"Fold {fold_idx} Best Acc: {best_acc:.4f}")
        fold_accuracies.append(best_acc)
        fold_models.append(model)

    return fold_models, sum(fold_accuracies) / len(fold_accuracies)

def predict_test(models, X_test):
    """Averages predictions from multiple models."""
    X_test_t = torch.from_numpy(X_test.values).float().to(DEVICE)
    all_probs = []
    
    for m in models:
        m.to(DEVICE)
        m.eval()
        with torch.no_grad():
            logits = m(X_test_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            
    return np.mean(np.stack(all_probs, axis=0), axis=0)

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

    # --- ROUND 1: Initial Training ---
    print("\n=== Starting Round 1: Standard 5-Fold CV ===")
    models, mean_acc = run_kfold(X_all, y, nn_cfg)
    print(f"\nRound 1 Mean Accuracy: {mean_acc:.4f}")

    # Generate Predictions
    probs = predict_test(models, X_test)

    # --- ROUND 2: Pseudo-Labeling ---
    # Select samples where confidence > 99% or < 1%
    high_conf_idx = np.where((probs > 0.99) | (probs < 0.01))[0]
    
    if len(high_conf_idx) > 0:
        print(f"\n=== Starting Round 2: Pseudo-Labeling with {len(high_conf_idx)} samples ===")
        
        X_pseudo = X_test.iloc[high_conf_idx]
        
        # FIX: Use the PREDICTED labels, not the training labels
        pseudo_labels = (probs[high_conf_idx] >= 0.5).astype(int)
        y_pseudo = pd.Series(pseudo_labels)
        
        # Combine Train + Pseudo
        X_total = pd.concat([X_all, X_pseudo], axis=0).reset_index(drop=True)
        y_total = pd.concat([y, y_pseudo], axis=0).reset_index(drop=True)
        
        # Retrain
        models, mean_acc = run_kfold(X_total, y_total, nn_cfg)
        print(f"\nRound 2 (Pseudo) Mean Accuracy: {mean_acc:.4f}")
        
        # Re-predict with improved models
        probs = predict_test(models, X_test)

    # --- Submission ---
    preds = (probs >= 0.5).astype(int)
    root = Path(__file__).resolve().parents[1]
    out_path = root / "spaceship-titanic" / "submission_nn.csv"
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": preds.astype(bool),
    })
    submission.to_csv(out_path, index=False)
    print(f"Wrote neural-network submission to {out_path}")

if __name__ == "__main__":
    main()