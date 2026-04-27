"""
baselines_lstm.py
=================
Baseline LSTM para predicción de flujos de movilidad urbana.

Arquitectura:
    LSTM multicapa sobre la serie temporal de todos los POIs simultáneamente.
    Entrada:  [B, T, N]  — últimos history_len pasos de todos los POIs
    Salida:   [B, N, horizon]  — predicción para los próximos horizon pasos

    Este baseline captura dependencias temporales globales pero ignora
    completamente la estructura espacial (grafos). Sirve para cuantificar
    cuánto aporta la componente de grafos del modelo STAR + GWN.

Uso:
    python baselines_lstm.py

Resultados guardados en:
    checkpoints/<config_name>/lstm_best.pt
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_CONFIG, DATA, TRAIN


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  (igual que en train.py)
# ─────────────────────────────────────────────────────────────────────────────

class FlowDataset(Dataset):
    def __init__(self, flow, history_len, horizon, t_start, t_end):
        self.flow        = torch.tensor(flow, dtype=torch.float32)
        self.history_len = history_len
        self.horizon     = horizon
        self.t_start     = t_start
        self.num_samples = t_end - t_start + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        t = self.t_start + idx
        X = self.flow[t - self.history_len:t].T.unsqueeze(1)  # [N, 1, T]
        Y = self.flow[t:t + self.horizon].T                    # [N, horizon]
        return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# Modelo LSTM
# ─────────────────────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """
    LSTM sobre la serie temporal conjunta de todos los POIs.

    Toma el mismo formato de entrada que FullModel ([B, N, 1, T]) y devuelve
    el mismo formato de salida ([B, N, horizon]) para poder comparar
    directamente usando el mismo bucle de evaluación.

    Args:
        num_nodes  (int): número de POIs (N)
        hidden_dim (int): dimensión del estado oculto del LSTM
        horizon    (int): pasos a predecir
        num_layers (int): capas del LSTM. Default: 2
        dropout    (float): dropout entre capas LSTM. Default: 0.3
    """
    def __init__(self, num_nodes, hidden_dim, horizon,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.num_nodes  = num_nodes
        self.horizon    = horizon

        self.lstm = nn.LSTM(
            input_size  = num_nodes,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_nodes * horizon)

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, N, 1, T]  — mismo formato que FullModel
        Returns:
            pred (Tensor): [B, N, horizon]
        """
        B, N, _, T = x.shape
        # [B, N, 1, T] → [B, T, N]
        x_seq = x.squeeze(2).permute(0, 2, 1)

        out, _ = self.lstm(x_seq)           # [B, T, hidden_dim]
        last   = out[:, -1, :]             # [B, hidden_dim]
        pred   = self.fc(last)             # [B, N * horizon]
        return pred.view(B, self.horizon, N).permute(0, 2, 1)  # [B, N, horizon]


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def masked_mae(pred, target, null_val=0.0):
    mask = (target != null_val).float()
    mask /= mask.mean().clamp(min=1e-5)
    return (torch.abs(pred - target) * mask).mean()

def masked_rmse(pred, target, null_val=0.0):
    mask = (target != null_val).float()
    mask /= mask.mean().clamp(min=1e-5)
    return torch.sqrt(((pred - target) ** 2 * mask).mean())

def masked_mape(pred, target, null_val=0.0):
    mask = (target != null_val).float()
    mask /= mask.mean().clamp(min=1e-5)
    return (torch.abs((pred - target) / target.clamp(min=1e-5)) * mask).mean() * 100

def denormalize(x, mean, std):
    return x * (std + 1e-8) + mean


# ─────────────────────────────────────────────────────────────────────────────
# Bucles de entrenamiento y evaluación
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = masked_mae(pred, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, device, mean, std):
    model.eval()
    maes, rmses, mapes = [], [], []
    with torch.no_grad():
        for X, Y in loader:
            X, Y   = X.to(device), Y.to(device)
            pred   = model(X)
            p_real = denormalize(pred, mean, std)
            y_real = denormalize(Y,    mean, std)
            maes.append(masked_mae(p_real,  y_real).item())
            rmses.append(masked_rmse(p_real, y_real).item())
            mapes.append(masked_mape(p_real, y_real).item())
    return np.mean(maes), np.mean(rmses), np.mean(mapes)


# ─────────────────────────────────────────────────────────────────────────────
# Hiperparámetros LSTM
# ─────────────────────────────────────────────────────────────────────────────
LSTM_CONFIG = {
    "hidden_dim" : 256,
    "num_layers" : 2,
    "dropout"    : 0.3,
    "lr"         : 0.001,
    "epochs"     : 100,
    "patience"   : 15,
    "batch_size" : TRAIN["batch_size"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device(TRAIN["device"])
    Path(DATA_CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

    # ── 1. Datos ──────────────────────────────────────────────────────────────
    print(f"Configuración: {DATA_CONFIG['config_name']}")
    flow  = np.load(DATA_CONFIG["flow_path"])
    dates = np.load(DATA_CONFIG["dates_path"])
    print(f"  Shape: {flow.shape}  ({dates[0]} → {dates[-1]})")

    T           = len(flow)
    history_len = DATA["history_len"]
    horizon     = DATA["horizon"]
    num_nodes   = DATA["num_nodes"]
    n_train     = int(T * DATA["train_ratio"])
    n_val       = int(T * DATA["val_ratio"])

    # ── 2. Normalización Z-score (solo sobre train) ───────────────────────────
    mean = flow[:n_train].mean()
    std  = flow[:n_train].std()
    flow_norm = (flow - mean) / (std + 1e-8)
    mean_t = torch.tensor(mean, dtype=torch.float32).to(device)
    std_t  = torch.tensor(std,  dtype=torch.float32).to(device)
    print(f"  Media train: {mean:.4f}  Std train: {std:.4f}")

    # ── 3. Datasets ───────────────────────────────────────────────────────────
    train_t_start = history_len
    train_t_end   = n_train - horizon
    val_t_start   = n_train - horizon + 1
    val_t_end     = n_train + n_val - horizon
    test_t_start  = n_train + n_val - horizon + 1
    test_t_end    = T - horizon

    train_ds = FlowDataset(flow_norm, history_len, horizon, train_t_start, train_t_end)
    val_ds   = FlowDataset(flow_norm, history_len, horizon, val_t_start,   val_t_end)
    test_ds  = FlowDataset(flow_norm, history_len, horizon, test_t_start,  test_t_end)

    print(f"\n  Train: {len(train_ds)} muestras | "
          f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=LSTM_CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=LSTM_CONFIG["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=LSTM_CONFIG["batch_size"], shuffle=False)

    # ── 4. Modelo ─────────────────────────────────────────────────────────────
    model = LSTMBaseline(
        num_nodes  = num_nodes,
        hidden_dim = LSTM_CONFIG["hidden_dim"],
        horizon    = horizon,
        num_layers = LSTM_CONFIG["num_layers"],
        dropout    = LSTM_CONFIG["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  LSTMBaseline: {total_params:,} parámetros")
    print(f"  hidden_dim={LSTM_CONFIG['hidden_dim']}  "
          f"layers={LSTM_CONFIG['num_layers']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # ── 5. Entrenamiento ──────────────────────────────────────────────────────
    save_path   = os.path.join(DATA_CONFIG["output_dir"], "lstm_best.pt")
    best_val    = float("inf")
    patience    = 0

    print(f"\nEntrenando LSTM ({LSTM_CONFIG['epochs']} épocas, "
          f"patience={LSTM_CONFIG['patience']})...")
    print("-" * 65)

    for epoch in range(1, LSTM_CONFIG["epochs"] + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mae, val_rmse, val_mape = eval_epoch(
            model, val_loader, device, mean_t, std_t
        )
        scheduler.step()

        print(
            f"Época {epoch:3d}/{LSTM_CONFIG['epochs']} | "
            f"Loss: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val MAPE: {val_mape:.2f}% | "
            f"{time.time()-t0:.1f}s"
        )

        if val_mae < best_val:
            best_val = val_mae
            patience = 0
            torch.save(model.state_dict(), save_path)
            print(f"  → Mejor modelo guardado (Val MAE: {best_val:.4f})")
        else:
            patience += 1
            if patience >= LSTM_CONFIG["patience"]:
                print(f"\nEarly stopping en época {epoch}.")
                break

    print(f"\n✓ Entrenamiento completado.")
    print(f"  Mejor Val MAE: {best_val:.4f}")
    print(f"  Modelo guardado en: {save_path}")
    print(f"\n  Para evaluar junto al resto de modelos:")
    print(f"  python analysis/taptap/evaluation.py")


if __name__ == "__main__":
    main()