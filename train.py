"""
train.py
========
Training loop para el modelo de predicción de flujos de movilidad urbana.

Pipeline:
  1. Cargar y normalizar datos de flujo
  2. Construir ventanas deslizantes (X, Y)
  3. Split train / val / test
  4. Inicializar FullModel (STAREmbedding congelado + GraphWaveNet)
  5. Entrenar con early stopping y lr decay
  6. Evaluar en test con MAE, RMSE y MAPE

Uso:
    python train.py

Para cambiar horizonte de predicción u otros hiperparámetros:
    editar config.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DATA_CONFIG, DATA, STAR, GWN, TRAIN
from model.full_model import FullModel, load_supports


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FlowDataset(Dataset):
    """
    Dataset de ventanas deslizantes sobre la serie temporal de flujos.

    Recibe el array completo normalizado y un rango [t_start, t_end] de
    instantes de predicción válidos. Así val y test pueden usar historia
    del período anterior sin fuga de datos hacia el futuro.

    Cada muestra es:
        X: flow[t-history_len:t]  → [N, 1, history_len]  historial
        Y: flow[t:t+horizon]      → [N, horizon]          target

    Args:
        flow        (ndarray): [T, num_nodes] normalizado (array completo)
        history_len (int)    : pasos de historial
        horizon     (int)    : pasos a predecir
        t_start     (int)    : primer instante de predicción válido (inclusive)
        t_end       (int)    : último instante de predicción válido (inclusive)
    """
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
        # X: [N, history_len] → [N, 1, history_len]
        X = self.flow[t - self.history_len:t].T.unsqueeze(1)
        # Y: [N, horizon]
        Y = self.flow[t:t + self.horizon].T
        return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def masked_mae(pred, target, null_val=0.0):
    """MAE ignorando valores nulos (POIs sin visitas)."""
    mask = (target != null_val).float()
    mask /= mask.mean().clamp(min=1e-5)
    loss  = torch.abs(pred - target) * mask
    return loss.mean()

def masked_rmse(pred, target, null_val=0.0):
    """RMSE ignorando valores nulos."""
    mask = (target != null_val).float()
    mask /= mask.mean().clamp(min=1e-5)
    loss  = ((pred - target) ** 2) * mask
    return torch.sqrt(loss.mean())

def masked_mape(pred, target, null_val=0.0):
    """MAPE ignorando valores nulos."""
    mask = (target != null_val).float()
    mask /= mask.mean().clamp(min=1e-5)
    loss  = torch.abs((pred - target) / target.clamp(min=1e-5)) * mask
    return loss.mean() * 100


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def normalize(flow, mean=None, std=None):
    """Normalización Z-score. Calcula mean/std si no se pasan."""
    if mean is None:
        mean = flow.mean()
    if std is None:
        std  = flow.std()
    return (flow - mean) / (std + 1e-8), mean, std

def denormalize(flow_norm, mean, std):
    """Desnormalización para obtener flujos reales."""
    return flow_norm * (std + 1e-8) + mean


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)                      # [B, N, horizon]
        loss = masked_mae(pred, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.get_trainable_params(),
            TRAIN["grad_clip"]
        )
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, device, mean, std):
    model.eval()
    maes, rmses, mapes = [], [], []
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            # Desnormalizar para métricas en escala real
            pred_real = denormalize(pred, mean, std)
            Y_real    = denormalize(Y,    mean, std)
            maes.append(masked_mae(pred_real,  Y_real).item())
            rmses.append(masked_rmse(pred_real, Y_real).item())
            mapes.append(masked_mape(pred_real, Y_real).item())
    return np.mean(maes), np.mean(rmses), np.mean(mapes)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device(TRAIN["device"])
    Path(DATA_CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(DATA_CONFIG["logs_dir"]).mkdir(parents=True, exist_ok=True)

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    print("Cargando datos de flujo...")
    flow  = np.load(DATA_CONFIG["flow_path"])   # [num_days, num_nodes]
    dates = np.load(DATA_CONFIG["dates_path"])
    print(f"  Shape: {flow.shape}")
    print(f"  Fechas: {dates[0]} → {dates[-1]}")

    # ── 2. Split train / val / test ──────────────────────────────────────────
    history_len = DATA["history_len"]
    horizon     = DATA["horizon"]
    T           = len(flow)
    n_train     = int(T * DATA["train_ratio"])
    n_val       = int(T * DATA["val_ratio"])

    print(f"\nSplit temporal:")
    print(f"  Train: {n_train} pasos ({dates[0]} → {dates[n_train-1]})")
    print(f"  Val:   {n_val} pasos ({dates[n_train]} → {dates[n_train+n_val-1]})")
    print(f"  Test:  {T-n_train-n_val} pasos ({dates[n_train+n_val]} → {dates[-1]})")

    # Rangos de instantes de predicción (t es el primer paso de Y)
    # Train: Y cae íntegramente en el período train
    train_t_start = history_len
    train_t_end   = n_train - horizon
    # Val: Y cae en el período val (X puede solapar con train)
    val_t_start   = n_train - horizon + 1
    val_t_end     = n_train + n_val - horizon
    # Test: Y cae en el período test (X puede solapar con val)
    test_t_start  = n_train + n_val - horizon + 1
    test_t_end    = T - horizon

    # ── 3. Normalización Z-score (calculada solo sobre train) ────────────────
    print("\nNormalizando...")
    _, mean, std = normalize(flow[:n_train])
    flow_norm    = (flow - mean) / (std + 1e-8)
    mean_t = torch.tensor(mean, dtype=torch.float32).to(device)
    std_t  = torch.tensor(std,  dtype=torch.float32).to(device)
    print(f"  Media train: {mean:.4f}, Std train: {std:.4f}")

    # ── 4. Datasets y DataLoaders ────────────────────────────────────────────
    train_dataset = FlowDataset(flow_norm, history_len, horizon, train_t_start, train_t_end)
    val_dataset   = FlowDataset(flow_norm, history_len, horizon, val_t_start,   val_t_end)
    test_dataset  = FlowDataset(flow_norm, history_len, horizon, test_t_start,  test_t_end)

    print(f"\nVentanas deslizantes:")
    print(f"  Train: {len(train_dataset)} muestras")
    print(f"  Val:   {len(val_dataset)} muestras")
    print(f"  Test:  {len(test_dataset)} muestras")

    train_loader = DataLoader(train_dataset, batch_size=TRAIN["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=TRAIN["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=TRAIN["batch_size"], shuffle=False)

    # ── 5. Modelo ────────────────────────────────────────────────────────────
    print("\nCargando grafos y modelo...")
    supports = load_supports(DATA_CONFIG["graphs_dir"], device=str(device))

    model = FullModel(
        propath           = DATA_CONFIG["graphs_dir"],
        supports          = supports,
        device            = str(device),
        fea_dim           = STAR["fea_dim"],
        hid_dim           = STAR["hid_dim"],
        out_dim_emb       = STAR["out_dim"],
        layer_num         = STAR["layer_num"],
        head_num          = STAR["head_num"],
        out_dim_pred      = horizon,
        history_len       = history_len,
        residual_channels = GWN["residual_channels"],
        dilation_channels = GWN["dilation_channels"],
        skip_channels     = GWN["skip_channels"],
        end_channels      = GWN["end_channels"],
        dropout           = GWN["dropout"],
        blocks            = GWN["blocks"],
        layers            = GWN["layers"],
    ).to(device)

    print("\nParámetros del modelo:")
    model.count_params()

    # ── 6. Optimizador y scheduler ───────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.get_trainable_params(),
        lr           = TRAIN["lr"],
        weight_decay = TRAIN["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma = TRAIN["lr_decay"]
    )

    # ── 7. Training loop con early stopping ──────────────────────────────────
    print(f"\nEntrenando durante {TRAIN['epochs']} épocas...")
    print(f"  Horizonte: {horizon} días")
    print(f"  Early stopping: {TRAIN['patience']} épocas sin mejora")
    print("-" * 60)

    best_val_mae  = float("inf")
    patience_cnt  = 0
    best_model_path = os.path.join(
        DATA_CONFIG["output_dir"],
        f"best_model_h{horizon}.pt"
    )

    for epoch in range(1, TRAIN["epochs"] + 1):
        t0 = time.time()

        train_loss            = train_epoch(model, train_loader, optimizer, device)
        val_mae, val_rmse, val_mape = eval_epoch(model, val_loader, device, mean_t, std_t)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Época {epoch:3d}/{TRAIN['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val MAPE: {val_mape:.2f}% | "
            f"{elapsed:.1f}s"
        )

        # Guardar mejor modelo
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_cnt = 0
            if TRAIN["save_best"]:
                torch.save(model.state_dict(), best_model_path)
                print(f"  → Mejor modelo guardado (Val MAE: {best_val_mae:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= TRAIN["patience"]:
                print(f"\nEarly stopping en época {epoch}.")
                break

    # ── 8. Evaluación final en test ──────────────────────────────────────────
    print("\nEvaluando en test con mejor modelo...")
    model.load_state_dict(torch.load(best_model_path))
    test_mae, test_rmse, test_mape = eval_epoch(
        model, test_loader, device, mean_t, std_t
    )

    print("-" * 60)
    print(f"Resultados finales en test (horizonte {horizon} días):")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAPE: {test_mape:.2f}%")
    print("-" * 60)


if __name__ == "__main__":
    main()