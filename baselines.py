"""
baselines.py
============
Baselines de comparación para el modelo de predicción de flujos.

Baselines implementados:
    1. Historical Mean: predice la media histórica de cada POI
    2. Last Value:      predice el último valor observado

Uso:
    Cambiar CONFIG_NAME según la configuración a evaluar y ejecutar:
    python baselines.py

"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_CONFIG, DATA

# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def masked_mae(pred, target, null_val=0.0):
    mask = (target != null_val).astype(float)
    denom = mask.mean()
    if denom < 1e-5:
        return 0.0
    mask /= denom
    return (np.abs(pred - target) * mask).mean()

def masked_rmse(pred, target, null_val=0.0):
    mask = (target != null_val).astype(float)
    denom = mask.mean()
    if denom < 1e-5:
        return 0.0
    mask /= denom
    return np.sqrt(((pred - target) ** 2 * mask).mean())

def masked_mape(pred, target, null_val=0.0):
    mask = (target != null_val).astype(float)
    denom = mask.mean()
    if denom < 1e-5:
        return 0.0
    mask /= denom
    return (np.abs((pred - target) / np.clip(target, 1e-5, None)) * mask).mean() * 100


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    print(f"Configuración: {DATA_CONFIG['config_name']}")
    flow  = np.load(DATA_CONFIG["flow_path"])   # [num_days, num_nodes]
    print(f"Shape flow: {flow.shape}")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    num_days = len(flow)
    n_train  = int(num_days * DATA["train_ratio"])
    n_val    = int(num_days * DATA["val_ratio"])

    flow_train = flow[:n_train]
    flow_test  = flow[n_train + n_val:]

    history_len = DATA["history_len"]
    horizon     = DATA["horizon"]

    print(f"Train: {len(flow_train)} días | Test: {len(flow_test)} días")
    print(f"Horizonte: {horizon} días | Historial: {history_len} días")

    # ── 3. Construir ventanas de test ─────────────────────────────────────────
    test_start = n_train + n_val
    num_test_windows = len(flow_test) - horizon + 1

    print(f"\nVentanas de test: {num_test_windows}")

    # ── 4. Historical Mean ────────────────────────────────────────────────────
    # Para cada POI, predice la media calculada sobre el train set
    historical_mean = flow_train.mean(axis=0)  # [num_nodes]

    mae_hm, rmse_hm, mape_hm = [], [], []
    for i in range(num_test_windows):
        t     = test_start + i
        Y     = flow[t:t + horizon]                        # [horizon, num_nodes]
        pred  = np.tile(historical_mean, (horizon, 1))     # [horizon, num_nodes]
        mae_hm.append(masked_mae(pred, Y))
        rmse_hm.append(masked_rmse(pred, Y))
        mape_hm.append(masked_mape(pred, Y))

    # ── 5. Last Value ─────────────────────────────────────────────────────────
    # Predice el valor del día anterior para todos los horizontes

    mae_lv, rmse_lv, mape_lv = [], [], []
    for i in range(num_test_windows):
        t    = test_start + i
        Y    = flow[t:t + horizon]                         # [horizon, num_nodes]
        last = flow[t - 1]                                 # [num_nodes]
        pred = np.tile(last, (horizon, 1))                 # [horizon, num_nodes]
        mae_lv.append(masked_mae(pred, Y))
        rmse_lv.append(masked_rmse(pred, Y))
        mape_lv.append(masked_mape(pred, Y))

    # ── 6. Resultados ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RESULTADOS BASELINES — {DATA_CONFIG['config_name']}")
    print("=" * 60)
    print(f"{'Baseline':<20} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print("-" * 60)
    print(f"{'Historical Mean':<20} {np.mean(mae_hm):>8.4f} {np.mean(rmse_hm):>8.4f} {np.mean(mape_hm):>7.2f}%")
    print(f"{'Last Value':<20} {np.mean(mae_lv):>8.4f} {np.mean(rmse_lv):>8.4f} {np.mean(mape_lv):>7.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()