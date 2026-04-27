"""
baselines.py
============
Baselines de comparación para el modelo de predicción de flujos.

Evalúa 4 baselines con 7 métricas: MAE, RMSE, MAPE (estándar) +
Directional Accuracy, Peak Detection F1, Spearman ρ, SMAPE (AdTech).

Baselines:
    1. Historical Mean   — media global de cada POI sobre train
    2. Last Value        — último valor observado repetido en todo el horizonte
    3. Seasonal Naive    — mismo slot de la semana anterior
                          (Madrid: -168h | NYC: -7d)
    4. Period Mean       — media por hora del día (Madrid) o día de semana (NYC)

Uso:
    python baselines.py
"""

import numpy as np
import os
import sys
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_CONFIG, DATA

# ─────────────────────────────────────────────────────────────────────────────
# Parámetros de periodicidad
# ─────────────────────────────────────────────────────────────────────────────
_IS_MADRID   = DATA_CONFIG["city"] == "Madrid"
PERIOD       = 24 if _IS_MADRID else 7
WEEKLY_LAG   = 7 * PERIOD
PERIOD_LABEL = "Hora del día" if _IS_MADRID else "Día de semana"

# ─────────────────────────────────────────────────────────────────────────────
# Métricas estándar
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
# Métricas AdTech
# ─────────────────────────────────────────────────────────────────────────────

def directional_accuracy(pred, target, prev):
    """
    % de pasos en que la predicción acierta la dirección del cambio.

    pred, target, prev: [horizon, N]
    prev: valor en t-1 (el paso inmediatamente anterior al primer horizonte)
    """
    # Para cada horizonte h: comparar signo de (val[h] - val[h-1])
    # Concatenar prev como paso 0 para tener la referencia
    target_full = np.vstack([prev[None, :], target])   # [horizon+1, N]
    pred_full   = np.vstack([prev[None, :], pred])

    d_target = np.sign(target_full[1:] - target_full[:-1])  # [horizon, N]
    d_pred   = np.sign(pred_full[1:]   - pred_full[:-1])

    # Ignorar pasos sin cambio real (d_target == 0)
    valid = d_target != 0
    if valid.sum() == 0:
        return np.nan
    return (d_target[valid] == d_pred[valid]).mean() * 100


def smape(pred, target, eps=1.0):
    """SMAPE simétrico (%). eps evita división por cero."""
    return (2 * np.abs(pred - target) / (np.abs(pred) + np.abs(target) + eps)).mean() * 100


def peak_f1(pred, target, thresholds):
    """
    Detección de picos: F1 sobre si pred supera el umbral cuando target lo supera.

    pred, target : [horizon, N]
    thresholds   : [N]  umbral por POI (percentil 80 sobre train)
    """
    real_peak = (target > thresholds[None, :])   # [horizon, N]
    pred_peak = (pred   > thresholds[None, :])

    tp = (real_peak &  pred_peak).sum()
    fp = (~real_peak & pred_peak).sum()
    fn = (real_peak & ~pred_peak).sum()

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)
    return float(precision * 100), float(recall * 100), float(f1 * 100)


def mean_spearman(pred, target):
    """
    Correlación de Spearman entre rankings de POIs, promediada sobre horizontes.
    pred, target: [horizon, N]
    """
    rhos = []
    for h in range(pred.shape[0]):
        if np.std(pred[h]) < 1e-8 or np.std(target[h]) < 1e-8:
            continue
        rho, _ = spearmanr(pred[h], target[h])
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    print(f"Configuración: {DATA_CONFIG['config_name']}")
    flow = np.load(DATA_CONFIG["flow_path"])   # [T, N]
    print(f"Shape flow: {flow.shape}")
    print(f"Granularidad: {'horaria' if _IS_MADRID else 'diaria'} "
          f"| Período: {PERIOD} | Lag semanal: {WEEKLY_LAG}")

    T           = len(flow)
    n_train     = int(T * DATA["train_ratio"])
    n_val       = int(T * DATA["val_ratio"])
    history_len = DATA["history_len"]
    horizon     = DATA["horizon"]
    test_start  = n_train + n_val
    flow_train  = flow[:n_train]

    print(f"\nTrain: {n_train} | Val: {n_val} | "
          f"Test: {T - n_train - n_val} pasos")
    print(f"Horizonte: {horizon} | Historial: {history_len}")

    num_test_windows = T - horizon - test_start + 1
    print(f"Ventanas de test: {num_test_windows}")

    # ── 2. Estructuras para baselines estacionales ────────────────────────────
    # Period Mean
    num_nodes   = flow.shape[1]
    period_mean = np.zeros((PERIOD, num_nodes), dtype=np.float32)
    for p in range(PERIOD):
        idxs = [t for t in range(n_train) if t % PERIOD == p]
        if idxs:
            period_mean[p] = flow_train[idxs].mean(axis=0)

    # Historical Mean
    historical_mean = flow_train.mean(axis=0)   # [N]

    # Umbrales de pico: percentil 80 por POI sobre train
    peak_thresholds = np.percentile(flow_train, 80, axis=0)  # [N]

    # ── 3. Bucle de evaluación ────────────────────────────────────────────────
    metrics_keys = ["mae", "rmse", "mape", "da", "smape", "p_prec", "p_rec", "p_f1", "spearman"]
    baselines = ["Historical Mean", "Last Value", "Seasonal Naive", "Period Mean"]
    results   = {b: {k: [] for k in metrics_keys} for b in baselines}
    skipped_seasonal = 0

    for i in range(num_test_windows):
        t    = test_start + i
        Y    = flow[t:t + horizon]       # [horizon, N]
        prev = flow[t - 1]               # [N]  valor en t-1

        preds = {}

        # Historical Mean
        preds["Historical Mean"] = np.tile(historical_mean, (horizon, 1))

        # Last Value
        preds["Last Value"] = np.tile(prev, (horizon, 1))

        # Seasonal Naive
        lag = t - WEEKLY_LAG
        if lag >= 0 and lag + horizon <= T:
            preds["Seasonal Naive"] = flow[lag:lag + horizon]
        else:
            preds["Seasonal Naive"] = None
            skipped_seasonal += 1

        # Period Mean
        preds["Period Mean"] = np.stack([
            period_mean[(t + step) % PERIOD] for step in range(horizon)
        ])

        for name, pred in preds.items():
            if pred is None:
                continue
            r = results[name]
            r["mae"].append(masked_mae(pred, Y))
            r["rmse"].append(masked_rmse(pred, Y))
            r["mape"].append(masked_mape(pred, Y))
            r["da"].append(directional_accuracy(pred, Y, prev))
            r["smape"].append(smape(pred, Y))
            prec, rec, f1 = peak_f1(pred, Y, peak_thresholds)
            r["p_prec"].append(prec)
            r["p_rec"].append(rec)
            r["p_f1"].append(f1)
            r["spearman"].append(mean_spearman(pred, Y))

    if skipped_seasonal > 0:
        print(f"  (Seasonal Naive: {skipped_seasonal} ventanas saltadas)")

    # ── 4. Resultados ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"BASELINES — {DATA_CONFIG['config_name']}")
    print("=" * 90)

    # Tabla métricas estándar
    print(f"\n{'Baseline':<20} {'MAE':>7} {'RMSE':>7} {'MAPE':>8}  "
          f"{'DA%':>7} {'SMAPE':>7} {'PeakP':>7} {'PeakR':>7} {'PeakF1':>7} {'Spear':>7}")
    print("-" * 90)

    for name in baselines:
        r = results[name]
        if not r["mae"]:
            print(f"  {name:<18}  — sin datos —")
            continue

        def avg(k):
            vals = [v for v in r[k] if not np.isnan(v)]
            return np.mean(vals) if vals else float("nan")

        print(
            f"  {name:<18}  "
            f"{avg('mae'):>7.4f} "
            f"{avg('rmse'):>7.4f} "
            f"{avg('mape'):>7.2f}%  "
            f"{avg('da'):>6.1f}% "
            f"{avg('smape'):>6.1f}% "
            f"{avg('p_prec'):>6.1f}% "
            f"{avg('p_rec'):>6.1f}% "
            f"{avg('p_f1'):>6.1f}% "
            f"{avg('spearman'):>7.3f}"
        )

    print("=" * 90)
    print("\nLeyenda:")
    print("  DA      — Directional Accuracy: % aciertos en dirección del cambio")
    print("  SMAPE   — Symmetric MAPE (más estable con flujos bajos)")
    print("  PeakP/R/F1 — Precisión/Recall/F1 detección de picos (umbral: p80 train)")
    print("  Spear   — Correlación de Spearman entre rankings de POIs")
    print(f"\n  Período: {PERIOD_LABEL} ({PERIOD} slots) | "
          f"Lag semanal: {WEEKLY_LAG} pasos")


if __name__ == "__main__":
    main()