"""
baselines.py
============
Baselines de comparación para el modelo de predicción de flujos.

Baselines implementados:
    1. Historical Mean   — media global de cada POI sobre train
    2. Last Value        — último valor observado repetido en todo el horizonte
    3. Seasonal Naive    — mismo slot de la semana anterior
                          (Madrid: -168h | NYC: -7d)
    4. Period Mean       — media por hora del día (Madrid) o día de semana (NYC)
                          captura el patrón periódico típico de cada POI

Uso:
    Cambiar CONFIG_NAME en config.py según la configuración a evaluar y ejecutar:
    python baselines.py
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_CONFIG, DATA

# ─────────────────────────────────────────────────────────────────────────────
# Parámetros de periodicidad (según granularidad del dataset)
# ─────────────────────────────────────────────────────────────────────────────
_IS_MADRID   = DATA_CONFIG["city"] == "Madrid"
PERIOD       = 24 if _IS_MADRID else 7    # horas/día  o  días/semana
WEEKLY_LAG   = 7 * PERIOD                 # 168h (Madrid) | 7d (NYC)
PERIOD_LABEL = "Hora del día" if _IS_MADRID else "Día de semana"

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
    flow = np.load(DATA_CONFIG["flow_path"])   # [T, num_nodes]
    print(f"Shape flow: {flow.shape}")
    print(f"Granularidad: {'horaria' if _IS_MADRID else 'diaria'} "
          f"| Período: {PERIOD} | Lag semanal: {WEEKLY_LAG}")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    T        = len(flow)
    n_train  = int(T * DATA["train_ratio"])
    n_val    = int(T * DATA["val_ratio"])

    flow_train = flow[:n_train]
    flow_test  = flow[n_train + n_val:]

    history_len = DATA["history_len"]
    horizon     = DATA["horizon"]
    test_start  = n_train + n_val

    print(f"\nTrain: {n_train} pasos | Val: {n_val} pasos | "
          f"Test: {len(flow_test)} pasos")
    print(f"Horizonte: {horizon} | Historial: {history_len}")

    # ── 3. Construir ventanas de test ─────────────────────────────────────────
    num_test_windows = len(flow_test) - horizon + 1
    print(f"Ventanas de test: {num_test_windows}")

    # ── 4. Precalcular estructuras para baselines estacionales ───────────────

    # Period Mean: media por posición dentro del período [period, num_nodes]
    num_nodes   = flow.shape[1]
    period_mean = np.zeros((PERIOD, num_nodes), dtype=np.float32)
    for p in range(PERIOD):
        idxs = [t for t in range(n_train) if t % PERIOD == p]
        if idxs:
            period_mean[p] = flow_train[idxs].mean(axis=0)

    # ── 5. Evaluar los cuatro baselines ──────────────────────────────────────
    historical_mean = flow_train.mean(axis=0)   # [num_nodes]

    results = {
        "Historical Mean" : {"mae": [], "rmse": [], "mape": []},
        "Last Value"       : {"mae": [], "rmse": [], "mape": []},
        "Seasonal Naive"   : {"mae": [], "rmse": [], "mape": []},
        "Period Mean"      : {"mae": [], "rmse": [], "mape": []},
    }

    skipped_seasonal = 0

    for i in range(num_test_windows):
        t = test_start + i
        Y = flow[t:t + horizon]                     # [horizon, num_nodes]

        # ── Historical Mean ───────────────────────────────────────────────
        pred_hm = np.tile(historical_mean, (horizon, 1))
        results["Historical Mean"]["mae"].append(masked_mae(pred_hm, Y))
        results["Historical Mean"]["rmse"].append(masked_rmse(pred_hm, Y))
        results["Historical Mean"]["mape"].append(masked_mape(pred_hm, Y))

        # ── Last Value ────────────────────────────────────────────────────
        last     = flow[t - 1]
        pred_lv  = np.tile(last, (horizon, 1))
        results["Last Value"]["mae"].append(masked_mae(pred_lv, Y))
        results["Last Value"]["rmse"].append(masked_rmse(pred_lv, Y))
        results["Last Value"]["mape"].append(masked_mape(pred_lv, Y))

        # ── Seasonal Naive (mismo slot semana anterior) ───────────────────
        lag = t - WEEKLY_LAG
        if lag >= 0 and lag + horizon <= T:
            pred_sn = flow[lag:lag + horizon]       # [horizon, num_nodes]
            results["Seasonal Naive"]["mae"].append(masked_mae(pred_sn, Y))
            results["Seasonal Naive"]["rmse"].append(masked_rmse(pred_sn, Y))
            results["Seasonal Naive"]["mape"].append(masked_mape(pred_sn, Y))
        else:
            skipped_seasonal += 1

        # ── Period Mean (media por hora/día del período) ──────────────────
        pred_pm = np.stack([
            period_mean[(t + step) % PERIOD] for step in range(horizon)
        ])                                          # [horizon, num_nodes]
        results["Period Mean"]["mae"].append(masked_mae(pred_pm, Y))
        results["Period Mean"]["rmse"].append(masked_rmse(pred_pm, Y))
        results["Period Mean"]["mape"].append(masked_mape(pred_pm, Y))

    if skipped_seasonal > 0:
        print(f"  (Seasonal Naive: {skipped_seasonal} ventanas saltadas por "
              f"falta de historial)")

    # ── 6. Resultados ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"RESULTADOS BASELINES — {DATA_CONFIG['config_name']}")
    print("=" * 65)
    print(f"{'Baseline':<20} {'MAE':>8} {'RMSE':>8} {'MAPE':>9}")
    print("-" * 65)

    for name, metrics in results.items():
        if not metrics["mae"]:
            print(f"  {name:<18}  — sin datos suficientes —")
            continue
        print(
            f"  {name:<18}  "
            f"{np.mean(metrics['mae']):>8.4f}  "
            f"{np.mean(metrics['rmse']):>8.4f}  "
            f"{np.mean(metrics['mape']):>8.2f}%"
        )

    print("=" * 65)
    print(f"\nNota: Period Mean agrupa por {PERIOD_LABEL} "
          f"(período = {PERIOD} pasos).")
    print(f"      Seasonal Naive predice con el valor de {WEEKLY_LAG} "
          f"pasos atrás ({7} semanas × período).")


if __name__ == "__main__":
    main()