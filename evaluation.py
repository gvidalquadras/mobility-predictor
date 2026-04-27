"""
analysis/taptap/evaluation.py
==============================
Evaluación comparativa de todos los modelos sobre el conjunto de test.

Evalúa con 7 métricas (MAE, RMSE, MAPE, DA, SMAPE, PeakF1, Spearman) y
genera 3 figuras comparativas:

    Figura 1 — Barras comparativas por métrica AdTech
    Figura 2 — Serie temporal: predicción vs real para un POI representativo
    Figura 3 — Spearman ρ por categoría de POI (si el CSV de POIs está disponible)

Modelos evaluados (se saltan automáticamente si no existe el .pt):
    · Historical Mean, Last Value, Seasonal Naive, Period Mean  (estadísticos)
    · LSTM                                                       (baselines_lstm.py)
    · GWN-only                                                   (train.py USE_STAR=False)
    · STAR + GWN                                                 (train.py USE_STAR=True)

Uso:
    python analysis/taptap/evaluation.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_CONFIG, DATA, STAR, GWN, TRAIN
from model.full_model import FullModel, load_supports

# Importar LSTMBaseline del script de entrenamiento
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from baselines_lstm import LSTMBaseline

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────
DEVICE    = torch.device(TRAIN["device"])
HORIZON   = DATA["horizon"]
HIST_LEN  = DATA["history_len"]
N_NODES   = DATA["num_nodes"]

OUTPUT_DIR = Path(os.path.dirname(__file__)) / "figures_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = DATA_CONFIG["output_dir"]
POIS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "data", "raw", "taptap", "source_poibrandsesp_tags_202604170958.csv"
)

# Colores por modelo (para figuras consistentes)
MODEL_COLORS = {
    "Historical Mean" : "#aaaaaa",
    "Last Value"      : "#cccccc",
    "Seasonal Naive"  : "#888888",
    "Period Mean"     : "#bbbbbb",
    "LSTM"            : "#4e9af1",
    "GWN-only"        : "#f0a500",
    "STAR + GWN"      : "#e05c5c",
}

_IS_MADRID   = DATA_CONFIG["city"] == "Madrid"
PERIOD       = 24 if _IS_MADRID else 7
WEEKLY_LAG   = 7 * PERIOD


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def masked_mae(pred, target):
    mask = target != 0
    if mask.sum() == 0:
        return np.nan
    return np.abs(pred[mask] - target[mask]).mean()

def masked_rmse(pred, target):
    mask = target != 0
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(((pred[mask] - target[mask]) ** 2).mean())

def masked_mape(pred, target):
    mask = target != 0
    if mask.sum() == 0:
        return np.nan
    return (np.abs((pred[mask] - target[mask]) / target[mask])).mean() * 100

def directional_accuracy(pred, target, prev):
    """pred, target: [H, N] | prev: [N]"""
    full_t = np.vstack([prev[None], target])
    full_p = np.vstack([prev[None], pred])
    dt = np.sign(full_t[1:] - full_t[:-1])
    dp = np.sign(full_p[1:] - full_p[:-1])
    valid = dt != 0
    return (dt[valid] == dp[valid]).mean() * 100 if valid.sum() > 0 else np.nan

def smape(pred, target, eps=1.0):
    return (2 * np.abs(pred - target) / (np.abs(pred) + np.abs(target) + eps)).mean() * 100

def peak_metrics(pred, target, thresholds):
    """thresholds: [N] umbral p80 por POI"""
    rp = target > thresholds[None]
    pp = pred   > thresholds[None]
    tp = (rp & pp).sum()
    fp = (~rp & pp).sum()
    fn = (rp & ~pp).sum()
    prec = tp / (tp + fp + 1e-10)
    rec  = tp / (tp + fn + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    return float(prec * 100), float(rec * 100), float(f1 * 100)

def mean_spearman(pred, target):
    rhos = []
    for h in range(pred.shape[0]):
        if np.std(pred[h]) < 1e-8 or np.std(target[h]) < 1e-8:
            continue
        rho, _ = spearmanr(pred[h], target[h])
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else np.nan

def compute_all_metrics(preds_list, targets_list, prevs_list, thresholds):
    """Agrega métricas sobre todas las ventanas de test."""
    m = {"mae": [], "rmse": [], "mape": [], "da": [],
         "smape": [], "p_f1": [], "spearman": []}
    for pred, target, prev in zip(preds_list, targets_list, prevs_list):
        m["mae"].append(masked_mae(pred, target))
        m["rmse"].append(masked_rmse(pred, target))
        m["mape"].append(masked_mape(pred, target))
        m["da"].append(directional_accuracy(pred, target, prev))
        m["smape"].append(smape(pred, target))
        _, _, f1 = peak_metrics(pred, target, thresholds)
        m["p_f1"].append(f1)
        m["spearman"].append(mean_spearman(pred, target))
    return {k: np.nanmean(v) for k, v in m.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Carga de modelos neuronales
# ─────────────────────────────────────────────────────────────────────────────

def load_full_model(use_star, horizon):
    suffix = "" if use_star else "_gwn_only"
    path   = os.path.join(CHECKPOINT_DIR, f"best_model_h{horizon}{suffix}.pt")
    if not os.path.exists(path):
        return None, path
    supports = load_supports(DATA_CONFIG["graphs_dir"], device=str(DEVICE))
    model = FullModel(
        propath           = DATA_CONFIG["graphs_dir"],
        supports          = supports,
        device            = str(DEVICE),
        use_star          = use_star,
        fea_dim           = STAR["fea_dim"],
        hid_dim           = STAR["hid_dim"],
        out_dim_emb       = STAR["out_dim"],
        layer_num         = STAR["layer_num"],
        head_num          = STAR["head_num"],
        out_dim_pred      = horizon,
        history_len       = HIST_LEN,
        residual_channels = GWN["residual_channels"],
        dilation_channels = GWN["dilation_channels"],
        skip_channels     = GWN["skip_channels"],
        end_channels      = GWN["end_channels"],
        dropout           = GWN["dropout"],
        blocks            = GWN["blocks"],
        layers            = GWN["layers"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model, path

def load_lstm_model(horizon, num_nodes):
    path = os.path.join(CHECKPOINT_DIR, "lstm_best.pt")
    if not os.path.exists(path):
        return None, path
    model = LSTMBaseline(
        num_nodes  = num_nodes,
        hidden_dim = 256,
        horizon    = horizon,
        num_layers = 2,
        dropout    = 0.3,
    ).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model, path

def predict_model(model, flow_norm, t, mean, std):
    """Devuelve predicción [H, N] desnormalizada."""
    X = torch.tensor(
        flow_norm[t - HIST_LEN:t].T[None, :, None, :],
        dtype=torch.float32, device=DEVICE
    )
    with torch.no_grad():
        pred_norm = model(X)[0].cpu().numpy()   # [N, H]
    return (pred_norm.T * (std + 1e-8) + mean)  # [H, N]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"{'='*65}")
    print(f"EVALUACIÓN COMPARATIVA — {DATA_CONFIG['config_name']}")
    print(f"{'='*65}\n")

    # ── 1. Datos ──────────────────────────────────────────────────────────────
    flow  = np.load(DATA_CONFIG["flow_path"])
    dates = np.load(DATA_CONFIG["dates_path"])
    T     = len(flow)
    n_train = int(T * DATA["train_ratio"])
    n_val   = int(T * DATA["val_ratio"])
    test_start = n_train + n_val
    num_nodes  = flow.shape[1]

    mean = flow[:n_train].mean()
    std  = flow[:n_train].std()
    flow_norm = (flow - mean) / (std + 1e-8)

    num_windows   = T - HORIZON - test_start + 1
    peak_thresh   = np.percentile(flow[:n_train], 80, axis=0)
    period_mean   = np.zeros((PERIOD, num_nodes), dtype=np.float32)
    for p in range(PERIOD):
        idx = [t for t in range(n_train) if t % PERIOD == p]
        if idx:
            period_mean[p] = flow[:n_train][idx].mean(axis=0)
    hist_mean = flow[:n_train].mean(axis=0)

    print(f"Datos: {flow.shape}  ({dates[0]} → {dates[-1]})")
    print(f"Test: {num_windows} ventanas | horizonte: {HORIZON}")

    # ── 2. Cargar modelos neuronales ──────────────────────────────────────────
    nn_models = {}
    for name, use_star in [("STAR + GWN", True), ("GWN-only", False)]:
        m, path = load_full_model(use_star, HORIZON)
        if m is None:
            print(f"  [!] {name}: no encontrado ({path}) — se omite")
        else:
            nn_models[name] = m
            print(f"  ✓ {name} cargado desde {os.path.basename(path)}")

    lstm_model, lstm_path = load_lstm_model(HORIZON, num_nodes)
    if lstm_model is None:
        print(f"  [!] LSTM: no encontrado ({lstm_path}) — se omite")
    else:
        nn_models["LSTM"] = lstm_model
        print(f"  ✓ LSTM cargado")

    # ── 3. Calcular predicciones y métricas por ventana ───────────────────────
    print("\nCalculando predicciones...")

    all_targets, all_prevs = [], []
    all_preds = {n: [] for n in
                 ["Historical Mean", "Last Value", "Seasonal Naive",
                  "Period Mean"] + list(nn_models.keys())}

    for i in range(num_windows):
        t      = test_start + i
        target = flow[t:t + HORIZON]       # [H, N]
        prev   = flow[t - 1]               # [N]
        all_targets.append(target)
        all_prevs.append(prev)

        # Estadísticos
        all_preds["Historical Mean"].append(np.tile(hist_mean, (HORIZON, 1)))
        all_preds["Last Value"].append(np.tile(prev, (HORIZON, 1)))

        lag = t - WEEKLY_LAG
        if lag >= 0 and lag + HORIZON <= T:
            all_preds["Seasonal Naive"].append(flow[lag:lag + HORIZON])
        else:
            all_preds["Seasonal Naive"].append(None)

        all_preds["Period Mean"].append(np.stack(
            [period_mean[(t + s) % PERIOD] for s in range(HORIZON)]
        ))

        # Neuronales
        for name, model in nn_models.items():
            all_preds[name].append(
                predict_model(model, flow_norm, t, mean, std)
            )

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{num_windows}", end="\r")

    print(f"  {num_windows}/{num_windows} ventanas procesadas")

    # ── 4. Agregar métricas ───────────────────────────────────────────────────
    results = {}
    for name in list(all_preds.keys()):
        preds   = [p for p in all_preds[name] if p is not None]
        targets = [all_targets[i]
                   for i, p in enumerate(all_preds[name]) if p is not None]
        prevs   = [all_prevs[i]
                   for i, p in enumerate(all_preds[name]) if p is not None]
        if not preds:
            continue
        results[name] = compute_all_metrics(preds, targets, prevs, peak_thresh)

    # ── 5. Tabla de resultados ────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"{'Modelo':<20} {'MAE':>7} {'RMSE':>7} {'MAPE':>8}  "
          f"{'DA%':>6} {'SMAPE':>7} {'PeakF1':>7} {'Spear':>7}")
    print(f"{'-'*95}")
    for name, m in results.items():
        print(
            f"  {name:<18}  "
            f"{m['mae']:>7.4f} "
            f"{m['rmse']:>7.4f} "
            f"{m['mape']:>7.2f}%  "
            f"{m['da']:>5.1f}% "
            f"{m['smape']:>6.1f}% "
            f"{m['p_f1']:>6.1f}% "
            f"{m['spearman']:>7.3f}"
        )
    print(f"{'='*95}")

    # ── 6. Figura 1: Barras comparativas por métrica ──────────────────────────
    print("\nGenerando figuras...")

    adtech_metrics = {
        "Directional Accuracy (%)": "da",
        "Spearman ρ":               "spearman",
        "Peak Detection F1 (%)":    "p_f1",
        "SMAPE (%)":                "smape",
    }

    model_names = list(results.keys())
    n_models    = len(model_names)
    colors      = [MODEL_COLORS.get(n, "#999999") for n in model_names]

    fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig1.suptitle("Comparativa de modelos — métricas AdTech", fontsize=14, fontweight="bold")

    for ax, (title, key) in zip(axes.flat, adtech_metrics.items()):
        vals = [results[n][key] for n in model_names]
        bars = ax.barh(model_names, vals, color=colors, edgecolor="white", height=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(title.split("(")[0].strip())
        ax.axvline(0, color="black", linewidth=0.5)
        # Etiquetas en las barras
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path_fig1 = OUTPUT_DIR / "fig1_comparativa_adtech.png"
    fig1.savefig(path_fig1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  ✓ Figura 1 guardada: {path_fig1}")

    # Figura adicional: MAE y RMSE
    fig1b, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig1b.suptitle("Comparativa de modelos — métricas estándar", fontsize=14, fontweight="bold")
    for ax, key, title in [(ax1, "mae", "MAE"), (ax2, "rmse", "RMSE")]:
        vals = [results[n][key] for n in model_names]
        bars = ax.barh(model_names, vals, color=colors, edgecolor="white", height=0.6)
        ax.set_title(title, fontsize=11)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path_fig1b = OUTPUT_DIR / "fig1b_comparativa_mae_rmse.png"
    fig1b.savefig(path_fig1b, dpi=150, bbox_inches="tight")
    plt.close(fig1b)
    print(f"  ✓ Figura 1b guardada: {path_fig1b}")

    # ── 7. Figura 2: Serie temporal para POI representativo ───────────────────
    # Elegir POI con mayor flujo medio en test
    poi_idx = int(np.argmax(flow[test_start:].mean(axis=0)))
    poi_name = f"POI #{poi_idx}"

    # Intentar obtener nombre real del POI
    poi_to_idx_path = os.path.join(
        os.path.dirname(DATA_CONFIG["flow_path"]), "poi_to_idx.npy"
    )
    if os.path.exists(poi_to_idx_path) and os.path.exists(POIS_CSV):
        try:
            import pandas as pd
            poi_map  = np.load(poi_to_idx_path, allow_pickle=True)
            pois_df  = pd.read_csv(POIS_CSV)
            idx_to_id = {int(v): str(k) for k, v in poi_map}
            if poi_idx in idx_to_id:
                pid  = idx_to_id[poi_idx]
                row  = pois_df[pois_df["id"].astype(str) == pid]
                if not row.empty:
                    poi_name = row.iloc[0]["name"][:40]
        except Exception:
            pass

    # Ventana de visualización: 7 días (168h) desde mitad del test
    vis_start = test_start + num_windows // 2
    vis_end   = min(vis_start + HORIZON, T)
    actual    = flow[vis_start:vis_end, poi_idx]
    t_axis    = np.arange(len(actual))

    # Modelos a mostrar en la serie temporal
    viz_models = ["Seasonal Naive", "Period Mean"]
    if "GWN-only" in nn_models:
        viz_models.append("GWN-only")
    if "STAR + GWN" in nn_models:
        viz_models.append("STAR + GWN")

    fig2, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_axis, actual, color="black", linewidth=2, label="Real", zorder=5)

    for name in viz_models:
        preds_window = all_preds[name][num_windows // 2]
        if preds_window is None:
            continue
        pred_poi = preds_window[:len(actual), poi_idx]
        ax.plot(
            t_axis, pred_poi,
            color=MODEL_COLORS.get(name, "#999999"),
            linewidth=1.5, linestyle="--", label=name, alpha=0.85
        )

    # Marcar picos reales
    is_peak = actual > peak_thresh[poi_idx]
    ax.fill_between(t_axis, 0, actual.max() * 1.05,
                    where=is_peak, alpha=0.08, color="red", label="Horas pico")

    ax.set_title(f"Predicción vs Real — {poi_name}", fontsize=12)
    ax.set_xlabel("Hora (horizonte de predicción)")
    ax.set_ylabel("Visitas")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path_fig2 = OUTPUT_DIR / "fig2_serie_temporal.png"
    fig2.savefig(path_fig2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Figura 2 guardada: {path_fig2}  (POI: {poi_name})")

    # ── 8. Figura 3: Spearman por categoría de POI ───────────────────────────
    if os.path.exists(POIS_CSV) and os.path.exists(poi_to_idx_path):
        try:
            import pandas as pd
            poi_map  = np.load(poi_to_idx_path, allow_pickle=True)
            pois_df  = pd.read_csv(POIS_CSV)
            idx_to_id = {int(v): str(k) for k, v in poi_map}

            id_col   = pois_df["id"].astype(str)
            cat_map  = dict(zip(id_col, pois_df["tier1_category"]))
            poi_cats = np.array([
                cat_map.get(idx_to_id.get(i, ""), "Unknown")
                for i in range(num_nodes)
            ])

            # Top categorías por número de POIs
            from collections import Counter
            top_cats = [c for c, _ in Counter(poi_cats).most_common(8)]

            # Modelos a comparar en el desglose por categoría
            cat_models = [n for n in ["Seasonal Naive", "STAR + GWN"]
                          if n in results]

            if len(cat_models) >= 1:
                fig3, ax = plt.subplots(figsize=(13, 5))
                x      = np.arange(len(top_cats))
                width  = 0.8 / len(cat_models)

                for k, name in enumerate(cat_models):
                    cat_rhos = []
                    for cat in top_cats:
                        cat_idx = np.where(poi_cats == cat)[0]
                        rhos = []
                        for pred_w, tgt_w in zip(all_preds[name], all_targets):
                            if pred_w is None:
                                continue
                            for h in range(HORIZON):
                                p = pred_w[h, cat_idx]
                                t_ = tgt_w[h, cat_idx]
                                if np.std(p) > 1e-8 and np.std(t_) > 1e-8:
                                    rho, _ = spearmanr(p, t_)
                                    if not np.isnan(rho):
                                        rhos.append(rho)
                        cat_rhos.append(np.mean(rhos) if rhos else np.nan)

                    ax.bar(
                        x + k * width, cat_rhos, width,
                        label=name,
                        color=MODEL_COLORS.get(name, "#999999"),
                        edgecolor="white"
                    )

                ax.set_xticks(x + width * (len(cat_models) - 1) / 2)
                ax.set_xticklabels(top_cats, rotation=25, ha="right", fontsize=9)
                ax.set_ylabel("Spearman ρ")
                ax.set_title("Correlación de Spearman por categoría de POI", fontsize=12)
                ax.legend()
                ax.set_ylim(0, 1)
                ax.axhline(0, color="black", linewidth=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()

                path_fig3 = OUTPUT_DIR / "fig3_spearman_categoria.png"
                fig3.savefig(path_fig3, dpi=150, bbox_inches="tight")
                plt.close(fig3)
                print(f"  ✓ Figura 3 guardada: {path_fig3}")
        except Exception as e:
            print(f"  [!] Figura 3 omitida: {e}")
    else:
        print(f"  [!] Figura 3 omitida: CSV de POIs no encontrado")

    print(f"\n✓ Evaluación completada. Figuras en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()