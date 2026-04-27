"""
analysis_metricas_adtech.py
===========================
Evaluación del modelo ST-GNN con métricas orientadas al negocio AdTech/DOOH.

Métricas implementadas:
    1. Directional Accuracy  — % de pasos donde el modelo predice correctamente
                               si el flujo sube o baja respecto a la hora anterior
    2. Peak Detection        — Precision, Recall y F1 sobre horas pico por POI
    3. Spearman ρ            — Correlación de rangos entre predicción y real
    4. SMAPE                 — Error porcentual simétrico (robusto a valores bajos)

Todas las métricas se calculan sobre el conjunto de test y se desglosan por:
    - Categoría de POI
    - Hora del día (dayparting)
    - Horizonte de predicción (h=1..72)

Uso:
    Asegúrate de que config.py tiene CONFIG_NAME = "config_Madrid_3225" y ejecuta:
    python analysis/taptap/analysis_metricas_adtech.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)

from config import DATA_CONFIG, DATA, STAR, GWN, TRAIN
from model.full_model import FullModel, load_supports
from torch.utils.data import Dataset, DataLoader

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs", "metricas_adtech")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

CONFIG_NAME = DATA_CONFIG["config_name"]
print(f"Configuración: {CONFIG_NAME}")

# =============================================================================
# 1. CARGAR DATOS Y MODELO
# =============================================================================

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
        X = self.flow[t - self.history_len:t].T.unsqueeze(1)   # [N, 1, history_len]
        Y = self.flow[t:t + self.horizon].T                     # [N, horizon]
        # Último valor conocido antes de la predicción (para DA en h=0)
        last = self.flow[t - 1].unsqueeze(-1)                   # [N, 1]
        return X, Y, last

print("\nCargando datos de flujo...")
flow  = np.load(DATA_CONFIG["flow_path"])   # [T, N]
dates = np.load(DATA_CONFIG["dates_path"])
T, N  = flow.shape
print(f"  Shape: {flow.shape}")

history_len = DATA["history_len"]
horizon     = DATA["horizon"]
n_train     = int(T * DATA["train_ratio"])
n_val       = int(T * DATA["val_ratio"])

# Normalización (igual que en train.py)
mean = flow[:n_train].mean()
std  = flow[:n_train].std()
flow_norm = (flow - mean) / (std + 1e-8)

# Split test
test_t_start = n_train + n_val - horizon + 1
test_t_end   = T - horizon

test_dataset = FlowDataset(flow_norm, history_len, horizon, test_t_start, test_t_end)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)
print(f"  Muestras test: {len(test_dataset)}")

# Cargar modelo
device   = torch.device(TRAIN["device"])
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

best_model_path = os.path.join(DATA_CONFIG["output_dir"], f"best_model_h{horizon}.pt")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print(f"Modelo cargado: {best_model_path}")

# Generar predicciones sobre test completo
print("\nGenerando predicciones sobre test...")
all_pred, all_real, all_last = [], [], []

with torch.no_grad():
    for X, Y, last in test_loader:
        X = X.to(device)
        pred = model(X)                                          # [B, N, H]
        pred_real = pred * (std + 1e-8) + mean
        Y_real    = Y    * (std + 1e-8) + mean
        last_real = last * (std + 1e-8) + mean
        all_pred.append(pred_real.cpu().numpy())
        all_real.append(Y_real.cpu().numpy())
        all_last.append(last_real.cpu().numpy())

pred_arr = np.concatenate(all_pred, axis=0)   # [n_test, N, H]
real_arr = np.concatenate(all_real, axis=0)   # [n_test, N, H]
last_arr = np.concatenate(all_last, axis=0)   # [n_test, N, 1]

print(f"  Predicciones shape: {pred_arr.shape}")

# =============================================================================
# 2. CARGAR METADATOS DE CATEGORÍAS
# =============================================================================

poi_to_idx_path = os.path.join(PROJECT_DIR, "data", "processed", "flows", "Madrid", "poi_to_idx.npy")
poi_to_idx_arr  = np.load(poi_to_idx_path, allow_pickle=True)
idx_to_poi      = {int(row[1]): str(row[0]) for row in poi_to_idx_arr}

pois_csv = os.path.join(PROJECT_DIR, "data", "raw", "taptap",
                        "source_poibrandsesp_tags_202604170958.csv")
pois_df  = pd.read_csv(pois_csv)
pois_df["id"] = pois_df["id"].astype(str)
poi_to_cat = dict(zip(pois_df["id"], pois_df["tier1_category"]))

categories = np.array([
    poi_to_cat.get(idx_to_poi.get(i, ""), "Unknown")
    for i in range(N)
])
unique_cats = pd.Series(categories).value_counts().index.tolist()
print(f"\nCategorías: {unique_cats}")

# =============================================================================
# 3. CÁLCULO DE MÉTRICAS
# =============================================================================

def directional_accuracy(pred, real, last):
    """
    DA por paso de horizonte.

    pred, real : [n_test, N, H]
    last       : [n_test, N, 1]  último valor conocido antes de la predicción

    Para h=0: compara pred[:,n,0] vs last[:,n,0]
    Para h>0: compara pred[:,n,h] vs pred[:,n,h-1]  (dirección intra-predicción)
    """
    # Construir serie completa con last como paso -1
    pred_full = np.concatenate([last, pred], axis=2)  # [n, N, H+1]
    real_full = np.concatenate([last, real], axis=2)  # [n, N, H+1]

    pred_diff = np.sign(pred_full[:, :, 1:] - pred_full[:, :, :-1])  # [n, N, H]
    real_diff = np.sign(real_full[:, :, 1:] - real_full[:, :, :-1])  # [n, N, H]

    # Ignorar pasos donde real no cambia (signo 0)
    valid = real_diff != 0
    correct = (pred_diff == real_diff) & valid

    da_total = correct.sum() / valid.sum() if valid.sum() > 0 else 0.0
    da_by_h  = np.array([
        correct[:, :, h].sum() / valid[:, :, h].sum()
        if valid[:, :, h].sum() > 0 else 0.0
        for h in range(pred.shape[2])
    ])
    return da_total, da_by_h, correct, valid


def peak_detection(pred, real, percentile=80):
    """
    Para cada POI, define pico como horas donde real > percentil P.
    Calcula Precision, Recall y F1 globales y por POI.
    """
    n, num_nodes, H = real.shape
    real_flat = real.reshape(-1, num_nodes)    # [n*H, N]
    pred_flat = pred.reshape(-1, num_nodes)

    # Umbral por POI (percentil sobre todo el test)
    thresholds = np.percentile(real_flat, percentile, axis=0)  # [N]

    real_peak = real_flat > thresholds   # [n*H, N]
    pred_peak = pred_flat > thresholds

    tp = (real_peak & pred_peak).sum(axis=0).astype(float)  # [N]
    fp = (~real_peak & pred_peak).sum(axis=0).astype(float)
    fn = (real_peak & ~pred_peak).sum(axis=0).astype(float)

    precision = tp / np.clip(tp + fp, 1, None)
    recall    = tp / np.clip(tp + fn, 1, None)
    f1        = 2 * precision * recall / np.clip(precision + recall, 1e-8, None)

    return precision, recall, f1


def spearman_by_timestep(pred, real):
    """
    Spearman ρ entre ranking de POIs en cada timestep.
    Promedio sobre todos los timesteps del test.
    """
    n, num_nodes, H = pred.shape
    rhos = []
    for t in range(n):
        for h in range(H):
            rho, _ = spearmanr(pred[t, :, h], real[t, :, h])
            if not np.isnan(rho):
                rhos.append(rho)
    return np.array(rhos)


def smape(pred, real):
    """SMAPE: 2*|pred-real| / (|pred| + |real| + ε)  × 100"""
    denom = np.abs(pred) + np.abs(real) + 1e-8
    return np.mean(2 * np.abs(pred - real) / denom) * 100


# ── Calcular todo ──────────────────────────────────────────────────────────

print("\nCalculando métricas...")

da_total, da_by_h, da_correct, da_valid = directional_accuracy(pred_arr, real_arr, last_arr)
precision, recall, f1 = peak_detection(pred_arr, real_arr, percentile=80)
spearman_rhos = spearman_by_timestep(pred_arr, real_arr)
smape_total   = smape(pred_arr, real_arr)

print(f"\n{'='*55}")
print(f"  MÉTRICAS ADTECH — {CONFIG_NAME}")
print(f"{'='*55}")
print(f"  Directional Accuracy  : {da_total*100:.1f}%")
print(f"  Peak Detection Recall : {recall.mean()*100:.1f}%")
print(f"  Peak Detection Prec.  : {precision.mean()*100:.1f}%")
print(f"  Peak Detection F1     : {f1.mean()*100:.1f}%")
print(f"  Spearman ρ (medio)    : {spearman_rhos.mean():.4f}")
print(f"  SMAPE                 : {smape_total:.2f}%")
print(f"{'='*55}")

# ── Métricas por categoría ─────────────────────────────────────────────────
print(f"\n{'Categoría':<22} {'DA':>8} {'P.Recall':>9} {'P.Prec':>8} {'F1':>8} {'Spearman':>10} {'SMAPE':>8}")
print("-" * 77)

cat_metrics = {}
for cat in unique_cats:
    mask = categories == cat
    da_cat = (da_correct[:, mask, :].sum() / da_valid[:, mask, :].sum()
              if da_valid[:, mask, :].sum() > 0 else 0.0)
    recall_cat    = recall[mask].mean()
    precision_cat = precision[mask].mean()
    f1_cat        = f1[mask].mean()
    rhos_cat = []
    for t in range(pred_arr.shape[0]):
        for h in range(horizon):
            rho, _ = spearmanr(pred_arr[t, mask, h], real_arr[t, mask, h])
            if not np.isnan(rho):
                rhos_cat.append(rho)
    spearman_cat = np.mean(rhos_cat) if rhos_cat else 0.0
    smape_cat = smape(pred_arr[:, mask, :], real_arr[:, mask, :])

    cat_metrics[cat] = dict(da=da_cat, recall=recall_cat, precision=precision_cat,
                             f1=f1_cat, spearman=spearman_cat, smape=smape_cat)
    print(f"  {cat:<20} {da_cat*100:>7.1f}% {recall_cat*100:>8.1f}% "
          f"{precision_cat*100:>7.1f}% {f1_cat*100:>7.1f}% "
          f"{spearman_cat:>9.4f} {smape_cat:>7.1f}%")

# =============================================================================
# 4. VISUALIZACIONES
# =============================================================================

PALETTE = ['#2E75B6', '#1D9E75', '#D85A30', '#8064A2', '#C0504D']
cat_colors = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(unique_cats)}

# ── Fig 1: Directional Accuracy por horizonte y por categoría ────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Directional Accuracy — {CONFIG_NAME}', fontsize=13, fontweight='bold')

axes[0].plot(range(1, horizon + 1), da_by_h * 100, color='#2E75B6', lw=2)
axes[0].axhline(50, color='gray', ls='--', lw=1, label='Azar (50%)')
axes[0].axhline(da_total * 100, color='#D85A30', ls='--', lw=1.5,
                label=f'Media: {da_total*100:.1f}%')
axes[0].set_title('DA por horizonte de predicción')
axes[0].set_xlabel('Horizonte h (horas)')
axes[0].set_ylabel('Directional Accuracy (%)')
axes[0].set_ylim(40, 100)
axes[0].legend()
axes[0].grid(alpha=0.3)

cats_sorted = sorted(cat_metrics.keys(), key=lambda c: cat_metrics[c]['da'], reverse=True)
da_vals = [cat_metrics[c]['da'] * 100 for c in cats_sorted]
bars = axes[1].barh(cats_sorted, da_vals,
                    color=[cat_colors[c] for c in cats_sorted], edgecolor='white')
axes[1].axvline(50, color='gray', ls='--', lw=1, label='Azar')
axes[1].axvline(da_total * 100, color='#D85A30', ls='--', lw=1.5,
                label=f'Global: {da_total*100:.1f}%')
for bar, v in zip(bars, da_vals):
    axes[1].text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{v:.1f}%', va='center', fontsize=10)
axes[1].set_title('DA por categoría')
axes[1].set_xlabel('Directional Accuracy (%)')
axes[1].set_xlim(40, 100)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/da_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nGuardado: da_{CONFIG_NAME}.png")

# ── Fig 2: Peak Detection — Precision / Recall / F1 por categoría ────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f'Peak Detection (top 20% horas) — {CONFIG_NAME}', fontsize=13, fontweight='bold')

metrics_labels = ['precision', 'recall', 'f1']
metrics_titles = ['Precision', 'Recall', 'F1']
colors_pd = ['#2E75B6', '#1D9E75', '#D85A30']

for ax, met, title, color in zip(axes, metrics_labels, metrics_titles, colors_pd):
    vals = [cat_metrics[c][met] * 100 for c in cats_sorted]
    global_val = np.mean([cat_metrics[c][met] for c in unique_cats]) * 100
    bars = ax.barh(cats_sorted, vals, color=color, alpha=0.85, edgecolor='white')
    ax.axvline(global_val, color='black', ls='--', lw=1.5,
               label=f'Global: {global_val:.1f}%')
    for bar, v in zip(bars, vals):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{v:.1f}%', va='center', fontsize=9)
    ax.set_title(title)
    ax.set_xlabel('(%)')
    ax.set_xlim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/peak_detection_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: peak_detection_{CONFIG_NAME}.png")

# ── Fig 3: Spearman ρ — distribución y por categoría ─────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Spearman ρ (correlación de rankings) — {CONFIG_NAME}',
             fontsize=13, fontweight='bold')

axes[0].hist(spearman_rhos, bins=40, color='#2E75B6', alpha=0.8, edgecolor='white')
axes[0].axvline(spearman_rhos.mean(), color='#D85A30', lw=2,
                label=f'Media: {spearman_rhos.mean():.3f}')
axes[0].set_title('Distribución de Spearman ρ\n(por timestep × horizonte)')
axes[0].set_xlabel('Spearman ρ')
axes[0].set_ylabel('Frecuencia')
axes[0].legend()
axes[0].grid(alpha=0.3)

rho_vals = [cat_metrics[c]['spearman'] for c in cats_sorted]
bars = axes[1].barh(cats_sorted, rho_vals,
                    color=[cat_colors[c] for c in cats_sorted], edgecolor='white')
axes[1].axvline(spearman_rhos.mean(), color='#D85A30', ls='--', lw=1.5,
                label=f'Global: {spearman_rhos.mean():.3f}')
for bar, v in zip(bars, rho_vals):
    axes[1].text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{v:.3f}', va='center', fontsize=10)
axes[1].set_title('Spearman ρ por categoría')
axes[1].set_xlabel('Spearman ρ')
axes[1].set_xlim(0, 1)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/spearman_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: spearman_{CONFIG_NAME}.png")

# ── Fig 4: DA por franja horaria (dayparting) ─────────────────────────────
# Los timestamps del test cubren las últimas horas del dataset

test_hours_start = test_t_start  # primer timestep del test
da_by_daypart = np.zeros(24)
count_by_daypart = np.zeros(24)

n_test = pred_arr.shape[0]
for sample_idx in range(n_test):
    for h in range(horizon):
        abs_t = test_t_start + sample_idx + h
        hour_of_day = abs_t % 24
        valid_h = da_valid[sample_idx, :, h].sum()
        correct_h = da_correct[sample_idx, :, h].sum()
        if valid_h > 0:
            da_by_daypart[hour_of_day] += correct_h / valid_h
            count_by_daypart[hour_of_day] += 1

da_by_daypart_mean = np.where(
    count_by_daypart > 0,
    da_by_daypart / count_by_daypart,
    np.nan
)

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(f'Directional Accuracy por hora del día — {CONFIG_NAME}',
             fontsize=13, fontweight='bold')

hours = np.arange(24)
colors_hour = ['#1a3a5c' if (h >= 9 and h <= 22) else '#aab8c7' for h in hours]
bars = ax.bar(hours, da_by_daypart_mean * 100, color=colors_hour, edgecolor='white', width=0.8)
ax.axhline(50, color='gray', ls='--', lw=1, label='Azar (50%)')
ax.axhline(da_total * 100, color='#D85A30', ls='--', lw=1.5,
           label=f'Media global: {da_total*100:.1f}%')
ax.set_xlabel('Hora del día')
ax.set_ylabel('Directional Accuracy (%)')
ax.set_xticks(hours)
ax.set_ylim(40, 100)
ax.legend()
ax.grid(alpha=0.3, axis='y')

from matplotlib.patches import Patch
legend_extra = [Patch(facecolor='#1a3a5c', label='Franja valor DOOH (09h–22h)'),
                Patch(facecolor='#aab8c7', label='Resto')]
ax.legend(handles=ax.get_legend().legend_handles + legend_extra, fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/da_daypart_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: da_daypart_{CONFIG_NAME}.png")

# ── Fig 5: Resumen dashboard ──────────────────────────────────────────────

fig = plt.figure(figsize=(14, 8))
fig.suptitle(f'Resumen métricas AdTech — {CONFIG_NAME}', fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

# Panel 1: KPIs globales
ax0 = fig.add_subplot(gs[0, 0])
kpis = {
    'Directional\nAccuracy': da_total * 100,
    'Peak\nRecall': recall.mean() * 100,
    'Peak\nPrecision': precision.mean() * 100,
    'Peak\nF1': f1.mean() * 100,
}
kpi_colors = ['#2E75B6', '#1D9E75', '#8064A2', '#D85A30']
bars0 = ax0.bar(list(kpis.keys()), list(kpis.values()),
                color=kpi_colors, edgecolor='white')
ax0.axhline(50, color='gray', ls='--', lw=1, alpha=0.7)
for bar, v in zip(bars0, kpis.values()):
    ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax0.set_ylim(0, 110)
ax0.set_title('KPIs globales', fontsize=11)
ax0.set_ylabel('%')
ax0.grid(alpha=0.3, axis='y')

# Panel 2: Spearman ρ por categoría
ax1 = fig.add_subplot(gs[0, 1])
rho_vals2 = [cat_metrics[c]['spearman'] for c in cats_sorted]
ax1.barh(cats_sorted, rho_vals2,
         color=[cat_colors[c] for c in cats_sorted], edgecolor='white')
ax1.axvline(spearman_rhos.mean(), color='#D85A30', ls='--', lw=1.5)
ax1.set_title(f'Spearman ρ por cat.\n(global: {spearman_rhos.mean():.3f})', fontsize=11)
ax1.set_xlabel('ρ')
ax1.set_xlim(0, 1)
ax1.grid(alpha=0.3, axis='x')

# Panel 3: SMAPE por categoría
ax2 = fig.add_subplot(gs[0, 2])
smape_vals = [cat_metrics[c]['smape'] for c in cats_sorted]
ax2.barh(cats_sorted, smape_vals,
         color=[cat_colors[c] for c in cats_sorted], edgecolor='white')
ax2.axvline(smape_total, color='#D85A30', ls='--', lw=1.5)
ax2.set_title(f'SMAPE por cat.\n(global: {smape_total:.1f}%)', fontsize=11)
ax2.set_xlabel('%')
ax2.grid(alpha=0.3, axis='x')

# Panel 4: DA por horizonte
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(range(1, horizon + 1), da_by_h * 100, color='#2E75B6', lw=2)
ax3.fill_between(range(1, horizon + 1), 50, da_by_h * 100,
                 where=da_by_h * 100 > 50, alpha=0.15, color='#2E75B6',
                 label='Por encima del azar')
ax3.axhline(50, color='gray', ls='--', lw=1, label='Azar (50%)')
ax3.axhline(da_total * 100, color='#D85A30', ls='--', lw=1.5,
            label=f'Global: {da_total*100:.1f}%')
ax3.set_title('Directional Accuracy por horizonte', fontsize=11)
ax3.set_xlabel('Horizonte h (horas)')
ax3.set_ylabel('DA (%)')
ax3.set_ylim(40, 100)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Panel 5: F1 peak detection por categoría
ax4 = fig.add_subplot(gs[1, 2])
f1_vals = [cat_metrics[c]['f1'] * 100 for c in cats_sorted]
ax4.barh(cats_sorted, f1_vals,
         color=[cat_colors[c] for c in cats_sorted], edgecolor='white')
ax4.axvline(f1.mean() * 100, color='#D85A30', ls='--', lw=1.5)
ax4.set_title(f'Peak F1 por cat.\n(global: {f1.mean()*100:.1f}%)', fontsize=11)
ax4.set_xlabel('%')
ax4.set_xlim(0, 100)
ax4.grid(alpha=0.3, axis='x')

plt.savefig(f'{OUTPUT_DIR}/dashboard_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: dashboard_{CONFIG_NAME}.png")

print(f"\nTodas las figuras guardadas en: {OUTPUT_DIR}/")
print("\n=== RESUMEN PARA EL TFG ===")
print(f"  El modelo predice correctamente la dirección del flujo en el {da_total*100:.1f}% de los casos")
print(f"  Detecta el {recall.mean()*100:.1f}% de las horas pico (top 20% de actividad)")
print(f"  Correlación de rankings entre POIs: Spearman ρ = {spearman_rhos.mean():.3f}")
print(f"  Error porcentual simétrico: SMAPE = {smape_total:.1f}%")