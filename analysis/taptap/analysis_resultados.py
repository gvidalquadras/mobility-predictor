"""
analysis_resultados_madrid.py
==============================
Análisis completo de resultados del modelo ST-GNN sobre datos TapTap Madrid.

Genera:
    1. Curva de entrenamiento (loss + val MAE por época)
    2. Comparativa con baselines
    3. Error por horizonte de predicción (h=1 → h=72)
    4. Error por hora del día (dayparting)
    5. Error por día de la semana
    6. Error por categoría de POI
    7. Predicción vs real para POIs de ejemplo

Uso:
    PYTHONPATH=/home/user/work python analysis/taptap/analysis_resultados_madrid.py
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)

from config import DATA_CONFIG, DATA, STAR, GWN, TRAIN
from model.full_model import FullModel, load_supports

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs", "resultados")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

CONFIG_NAME = DATA_CONFIG["config_name"]
COLORS = {"blue": "#378ADD", "red": "#D85A30", "green": "#1D9E75", "orange": "#BA7517", "purple": "#9B59B6"}
print(f"Configuración: {CONFIG_NAME}")


# =============================================================================
# UTILIDADES (replicadas de train.py)
# =============================================================================

class FlowDataset(Dataset):
    def __init__(self, flow, history_len, horizon, t_start, t_end):
        self.flow        = torch.tensor(flow, dtype=torch.float32)
        self.history_len = history_len
        self.horizon     = horizon
        self.t_start     = t_start
        self.num_samples = t_end - t_start + 1
        self.t_indices   = list(range(t_start, t_end + 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        t = self.t_start + idx
        X = self.flow[t - self.history_len:t].T.unsqueeze(1)
        Y = self.flow[t:t + self.horizon].T
        return X, Y

def denormalize(x, mean, std):
    return x * (std + 1e-8) + mean


# =============================================================================
# 1. CARGAR DATOS Y MODELO
# =============================================================================

print("\nCargando datos...")
flow  = np.load(DATA_CONFIG["flow_path"])   # [720, 3225]
dates = np.load(DATA_CONFIG["dates_path"], allow_pickle=True)
hours_index = pd.to_datetime(dates)

T           = len(flow)
history_len = DATA["history_len"]
horizon     = DATA["horizon"]
n_train     = int(T * DATA["train_ratio"])
n_val       = int(T * DATA["val_ratio"])

# Normalización (solo sobre train, igual que en train.py)
mean_val = flow[:n_train].mean()
std_val  = flow[:n_train].std()
flow_norm = (flow - mean_val) / (std_val + 1e-8)

# Índices test
test_t_start = n_train + n_val - horizon + 1
test_t_end   = T - horizon

print(f"  Train: {n_train} pasos | Val: {n_val} | Test: {T-n_train-n_val}")
print(f"  Ventanas test: {test_t_end - test_t_start + 1}")

# Cargar modelo
device   = torch.device(TRAIN["device"])
supports = load_supports(DATA_CONFIG["graphs_dir"], device=str(device))

model = FullModel(
    propath=DATA_CONFIG["graphs_dir"], supports=supports, device=str(device),
    fea_dim=STAR["fea_dim"], hid_dim=STAR["hid_dim"], out_dim_emb=STAR["out_dim"],
    layer_num=STAR["layer_num"], head_num=STAR["head_num"],
    out_dim_pred=horizon, history_len=history_len,
    residual_channels=GWN["residual_channels"], dilation_channels=GWN["dilation_channels"],
    skip_channels=GWN["skip_channels"], end_channels=GWN["end_channels"],
    dropout=GWN["dropout"], blocks=GWN["blocks"], layers=GWN["layers"],
).to(device)

best_model_path = os.path.join(DATA_CONFIG["output_dir"], f"best_model_h{horizon}.pt")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print(f"Modelo cargado: {best_model_path}")

# Cargar categorías
poi_to_idx_arr = np.load(
    DATA_CONFIG["flow_path"].replace("flow.npy", "poi_to_idx.npy"), allow_pickle=True
)
idx_to_poi = {int(row[1]): str(row[0]) for row in poi_to_idx_arr}
pois_df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "raw", "taptap",
                                    "source_poibrandsesp_tags_202604170958.csv"))
pois_df["id"] = pois_df["id"].astype(str)
poi_to_cat  = dict(zip(pois_df["id"], pois_df["tier1_category"]))
poi_to_name = dict(zip(pois_df["id"], pois_df["name"]))
N = flow.shape[1]
categories = np.array([poi_to_cat.get(idx_to_poi.get(i, ""), "Unknown") for i in range(N)])
poi_names  = np.array([poi_to_name.get(idx_to_poi.get(i, ""), f"POI_{i}") for i in range(N)])


# =============================================================================
# 2. GENERAR PREDICCIONES EN TEST
# =============================================================================

print("\nGenerando predicciones en test...")
test_dataset = FlowDataset(flow_norm, history_len, horizon, test_t_start, test_t_end)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_preds  = []  # [num_windows, N, horizon]
all_targets = [] # [num_windows, N, horizon]

with torch.no_grad():
    for X, Y in test_loader:
        X = X.to(device)
        pred = model(X)
        pred_real = denormalize(pred, mean_val, std_val).cpu().numpy()
        Y_real    = denormalize(Y,    mean_val, std_val).numpy()
        all_preds.append(pred_real)
        all_targets.append(Y_real)

all_preds   = np.concatenate(all_preds,   axis=0)  # [145, 3225, 72]
all_targets = np.concatenate(all_targets, axis=0)  # [145, 3225, 72]
all_errors  = np.abs(all_preds - all_targets)       # [145, 3225, 72]

print(f"  Predicciones shape: {all_preds.shape}")

# Baselines
def baseline_historical_mean(flow_train, test_dataset, horizon):
    mean_by_hour = np.zeros((24, flow_train.shape[1]))
    flow_hours_arr = hours_index[:n_train]
    for h in range(24):
        mask = flow_hours_arr.hour == h
        mean_by_hour[h] = flow_train[mask].mean(axis=0) if mask.any() else 0
    preds = []
    for i in range(len(test_dataset)):
        t = test_dataset.t_indices[i]
        pred = np.stack([mean_by_hour[hours_index[t + k].hour] for k in range(horizon)], axis=1)
        preds.append(pred)
    return np.stack(preds, axis=0)

def baseline_last_value(flow_orig, test_dataset, horizon):
    preds = []
    for i in range(len(test_dataset)):
        t = test_dataset.t_indices[i]
        last = flow_orig[t - 1]
        preds.append(np.tile(last[:, np.newaxis], (1, horizon)))
    return np.stack(preds, axis=0)

print("  Calculando baselines...")
hist_mean_pred = baseline_historical_mean(flow[:n_train], test_dataset, horizon)
last_val_pred  = baseline_last_value(flow, test_dataset, horizon)

def mae(pred, target):
    mask = target > 0
    if mask.sum() == 0: return np.nan
    return np.abs(pred[mask] - target[mask]).mean()

def rmse(pred, target):
    mask = target > 0
    if mask.sum() == 0: return np.nan
    return np.sqrt(((pred[mask] - target[mask])**2).mean())

mae_model = mae(all_preds, all_targets)
mae_hm    = mae(hist_mean_pred, all_targets)
mae_lv    = mae(last_val_pred,  all_targets)
rmse_model = rmse(all_preds, all_targets)
rmse_hm    = rmse(hist_mean_pred, all_targets)
rmse_lv    = rmse(last_val_pred,  all_targets)

print(f"\n  MAE  — Modelo: {mae_model:.4f} | Hist. Mean: {mae_hm:.4f} | Last Value: {mae_lv:.4f}")
print(f"  RMSE — Modelo: {rmse_model:.4f} | Hist. Mean: {rmse_hm:.4f} | Last Value: {rmse_lv:.4f}")


# =============================================================================
# 3. CURVA DE ENTRENAMIENTO (desde train.log)
# =============================================================================

print("\nParsing train.log...")
log_path = os.path.join(PROJECT_DIR, "train.log")
epochs_data = {"epoch": [], "train_loss": [], "val_mae": [], "val_rmse": []}

if os.path.exists(log_path):
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Época\s+(\d+)/\d+.*Train Loss:\s+([\d.]+).*Val MAE:\s+([\d.]+).*Val RMSE:\s+([\d.]+)', line)
            if m:
                epochs_data["epoch"].append(int(m.group(1)))
                epochs_data["train_loss"].append(float(m.group(2)))
                epochs_data["val_mae"].append(float(m.group(3)))
                epochs_data["val_rmse"].append(float(m.group(4)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Curva de entrenamiento — {CONFIG_NAME}', fontsize=14, fontweight='bold')

if epochs_data["epoch"]:
    axes[0].plot(epochs_data["epoch"], epochs_data["train_loss"],
                 color=COLORS["blue"], linewidth=1.5, label='Train Loss (MAE norm.)')
    axes[0].set_title('Loss de entrenamiento')
    axes[0].set_xlabel('Época'); axes[0].set_ylabel('MAE normalizado')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_data["epoch"], epochs_data["val_mae"],
                 color=COLORS["green"], linewidth=1.5, label='Val MAE')
    axes[1].axhline(mae_hm, color=COLORS["red"], linestyle='--', linewidth=1.5,
                    label=f'Baseline Hist. Mean: {mae_hm:.2f}')
    axes[1].axhline(mae_lv, color=COLORS["orange"], linestyle='--', linewidth=1.5,
                    label=f'Baseline Last Value: {mae_lv:.2f}')
    best_epoch = epochs_data["epoch"][np.argmin(epochs_data["val_mae"])]
    best_mae   = min(epochs_data["val_mae"])
    axes[1].scatter([best_epoch], [best_mae], color=COLORS["green"], s=80, zorder=5,
                    label=f'Mejor (época {best_epoch}): {best_mae:.2f}')
    axes[1].set_title('Val MAE vs baselines')
    axes[1].set_xlabel('Época'); axes[1].set_ylabel('MAE (visitas/hora)')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/curva_entrenamiento_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardado: curva_entrenamiento_{CONFIG_NAME}.png")


# =============================================================================
# 4. COMPARATIVA CON BASELINES
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle(f'Comparativa con baselines — {CONFIG_NAME}', fontsize=14, fontweight='bold')

modelos  = ['Historical\nMean', 'Last\nValue', 'STAR +\nGWN']
maes_bar = [mae_hm, mae_lv, mae_model]
rmses_bar= [rmse_hm, rmse_lv, rmse_model]
colores  = [COLORS["orange"], COLORS["red"], COLORS["blue"]]

bars = axes[0].bar(modelos, maes_bar, color=colores, edgecolor='white', linewidth=0.5)
axes[0].set_title('MAE (visitas/hora)'); axes[0].set_ylabel('MAE')
for bar, val in zip(bars, maes_bar):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
mejora_mae = (1 - mae_model/mae_hm)*100
axes[0].text(0.5, 0.95, f'Mejora vs Hist. Mean: {mejora_mae:.1f}%',
             transform=axes[0].transAxes, ha='center', va='top',
             fontsize=10, color=COLORS["blue"],
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

bars = axes[1].bar(modelos, rmses_bar, color=colores, edgecolor='white', linewidth=0.5)
axes[1].set_title('RMSE (visitas/hora)'); axes[1].set_ylabel('RMSE')
for bar, val in zip(bars, rmses_bar):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/comparativa_baselines_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardado: comparativa_baselines_{CONFIG_NAME}.png")


# =============================================================================
# 5. ERROR POR HORIZONTE DE PREDICCIÓN (h=1 a h=72)
# =============================================================================

mae_by_h  = [mae(all_preds[:, :, h], all_targets[:, :, h]) for h in range(horizon)]
mae_hm_h  = [mae(hist_mean_pred[:, :, h], all_targets[:, :, h]) for h in range(horizon)]
mae_lv_h  = [mae(last_val_pred[:, :, h],  all_targets[:, :, h]) for h in range(horizon)]

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(f'MAE por horizonte de predicción — {CONFIG_NAME}', fontsize=14, fontweight='bold')
ax.plot(range(1, horizon+1), mae_by_h,  color=COLORS["blue"],   linewidth=2,   label='STAR + GWN')
ax.plot(range(1, horizon+1), mae_hm_h,  color=COLORS["orange"], linewidth=1.5, linestyle='--', label='Historical Mean')
ax.plot(range(1, horizon+1), mae_lv_h,  color=COLORS["red"],    linewidth=1.5, linestyle='--', label='Last Value')
ax.set_xlabel('Horizonte (horas)'); ax.set_ylabel('MAE (visitas/hora)')
ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
ax.set_xticklabels(['h=1\n(1h)', 'h=12\n(12h)', 'h=24\n(1d)', 'h=36', 'h=48\n(2d)', 'h=60', 'h=72\n(3d)'])
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/mae_por_horizonte_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardado: mae_por_horizonte_{CONFIG_NAME}.png")


# =============================================================================
# 6. ERROR POR HORA DEL DÍA (DAYPARTING)
# =============================================================================

# Para cada ventana de test, tomamos el primer paso de predicción (h=0)
# y lo agrupamos por hora del día de ese timestep
mae_by_hour = np.zeros(24)
mae_hm_hour = np.zeros(24)
n_by_hour   = np.zeros(24)

for i in range(len(test_dataset)):
    t = test_dataset.t_indices[i]
    for h in range(horizon):
        hora = hours_index[t + h].hour
        mask = all_targets[i, :, h] > 0
        if mask.sum() > 0:
            mae_by_hour[hora] += np.abs(all_preds[i, mask, h] - all_targets[i, mask, h]).mean()
            mae_hm_hour[hora] += np.abs(hist_mean_pred[i, mask, h] - all_targets[i, mask, h]).mean()
            n_by_hour[hora]   += 1

mae_by_hour /= np.maximum(n_by_hour, 1)
mae_hm_hour /= np.maximum(n_by_hour, 1)

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(f'MAE por hora del día (dayparting) — {CONFIG_NAME}', fontsize=14, fontweight='bold')
x = np.arange(24)
w = 0.35
ax.bar(x - w/2, mae_by_hour, w, color=COLORS["blue"],   label='STAR + GWN',      edgecolor='white')
ax.bar(x + w/2, mae_hm_hour, w, color=COLORS["orange"], label='Historical Mean',  edgecolor='white', alpha=0.8)
for h_label, label in [(6,'Mañana'), (12,'Mediodía'), (18,'Tarde'), (22,'Noche')]:
    ax.axvline(h_label - 0.5, color='gray', linestyle=':', alpha=0.5)
    ax.text(h_label + 0.1, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 1,
            label, fontsize=8, color='gray')
ax.set_xlabel('Hora del día'); ax.set_ylabel('MAE medio (visitas/hora)')
ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels([f'{h:02d}h' for h in range(0, 24, 2)])
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/mae_dayparting_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardado: mae_dayparting_{CONFIG_NAME}.png")


# =============================================================================
# 7. ERROR POR DÍA DE LA SEMANA
# =============================================================================

dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
mae_by_dow = np.zeros(7)
mae_hm_dow = np.zeros(7)
n_by_dow   = np.zeros(7)

for i in range(len(test_dataset)):
    t = test_dataset.t_indices[i]
    for h in range(horizon):
        dow  = hours_index[t + h].dayofweek
        mask = all_targets[i, :, h] > 0
        if mask.sum() > 0:
            mae_by_dow[dow] += np.abs(all_preds[i, mask, h] - all_targets[i, mask, h]).mean()
            mae_hm_dow[dow] += np.abs(hist_mean_pred[i, mask, h] - all_targets[i, mask, h]).mean()
            n_by_dow[dow]   += 1

mae_by_dow /= np.maximum(n_by_dow, 1)
mae_hm_dow /= np.maximum(n_by_dow, 1)

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle(f'MAE por día de la semana — {CONFIG_NAME}', fontsize=14, fontweight='bold')
x = np.arange(7)
w = 0.35
colors_dow = [COLORS["orange"] if d >= 5 else COLORS["blue"] for d in range(7)]
colors_hm  = [COLORS["red"] if d >= 5 else '#BA7517' for d in range(7)]
ax.bar(x - w/2, mae_by_dow, w, color=colors_dow, label='STAR + GWN',     edgecolor='white')
ax.bar(x + w/2, mae_hm_dow, w, color=COLORS["orange"], label='Historical Mean', edgecolor='white', alpha=0.7)
ax.set_xticks(range(7)); ax.set_xticklabels(dias)
ax.set_ylabel('MAE medio (visitas/hora)'); ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
ax.axvspan(4.5, 6.5, alpha=0.08, color='orange', label='Fin de semana')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/mae_por_dia_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardado: mae_por_dia_{CONFIG_NAME}.png")


# =============================================================================
# 8. ERROR POR CATEGORÍA
# =============================================================================

unique_cats = pd.Series(categories).value_counts().index.tolist()
mae_model_cat = {}
mae_hm_cat    = {}
mae_lv_cat    = {}

for cat in unique_cats:
    mask_cat = categories == cat
    mae_model_cat[cat] = mae(all_preds[:, mask_cat, :],       all_targets[:, mask_cat, :])
    mae_hm_cat[cat]    = mae(hist_mean_pred[:, mask_cat, :],  all_targets[:, mask_cat, :])
    mae_lv_cat[cat]    = mae(last_val_pred[:, mask_cat, :],   all_targets[:, mask_cat, :])

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle(f'MAE por categoría de POI — {CONFIG_NAME}', fontsize=14, fontweight='bold')
x   = np.arange(len(unique_cats))
w   = 0.28
ax.bar(x - w,   [mae_hm_cat[c]    for c in unique_cats], w, color=COLORS["orange"], label='Historical Mean', edgecolor='white')
ax.bar(x,       [mae_lv_cat[c]    for c in unique_cats], w, color=COLORS["red"],    label='Last Value',      edgecolor='white')
ax.bar(x + w,   [mae_model_cat[c] for c in unique_cats], w, color=COLORS["blue"],   label='STAR + GWN',      edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([f'{c}\n({(categories==c).sum()} POIs)' for c in unique_cats], fontsize=9)
ax.set_ylabel('MAE (visitas/hora)'); ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
for i, cat in enumerate(unique_cats):
    ax.text(i + w, mae_model_cat[cat] + 0.1, f'{mae_model_cat[cat]:.1f}',
            ha='center', va='bottom', fontsize=8, color=COLORS["blue"], fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/mae_por_categoria_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardado: mae_por_categoria_{CONFIG_NAME}.png")


# =============================================================================
# 9. PREDICCIÓN VS REAL — POIs de ejemplo (uno por categoría)
# =============================================================================

# Para cada categoría, tomamos el POI más activo
example_pois = {}
for cat in unique_cats:
    mask_cat = categories == cat
    idx_cat  = np.where(mask_cat)[0]
    flujo_medio = flow[:, idx_cat].mean(axis=0)
    example_pois[cat] = idx_cat[np.argmax(flujo_medio)]

# Tomamos la ventana de test del centro del período
mid_window = len(test_dataset) // 2
t_mid = test_dataset.t_indices[mid_window]
t_range = pd.date_range(hours_index[t_mid], periods=horizon, freq='h')

fig, axes = plt.subplots(len(unique_cats), 1, figsize=(14, 3*len(unique_cats)))
fig.suptitle(f'Predicción vs Real — POI más activo por categoría\n{CONFIG_NAME}',
             fontsize=13, fontweight='bold')

for i, cat in enumerate(unique_cats):
    poi_idx = example_pois[cat]
    ax = axes[i] if len(unique_cats) > 1 else axes
    real  = all_targets[mid_window, poi_idx, :]
    pred  = all_preds[mid_window, poi_idx, :]
    hm    = hist_mean_pred[mid_window, poi_idx, :]
    ax.plot(range(horizon), real, color='black',         linewidth=1.5, label='Real',           alpha=0.9)
    ax.plot(range(horizon), pred, color=COLORS["blue"],  linewidth=1.5, label='STAR + GWN',     alpha=0.9)
    ax.plot(range(horizon), hm,   color=COLORS["orange"],linewidth=1.2, label='Hist. Mean',     alpha=0.7, linestyle='--')
    ax.fill_between(range(horizon), pred, real, alpha=0.1, color=COLORS["blue"])
    mae_poi = np.abs(pred - real).mean()
    ax.set_title(f'{cat} — {poi_names[poi_idx][:40]} (MAE={mae_poi:.1f})', fontsize=9)
    ax.set_ylabel('Visitas/hora')
    ax.set_xticks([0, 12, 24, 36, 48, 60, 71])
    ax.set_xticklabels(['h+1', 'h+12', 'h+24\n(+1d)', 'h+36', 'h+48\n(+2d)', 'h+60', 'h+72\n(+3d)'], fontsize=8)
    if i == 0:
        ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/prediccion_vs_real_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardado: prediccion_vs_real_{CONFIG_NAME}.png")


# =============================================================================
# RESUMEN FINAL
# =============================================================================

print(f"\n{'='*60}")
print(f"RESUMEN DE RESULTADOS — {CONFIG_NAME}")
print(f"{'='*60}")
print(f"{'Modelo':<20} {'MAE':>8} {'RMSE':>8}  {'Mejora MAE':>12}")
print(f"{'-'*50}")
print(f"{'Historical Mean':<20} {mae_hm:>8.4f} {rmse_hm:>8.4f}  {'(baseline)':>12}")
print(f"{'Last Value':<20} {mae_lv:>8.4f} {rmse_lv:>8.4f}  {'(baseline)':>12}")
print(f"{'STAR + GWN':<20} {mae_model:>8.4f} {rmse_model:>8.4f}  {(1-mae_model/mae_hm)*100:>11.1f}%")
print(f"\nMAE por categoría:")
for cat in unique_cats:
    mejora = (1 - mae_model_cat[cat]/mae_hm_cat[cat])*100
    print(f"  {cat:<20} {mae_model_cat[cat]:>8.4f}  ({mejora:+.1f}% vs Hist. Mean)")
print(f"\nImágenes en: {OUTPUT_DIR}/")