"""
analysis_embeddings.py - TapTap Madrid
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)

from config import DATA_CONFIG, DATA, STAR, TRAIN
from model.full_model import FullModel, load_supports
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs", "embeddings")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

CONFIG_NAME = DATA_CONFIG["config_name"]
print(f"Configuración: {CONFIG_NAME}")

# 1. CARGAR MODELO Y EXTRAER EMBEDDINGS
device   = torch.device(TRAIN["device"])
supports = load_supports(DATA_CONFIG["graphs_dir"], device=str(device))
model = FullModel(
    propath=DATA_CONFIG["graphs_dir"], supports=supports, device=str(device),
    fea_dim=STAR["fea_dim"], hid_dim=STAR["hid_dim"], out_dim_emb=STAR["out_dim"],
    layer_num=STAR["layer_num"], head_num=STAR["head_num"],
    out_dim_pred=DATA["horizon"], history_len=DATA["history_len"],
).to(device)
best_model_path = os.path.join(DATA_CONFIG["output_dir"], f"best_model_h{DATA['horizon']}.pt")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print(f"Modelo cargado desde: {best_model_path}")
with torch.no_grad():
    embeddings, attn_weights = model.star_embedding()
embeddings   = embeddings.cpu().numpy()
attn_weights = attn_weights.cpu().numpy()
N = embeddings.shape[0]
print(f"Embeddings: {embeddings.shape}  Attn weights: {attn_weights.shape}")

# 2. CARGAR METADATOS (GPS + categorías TapTap)
gps = np.load(os.path.join(PROJECT_DIR, "data", "processed", "flows", "Madrid", "gps.npy"))
lats, lons = gps[:, 0], gps[:, 1]

poi_to_idx_arr = np.load(
    os.path.join(PROJECT_DIR, "data", "processed", "flows", "Madrid", "poi_to_idx.npy"),
    allow_pickle=True
)
idx_to_poi = {int(row[1]): str(row[0]) for row in poi_to_idx_arr}

pois_df = pd.read_csv(os.path.join(
    PROJECT_DIR, "data", "raw", "taptap",
    "source_poibrandsesp_tags_202604170958.csv"
))
pois_df["id"] = pois_df["id"].astype(str)
poi_to_cat  = dict(zip(pois_df["id"], pois_df["tier1_category"]))
poi_to_name = dict(zip(pois_df["id"], pois_df["name"]))

categories = np.array([poi_to_cat.get(idx_to_poi.get(i, ""), "Unknown") for i in range(N)])
names      = np.array([poi_to_name.get(idx_to_poi.get(i, ""), "Unknown") for i in range(N)])

print(f"\nDistribución de categorías:")
for cat, cnt in pd.Series(categories).value_counts().items():
    print(f"  {cat:<22} {cnt:4d} POIs ({cnt/N*100:.1f}%)")

# 3. t-SNE COLOREADO POR CATEGORÍA
print("\nCalculando t-SNE...")
scaler     = StandardScaler()
emb_scaled = scaler.fit_transform(embeddings)
tsne       = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
emb_2d     = tsne.fit_transform(emb_scaled)

unique_cats  = pd.Series(categories).value_counts().index.tolist()
palette      = plt.cm.tab10(np.linspace(0, 1, min(len(unique_cats), 10)))
cat_to_color = {cat: palette[i % 10] for i, cat in enumerate(unique_cats)}

fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle(f't-SNE embeddings STAR — {CONFIG_NAME}', fontsize=14, fontweight='bold')
for cat in unique_cats:
    mask = categories == cat
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[cat_to_color[cat]],
               s=15, alpha=0.7, label=f"{cat} ({mask.sum()})")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tsne_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: tsne_{CONFIG_NAME}.png")

# 4. MAPA GEOESPACIAL + PCA
print("Calculando PCA...")
pca     = PCA(n_components=3)
emb_pca = pca.fit_transform(emb_scaled)
print(f"  Varianza explicada: {pca.explained_variance_ratio_[:3]*100}")
pc1      = emb_pca[:, 0]
pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
valid    = ~(np.isnan(lats) | np.isnan(lons) | (lats == 0) | (lons == 0))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f'Mapa geoespacial embeddings STAR — {CONFIG_NAME}', fontsize=14, fontweight='bold')
sc1 = axes[0].scatter(lons[valid], lats[valid], c=pc1_norm[valid], cmap='RdYlBu', s=12, alpha=0.7)
axes[0].set_title('Coloreado por PC1'); axes[0].set_xlabel('Longitud'); axes[0].set_ylabel('Latitud')
plt.colorbar(sc1, ax=axes[0], label='PC1 (normalizado)')
for cat in unique_cats:
    mask = (categories == cat) & valid
    if mask.sum() == 0: continue
    axes[1].scatter(lons[mask], lats[mask], c=[cat_to_color[cat]], s=10, alpha=0.6,
                    label=f"{cat} ({mask.sum()})")
axes[1].set_title('Categorías en el mapa'); axes[1].set_xlabel('Longitud')
axes[1].legend(fontsize=8, markerscale=1.5)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/mapa_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: mapa_{CONFIG_NAME}.png")

# 5. ATTENTION WEIGHTS
print("Analizando attention weights...")
graph_names = ['SDG\n(Distancia)', 'TTG\n(Transiciones)', 'STG\n(Similitud)']
colors_att  = ['#378ADD', '#1D9E75', '#D85A30']

print("\n=== ATTENTION WEIGHTS MEDIOS POR GRAFO ===")
for i, name in enumerate(['SDG', 'TTG', 'STG']):
    w = attn_weights[:, i]
    print(f"  {name}: media={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Attention weights STAR — {CONFIG_NAME}', fontsize=14, fontweight='bold')
for i, (name, color) in enumerate(zip(graph_names, colors_att)):
    axes[0].hist(attn_weights[:, i], bins=30, alpha=0.6, color=color,
                 label=name.replace('\n',' '), edgecolor='white')
axes[0].set_title('Distribución'); axes[0].set_xlabel('Peso'); axes[0].set_ylabel('Nº POIs')
axes[0].legend(fontsize=9)

means = attn_weights.mean(axis=0)
bars  = axes[1].bar(graph_names, means, color=colors_att, edgecolor='white')
axes[1].set_title('Peso medio por grafo'); axes[1].set_ylabel('Peso medio')
for bar, val in zip(bars, means):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)

dominant = attn_weights.argmax(axis=1)
axes[2].scatter(lons[valid], lats[valid],
    c=[np.array(colors_att)[dominant[i]] for i in np.where(valid)[0]], s=10, alpha=0.6)
axes[2].set_title('Grafo dominante por POI'); axes[2].set_xlabel('Longitud'); axes[2].set_ylabel('Latitud')
from matplotlib.patches import Patch
axes[2].legend(handles=[Patch(facecolor=colors_att[i], label=graph_names[i].replace('\n',' '))
                         for i in range(3)], fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/attention_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: attention_{CONFIG_NAME}.png")

print("\n=== ATTENTION WEIGHTS POR CATEGORÍA ===")
print(f"{'Categoría':<22} {'SDG':>8} {'TTG':>8} {'STG':>8} {'Dominante':>12}")
print("-" * 62)
for cat in unique_cats:
    mask   = categories == cat
    w_mean = attn_weights[mask].mean(axis=0)
    dom    = ['SDG', 'TTG', 'STG'][w_mean.argmax()]
    print(f"  {cat:<20} {w_mean[0]:>8.4f} {w_mean[1]:>8.4f} {w_mean[2]:>8.4f} {dom:>12}")

# 6. SIMILITUD COSENO ENTRE CATEGORÍAS
emb_norm   = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
n_cats     = len(unique_cats)
sim_matrix = np.zeros((n_cats, n_cats))
for i, ci in enumerate(unique_cats):
    for j, cj in enumerate(unique_cats):
        sim_matrix[i, j] = (emb_norm[categories==ci] @ emb_norm[categories==cj].T).mean()

fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle(f'Similitud coseno entre categorías — {CONFIG_NAME}', fontsize=13, fontweight='bold')
im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(n_cats)); ax.set_yticks(range(n_cats))
ax.set_xticklabels(unique_cats, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(unique_cats, fontsize=9)
plt.colorbar(im, ax=ax, label='Similitud coseno media')
for i in range(n_cats):
    for j in range(n_cats):
        ax.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/similitud_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: similitud_{CONFIG_NAME}.png")
print(f"\nImágenes en: {OUTPUT_DIR}/")

# 7. PATRONES DE FLUJO POR CATEGORÍA
print("\nCalculando patrones de flujo por categoría...")

flow = np.load(os.path.join(PROJECT_DIR, "data", "processed", "flows", "Madrid", "flow.npy"))
flow_hours = np.load(os.path.join(PROJECT_DIR, "data", "processed", "flows", "Madrid", "flow_hours.npy"), allow_pickle=True)

import pandas as pd
hours_index = pd.to_datetime(flow_hours)
hour_of_day  = hours_index.hour      # 0-23
day_of_week  = hours_index.dayofweek # 0=Lun, 6=Dom

fig, axes = plt.subplots(1, len(unique_cats), figsize=(4*len(unique_cats), 5))
fig.suptitle(f'Patrón de flujo medio por categoría — hora × día semana\n{CONFIG_NAME}',
             fontsize=13, fontweight='bold')

vmin_global = None
vmax_global = None
heatmaps = []
for cat in unique_cats:
    mask_cat = categories == cat
    flow_cat = flow[:, mask_cat].mean(axis=1)  # flujo medio horario de esa categoría
    hm = np.zeros((24, 7))
    for h in range(24):
        for d in range(7):
            idx = (hour_of_day == h) & (day_of_week == d)
            hm[h, d] = flow_cat[idx].mean() if idx.any() else 0
    heatmaps.append(hm)
    vmin_global = hm.min() if vmin_global is None else min(vmin_global, hm.min())
    vmax_global = hm.max() if vmax_global is None else max(vmax_global, hm.max())

dias = ['Lun','Mar','Mié','Jue','Vie','Sáb','Dom']
for i, (cat, hm) in enumerate(zip(unique_cats, heatmaps)):
    ax = axes[i] if len(unique_cats) > 1 else axes
    im = ax.imshow(hm, aspect='auto', cmap='YlOrRd', origin='upper',
                   vmin=vmin_global, vmax=vmax_global)
    ax.set_title(f'{cat}\n({(categories==cat).sum()} POIs)', fontsize=9)
    ax.set_xlabel('Día semana')
    ax.set_xticks(range(7)); ax.set_xticklabels(dias, fontsize=7)
    ax.set_yticks(range(0, 24, 3)); ax.set_yticklabels([f'{h:02d}h' for h in range(0, 24, 3)], fontsize=7)
    if i == 0:
        ax.set_ylabel('Hora del día')

plt.colorbar(im, ax=axes[-1] if len(unique_cats) > 1 else axes, label='Flujo medio')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/patrones_flujo_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: patrones_flujo_{CONFIG_NAME}.png")