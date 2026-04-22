"""
analysis_embeddings.py
======================
Análisis e interpretación de los embeddings aprendidos por STAREmbedding.

Análisis implementados:
    1. t-SNE coloreado por categoría de POI
    2. Mapa geoespacial coloreado por primer componente PCA
    3. Attention weights: importancia de cada grafo por POI

Uso:
    Asegúrate de que config.py tiene el CONFIG_NAME correcto y ejecuta:
    python analysis_embeddings.py

"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from config import DATA_CONFIG, DATA, STAR, TRAIN
from model.full_model import FullModel, load_supports

# ── Sklearn para reducción de dimensionalidad ─────────────────────────────────
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs", "embeddings")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

CONFIG_NAME = DATA_CONFIG["config_name"]
print(f"Configuración: {CONFIG_NAME}")


# =============================================================================
# 1. CARGAR MODELO Y EXTRAER EMBEDDINGS
# =============================================================================

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
    out_dim_pred      = DATA["horizon"],
    history_len       = DATA["history_len"],
).to(device)

# Cargar pesos del mejor modelo entrenado
best_model_path = os.path.join(
    DATA_CONFIG["output_dir"],
    f"best_model_h{DATA['horizon']}.pt"
)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print(f"Modelo cargado desde: {best_model_path}")

# Extraer embeddings y attention weights
with torch.no_grad():
    embeddings, attn_weights = model.star_embedding()

embeddings   = embeddings.cpu().numpy()    # [N, 32]
attn_weights = attn_weights.cpu().numpy()  # [N, 3]
N = embeddings.shape[0]
print(f"Embeddings shape:    {embeddings.shape}")
print(f"Attn weights shape:  {attn_weights.shape}")


# =============================================================================
# 2. CARGAR METADATOS DE POIs (coordenadas y categorías)
# =============================================================================

# Cargar poi_to_idx para mapear poi_id → venueId
poi_to_idx = np.load(
    os.path.join(DATA_CONFIG["flow_path"].replace("flow.npy", "poi_to_idx.npy")),
    allow_pickle=True
)
# poi_to_idx es array de pares [venueId, poi_id]
venue_to_id = {row[0]: int(row[1]) for row in poi_to_idx}
id_to_venue = {v: k for k, v in venue_to_id.items()}

# Cargar CSV raw para obtener categorías y coordenadas
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "raw", "dataset_TSMC2014_NYC.csv"))
df['datetime'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S +0000 %Y')

# Construir DataFrame de POIs con sus metadatos
poi_meta = df.drop_duplicates('venueId')[['venueId', 'venueCategory', 'latitude', 'longitude']].copy()
poi_meta['poi_id'] = poi_meta['venueId'].map(venue_to_id)
poi_meta = poi_meta.dropna(subset=['poi_id'])
poi_meta['poi_id'] = poi_meta['poi_id'].astype(int)
poi_meta = poi_meta[poi_meta['poi_id'] < N].sort_values('poi_id').reset_index(drop=True)

print(f"POIs con metadatos: {len(poi_meta)}")

# Arrays ordenados por poi_id
lats       = poi_meta.set_index('poi_id')['latitude'].reindex(range(N)).values
lons       = poi_meta.set_index('poi_id')['longitude'].reindex(range(N)).values
categories = poi_meta.set_index('poi_id')['venueCategory'].reindex(range(N)).fillna('Unknown').values


# =============================================================================
# 3. ANÁLISIS 1: t-SNE COLOREADO POR CATEGORÍA
# =============================================================================

print("\nCalculando t-SNE...")

# Normalizar embeddings antes de t-SNE
scaler     = StandardScaler()
emb_scaled = scaler.fit_transform(embeddings)

tsne     = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
emb_2d   = tsne.fit_transform(emb_scaled)

# Top 10 categorías más frecuentes
top_cats  = pd.Series(categories).value_counts().head(10).index.tolist()
cat_colors = plt.cm.tab10(np.linspace(0, 1, len(top_cats)))
cat_to_color = {cat: cat_colors[i] for i, cat in enumerate(top_cats)}

fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle(f't-SNE embeddings STAR — {CONFIG_NAME}', fontsize=14, fontweight='bold')

# Primero plotear los POIs que no están en top 10 en gris
mask_other = ~np.isin(categories, top_cats)
ax.scatter(emb_2d[mask_other, 0], emb_2d[mask_other, 1],
           c='lightgray', s=8, alpha=0.3, label='Otras categorías')

# Luego plotear top 10 categorías con colores
for cat in top_cats:
    mask = categories == cat
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
               c=[cat_to_color[cat]], s=15, alpha=0.7,
               label=cat[:25])

ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.set_xlabel('t-SNE dim 1')
ax.set_ylabel('t-SNE dim 2')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tsne_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: tsne_{CONFIG_NAME}.png")


# =============================================================================
# 4. ANÁLISIS 2: MAPA GEOESPACIAL COLOREADO POR PCA
# =============================================================================

print("Calculando PCA...")

pca       = PCA(n_components=3)
emb_pca   = pca.fit_transform(emb_scaled)
print(f"  Varianza explicada (PC1, PC2, PC3): "
      f"{pca.explained_variance_ratio_[:3]*100}")

# Colorear por primer componente PCA
pc1 = emb_pca[:, 0]
pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)

# Filtrar POIs con coordenadas válidas
valid_mask = ~(np.isnan(lats) | np.isnan(lons))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Mapa geoespacial embeddings STAR — {CONFIG_NAME}',
             fontsize=14, fontweight='bold')

# Izquierda: coloreado por PC1
sc1 = axes[0].scatter(
    lons[valid_mask], lats[valid_mask],
    c=pc1_norm[valid_mask], cmap='RdYlBu',
    s=15, alpha=0.7
)
axes[0].set_title('Coloreado por PC1 del embedding')
axes[0].set_xlabel('Longitud')
axes[0].set_ylabel('Latitud')
plt.colorbar(sc1, ax=axes[0], label='PC1 (normalizado)')

# Derecha: coloreado por categoría (top 5)
top5_cats = pd.Series(categories).value_counts().head(5).index.tolist()
colors5   = plt.cm.Set1(np.linspace(0, 1, len(top5_cats)))

mask_other2 = ~np.isin(categories, top5_cats) & valid_mask
axes[1].scatter(lons[mask_other2], lats[mask_other2],
                c='lightgray', s=5, alpha=0.2)

for i, cat in enumerate(top5_cats):
    mask = (categories == cat) & valid_mask
    axes[1].scatter(lons[mask], lats[mask],
                    c=[colors5[i]], s=20, alpha=0.8,
                    label=cat[:20])

axes[1].set_title('Top 5 categorías en el mapa')
axes[1].set_xlabel('Longitud')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/mapa_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: mapa_{CONFIG_NAME}.png")


# =============================================================================
# 5. ANÁLISIS 3: ATTENTION WEIGHTS
# =============================================================================

print("Analizando attention weights...")

graph_names = ['SDG\n(Distancia)', 'TTG\n(Transiciones)', 'STG\n(Similitud temporal)']

# ── Estadísticas globales ─────────────────────────────────────────────────────
print("\n=== ATTENTION WEIGHTS MEDIOS POR GRAFO ===")
for i, name in enumerate(['SDG', 'TTG', 'STG']):
    w = attn_weights[:, i]
    print(f"  {name}: media={w.mean():.4f}, std={w.std():.4f}, "
          f"min={w.min():.4f}, max={w.max():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Attention weights STAR — {CONFIG_NAME}',
             fontsize=14, fontweight='bold')

# ── Plot 1: distribución de weights por grafo ─────────────────────────────────
colors_att = ['#378ADD', '#1D9E75', '#D85A30']
for i, (name, color) in enumerate(zip(graph_names, colors_att)):
    axes[0].hist(attn_weights[:, i], bins=30, alpha=0.6,
                 color=color, label=name.replace('\n', ' '), edgecolor='white')
axes[0].set_title('Distribución de attention weights')
axes[0].set_xlabel('Peso')
axes[0].set_ylabel('Número de POIs')
axes[0].legend(fontsize=9)

# ── Plot 2: peso medio por grafo (barras) ─────────────────────────────────────
means = attn_weights.mean(axis=0)
bars  = axes[1].bar(graph_names, means, color=colors_att, edgecolor='white')
axes[1].set_title('Peso medio por grafo')
axes[1].set_ylabel('Peso medio de atención')
for bar, val in zip(bars, means):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)

# ── Plot 3: mapa coloreado por grafo dominante ────────────────────────────────
dominant_graph = attn_weights.argmax(axis=1)  # 0=SDG, 1=TTG, 2=STG
dom_colors     = np.array(colors_att)

axes[2].scatter(
    lons[valid_mask], lats[valid_mask],
    c=[dom_colors[dominant_graph[i]] for i in np.where(valid_mask)[0]],
    s=10, alpha=0.6
)
axes[2].set_title('Grafo dominante por POI')
axes[2].set_xlabel('Longitud')
axes[2].set_ylabel('Latitud')

# Leyenda manual
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_att[i], label=graph_names[i].replace('\n', ' '))
                   for i in range(3)]
axes[2].legend(handles=legend_elements, fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/attention_{CONFIG_NAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: attention_{CONFIG_NAME}.png")

# ── Análisis por categoría ────────────────────────────────────────────────────
print("\n=== ATTENTION WEIGHTS MEDIOS POR CATEGORÍA (top 10) ===")
print(f"{'Categoría':<30} {'SDG':>8} {'TTG':>8} {'STG':>8} {'Dominante':>12}")
print("-" * 70)

cat_series = pd.Series(categories)
top10 = cat_series.value_counts().head(10).index.tolist()

for cat in top10:
    mask    = categories == cat
    w_mean  = attn_weights[mask].mean(axis=0)
    dom     = ['SDG', 'TTG', 'STG'][w_mean.argmax()]
    print(f"  {cat:<28} {w_mean[0]:>8.4f} {w_mean[1]:>8.4f} {w_mean[2]:>8.4f} {dom:>12}")

print(f"\nImágenes guardadas en: {OUTPUT_DIR}/")