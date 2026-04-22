"""
analysis_geo_taptap.py
Visualización geoespacial de los POIs de TapTap (Madrid).

Uso:
    python analysis/analysis_geo_taptap.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FLOWS_PATH = Path("data/raw/taptap/flows_hourly.csv")
POIS_PATH  = Path("data/raw/taptap/source_poibrandsesp_tags_202604170958.csv")
OUT_DIR    = Path("analysis/imgs/taptap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Carga ──────────────────────────────────────────────────────────────────────
pois = pd.read_csv(POIS_PATH)
pois["id"] = pois["id"].astype(str)

flows = pd.read_csv(FLOWS_PATH)
flows["poi_id"] = flows["poi_id"].astype(str)
visitas_poi = flows.groupby("poi_id")["visit_count"].sum().reset_index()
visitas_poi.columns = ["id", "visitas"]

pois = pois.merge(visitas_poi, on="id", how="left").fillna(0)

print(f"POIs totales: {len(pois)}")
print(f"Rango lat: {pois['latitude'].min():.4f} → {pois['latitude'].max():.4f}")
print(f"Rango lon: {pois['longitude'].min():.4f} → {pois['longitude'].max():.4f}")

# ── Colores por categoría ──────────────────────────────────────────────────────
cat_colors = {
    "Pharmacy":    "#378ADD",
    "Cafe":        "#D85A30",
    "Coffee shop": "#1D9E75",
    "Bar & grill": "#BA7517",
    "Tapas bar":   "#9B59B6",
}
pois["color"] = pois["tier1_category"].map(cat_colors).fillna("gray")

# ── Gráficos ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Análisis geoespacial — TapTap Madrid, Septiembre 2025",
             fontsize=14, fontweight="bold")

# 1. Todos los POIs por categoría
for cat, color in cat_colors.items():
    sub = pois[pois["tier1_category"] == cat]
    axes[0].scatter(sub["longitude"], sub["latitude"],
                    s=4, alpha=0.5, color=color, label=f"{cat} ({len(sub)})")
axes[0].set_title(f"Todos los POIs por categoría ({len(pois):,})")
axes[0].set_xlabel("Longitud")
axes[0].set_ylabel("Latitud")
axes[0].legend(fontsize=8, markerscale=2)

# 2. Tamaño proporcional a visitas (solo activos)
activos = pois[pois["visitas"] > 0].copy()
max_v = activos["visitas"].max()
sizes = (activos["visitas"] / max_v * 80).clip(lower=1)

sc = axes[1].scatter(activos["longitude"], activos["latitude"],
                     s=sizes, alpha=0.5,
                     c=activos["tier1_category"].map(cat_colors).fillna("gray"))
axes[1].set_title(f"POIs activos — tamaño ∝ visitas ({len(activos):,})")
axes[1].set_xlabel("Longitud")
axes[1].set_ylabel("Latitud")

# Anotar top 5 POIs
top5 = activos.nlargest(5, "visitas")
for _, row in top5.iterrows():
    axes[1].annotate(row["name"][:20],
                     (row["longitude"], row["latitude"]),
                     fontsize=6, ha="left",
                     xytext=(4, 4), textcoords="offset points")

# Leyenda categorías
patches = [mpatches.Patch(color=c, label=cat) for cat, c in cat_colors.items()]
axes[1].legend(handles=patches, fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "taptap_analisis_geo.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Guardado en {OUT_DIR}/taptap_analisis_geo.png")