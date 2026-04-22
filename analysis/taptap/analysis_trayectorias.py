"""
analysis_trayectorias_madrid.py
================================
Analiza la distribución de longitudes de trayectorias diarias en el dataset
TapTap Madrid para decidir el min_seq_len óptimo antes de lanzar stg_gen.py.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="Madrid", type=str, help="Ciudad (subcarpeta en tra0.7-val0.1-test0.2/)")
args = parser.parse_args()

# Sube directorios hasta encontrar la raíz del proyecto (donde está config.py)
_here = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(_here)
while not os.path.exists(os.path.join(PROJECT_DIR, "config.py")):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
TRAJ_DIR    = os.path.join(PROJECT_DIR, "data", "processed",
                           "tra0.7-val0.1-test0.2", args.data)

# ── Leer longitudes ───────────────────────────────────────────────────────────
print("Leyendo train.txt...")
lengths = []
with open(f"{TRAJ_DIR}/train.txt") as f:
    for line in f:
        lengths.append(len(line.strip().split()))

lengths = np.array(lengths)
total   = len(lengths)

# ── Estadísticas ──────────────────────────────────────────────────────────────
print(f"\nTotal trayectorias train: {total:,}")
print(f"\nDistribución de longitudes:")
print(f"  Media:   {lengths.mean():.2f}")
print(f"  Mediana: {np.median(lengths):.0f}")
print(f"  p75:     {np.percentile(lengths, 75):.0f}")
print(f"  p90:     {np.percentile(lengths, 90):.0f}")
print(f"  p95:     {np.percentile(lengths, 95):.0f}")
print(f"  Máximo:  {lengths.max():.0f}")

print(f"\nConteo por longitud (primeras 20):")
c = Counter(lengths)
for k in sorted(c)[:20]:
    pct  = c[k] / total * 100
    bar  = "█" * int(pct / 0.5)
    print(f"  len={k:3d}: {c[k]:>9,}  ({pct:5.1f}%)  {bar}")

print(f"\n% trayectorias que sobreviven según min_seq_len:")
thresholds = [2, 3, 4, 5, 6, 8, 10]
surviving  = []
for t in thresholds:
    n = int((lengths >= t).sum())
    surviving.append(n)
    print(f"  >= {t:2d}: {n:>9,}  ({n/total*100:.1f}%)")

# ── Gráficos ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Longitudes de trayectorias diarias — {args.data} (train)", fontweight="bold")

# 1. Histograma de longitudes (cap en 30 para legibilidad)
cap = 30
ax = axes[0]
vals = lengths[lengths <= cap]
ax.hist(vals, bins=range(1, cap + 2), color="#378ADD", edgecolor="white", linewidth=0.5)
ax.axvline(np.median(lengths), color="#D85A30", linestyle="--", label=f"Mediana: {np.median(lengths):.0f}")
ax.axvline(lengths.mean(),     color="#E8A838", linestyle="--", label=f"Media: {lengths.mean():.1f}")
ax.set_xlabel("Longitud de trayectoria (POIs/día)")
ax.set_ylabel("Número de trayectorias")
ax.set_title(f"Distribución (mostrando ≤{cap}, {(lengths<=cap).mean()*100:.0f}% del total)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# 2. % supervivencia según umbral
ax = axes[1]
pcts = [s / total * 100 for s in surviving]
bars = ax.bar([str(t) for t in thresholds], pcts, color="#378ADD", edgecolor="white")
for bar, pct, n in zip(bars, pcts, surviving):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{pct:.0f}%\n({n/1e6:.2f}M)", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("min_seq_len (mínimo POIs distintos por trayectoria)")
ax.set_ylabel("% trayectorias supervivientes")
ax.set_title("Impacto del filtro min_seq_len sobre trayectorias train")
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = os.path.join(PROJECT_DIR, "analysis", "imgs", f"trayectorias_longitud_{args.data.lower()}.png")
Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nGuardado en {out}")