"""
analysis_usuarios_taptap.py
Análisis de usuarios (device_aid) a partir de visits.csv.
Procesa el fichero en chunks para evitar cargarlo entero en memoria.

Uso:
    python analysis/analysis_usuarios_taptap.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

VISITS_PATH = Path("data/raw/taptap/visits.csv")
OUT_DIR     = Path("analysis/imgs/taptap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1_000_000

# ── Una sola pasada: acumulamos las tres métricas ──────────────────────────────
print(f"Procesando {VISITS_PATH} en chunks de {CHUNK_SIZE:,}...")
print("(Una sola pasada para los tres análisis)\n")

# {device_aid: total_visitas}
user_visits   = defaultdict(int)
# {device_aid: set de poi_ids visitados}
user_pois     = defaultdict(set)
# {poi_id: set de device_aids}
poi_users     = defaultdict(set)

total_rows = 0

with open(VISITS_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    chunk  = []

    for row in reader:
        chunk.append(row)
        if len(chunk) >= CHUNK_SIZE:
            for r in chunk:
                aid    = r["device_aid"]
                poi_id = r["poi_id"]
                user_visits[aid]    += 1
                user_pois[aid].add(poi_id)
                poi_users[poi_id].add(aid)
            total_rows += len(chunk)
            chunk = []
            print(f"  {total_rows/1_000_000:.0f}M filas procesadas...", end="\r")

    # chunk final
    for r in chunk:
        aid    = r["device_aid"]
        poi_id = r["poi_id"]
        user_visits[aid]    += 1
        user_pois[aid].add(poi_id)
        poi_users[poi_id].add(aid)
    total_rows += len(chunk)

print(f"\n  Total filas procesadas: {total_rows:,}")

# ── Convertir a arrays ─────────────────────────────────────────────────────────
visits_arr   = np.array(list(user_visits.values()))
traj_arr     = np.array([len(v) for v in user_pois.values()])
poi_users_arr= np.array([len(v) for v in poi_users.values()])

# ── 1. USUARIOS Y ACTIVIDAD ────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== USUARIOS Y ACTIVIDAD ===")
print("="*60)
print(f"Usuarios únicos (device_aid): {len(user_visits):,}")
print(f"Visitas totales:              {visits_arr.sum():,}")
print(f"\nVisitas por usuario:")
print(f"  Media:    {visits_arr.mean():.1f}")
print(f"  Mediana:  {np.median(visits_arr):.1f}")
print(f"  Mínimo:   {visits_arr.min()}")
print(f"  Máximo:   {visits_arr.max():,}")
for p in [75, 90, 95, 99]:
    print(f"  p{p}:      {np.percentile(visits_arr, p):.0f}")

print(f"\nDistribución por nivel de actividad:")
for u in [1, 2, 5, 10, 50, 100]:
    n = (visits_arr >= u).sum()
    print(f"  ≥ {u:3d} visitas: {n:8,} usuarios ({n/len(visits_arr)*100:.1f}%)")

# ── 2. TRAYECTORIAS ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== TRAYECTORIAS (POIs distintos por usuario) ===")
print("="*60)
print(f"POIs distintos visitados por usuario:")
print(f"  Media:    {traj_arr.mean():.2f}")
print(f"  Mediana:  {np.median(traj_arr):.1f}")
print(f"  Mínimo:   {traj_arr.min()}")
print(f"  Máximo:   {traj_arr.max():,}")
for p in [75, 90, 95, 99]:
    print(f"  p{p}:      {np.percentile(traj_arr, p):.0f}")

print(f"\nDistribución de trayectorias:")
for n in [1, 2, 3, 5, 10, 20]:
    cnt = (traj_arr >= n).sum()
    print(f"  ≥ {n:2d} POIs distintos: {cnt:8,} usuarios ({cnt/len(traj_arr)*100:.1f}%)")

solo1 = (traj_arr == 1).sum()
print(f"\nUsuarios que visitan solo 1 POI: {solo1:,} ({solo1/len(traj_arr)*100:.1f}%) ← no útiles para TTG")

# ── 3. USUARIOS ÚNICOS POR POI ─────────────────────────────────────────────────
print("\n" + "="*60)
print("=== USUARIOS ÚNICOS POR POI ===")
print("="*60)
print(f"Usuarios únicos por POI:")
print(f"  Media:    {poi_users_arr.mean():.1f}")
print(f"  Mediana:  {np.median(poi_users_arr):.1f}")
print(f"  Mínimo:   {poi_users_arr.min()}")
print(f"  Máximo:   {poi_users_arr.max():,}")
for p in [75, 90, 95, 99]:
    print(f"  p{p}:      {np.percentile(poi_users_arr, p):.0f}")

print(f"\nTop 10 POIs por usuarios únicos:")
poi_ids_list   = list(poi_users.keys())
poi_users_list = [len(v) for v in poi_users.values()]
top10_idx = np.argsort(poi_users_list)[::-1][:10]
for rank, idx in enumerate(top10_idx, 1):
    print(f"  #{rank:2d}  {poi_users_list[idx]:8,} usuarios únicos — POI {poi_ids_list[idx]}")

# ── GRÁFICOS ───────────────────────────────────────────────────────────────────
print("\nGenerando gráficos...")
BLUE, RED, GREEN = "#378ADD", "#D85A30", "#1D9E75"

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Análisis de usuarios — TapTap Madrid, Septiembre 2025",
             fontsize=14, fontweight="bold")

# 1. Distribución visitas por usuario (log)
axes[0,0].hist(visits_arr, bins=60, color=BLUE, edgecolor="white", linewidth=0.3)
axes[0,0].axvline(visits_arr.mean(), color=RED, linestyle="--", linewidth=1.5,
                  label=f"Media: {visits_arr.mean():.1f}")
axes[0,0].axvline(np.median(visits_arr), color=GREEN, linestyle="--", linewidth=1.5,
                  label=f"Mediana: {np.median(visits_arr):.0f}")
axes[0,0].set_title("Visitas por usuario")
axes[0,0].set_xlabel("Visitas")
axes[0,0].set_ylabel("Número de usuarios")
axes[0,0].set_yscale("log")
axes[0,0].legend(fontsize=9)

# 2. Percentiles visitas por usuario
percs = [25, 50, 75, 90, 95, 99]
vals  = [np.percentile(visits_arr, p) for p in percs]
colors_p = [GREEN]*3 + ["#BA7517"]*2 + [RED]
bars = axes[0,1].bar([f"p{p}" for p in percs], vals, color=colors_p,
                     edgecolor="white", linewidth=0.5)
axes[0,1].set_yscale("log")
axes[0,1].set_title("Percentiles visitas por usuario")
axes[0,1].set_ylabel("Visitas (log)")
for bar, val in zip(bars, vals):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
                   f"{int(val)}", ha="center", va="bottom", fontsize=9)

# 3. Distribución trayectorias (POIs distintos)
max_traj = min(traj_arr.max(), 50)
axes[0,2].hist(traj_arr[traj_arr <= max_traj], bins=range(1, max_traj+2),
               color=BLUE, edgecolor="white", linewidth=0.3, align="left")
axes[0,2].axvline(traj_arr.mean(), color=RED, linestyle="--", linewidth=1.5,
                  label=f"Media: {traj_arr.mean():.2f}")
axes[0,2].axvline(np.median(traj_arr), color=GREEN, linestyle="--", linewidth=1.5,
                  label=f"Mediana: {np.median(traj_arr):.0f}")
axes[0,2].set_title(f"POIs distintos por usuario (≤{max_traj} mostrados)")
axes[0,2].set_xlabel("POIs distintos visitados")
axes[0,2].set_ylabel("Número de usuarios")
axes[0,2].set_yscale("log")
axes[0,2].legend(fontsize=9)

# 4. % usuarios útiles para TTG según umbral de trayectorias
umbrales_ttg = [1, 2, 3, 5, 10, 15, 20]
pct_utiles = [(traj_arr >= u).sum() / len(traj_arr) * 100 for u in umbrales_ttg]
axes[1,0].plot(umbrales_ttg, pct_utiles, color=BLUE, marker="o", linewidth=2)
axes[1,0].axhline(50, color="gray", linestyle=":", alpha=0.5)
axes[1,0].set_title("Usuarios útiles para TTG según umbral")
axes[1,0].set_xlabel("Mínimo de POIs distintos visitados")
axes[1,0].set_ylabel("% de usuarios")
axes[1,0].set_xticks(umbrales_ttg)
for x, y in zip(umbrales_ttg, pct_utiles):
    axes[1,0].annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=8)

# 5. Usuarios únicos por POI (distribución)
axes[1,1].hist(poi_users_arr, bins=50, color=BLUE, edgecolor="white", linewidth=0.3)
axes[1,1].axvline(poi_users_arr.mean(), color=RED, linestyle="--", linewidth=1.5,
                  label=f"Media: {poi_users_arr.mean():.0f}")
axes[1,1].axvline(np.median(poi_users_arr), color=GREEN, linestyle="--", linewidth=1.5,
                  label=f"Mediana: {np.median(poi_users_arr):.0f}")
axes[1,1].set_title("Usuarios únicos por POI")
axes[1,1].set_xlabel("Usuarios únicos")
axes[1,1].set_ylabel("Número de POIs")
axes[1,1].legend(fontsize=9)

# 6. Visitas vs usuarios únicos por POI
poi_visits_arr = np.array([user_visits.get(uid, 0) 
                            for uid in list(user_visits.keys())[:1000]])  # muestra
axes[1,2].scatter(poi_users_arr, 
                  [total_rows / len(poi_users)] * len(poi_users_arr),
                  s=2, alpha=0.3, color=BLUE)
# Ratio visitas/usuario por POI
poi_ids_l  = list(poi_users.keys())
ratios = []
for pid in poi_ids_l:
    n_vis  = sum(1 for aid in poi_users[pid])  # = len(poi_users[pid])
    ratios.append(len(poi_users[pid]))
axes[1,2].cla()
# Boxplot usuarios únicos por categoría de POI
# (necesitamos mapping poi→categoria, simplificamos con percentiles)
percs2 = [10, 25, 50, 75, 90, 95, 99]
vals2  = [np.percentile(poi_users_arr, p) for p in percs2]
axes[1,2].bar([f"p{p}" for p in percs2], vals2, color=BLUE,
              edgecolor="white", linewidth=0.5)
axes[1,2].set_title("Percentiles usuarios únicos por POI")
axes[1,2].set_ylabel("Usuarios únicos")
for bar, val in zip(axes[1,2].patches, vals2):
    axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                   f"{int(val):,}", ha="center", va="bottom", fontsize=8, rotation=30)

plt.tight_layout()
plt.savefig(OUT_DIR / "taptap_analisis_usuarios.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: {OUT_DIR}/taptap_analisis_usuarios.png")
print("\n¡Análisis de usuarios completado\!")