"""
analysis_taptap.py
Análisis exploratorio de los datos de TapTap (Predicio) para la Comunidad de Madrid.
Lee flows_hourly.csv y el CSV de POIs y genera estadísticas y gráficos.

Uso:
    python analysis/analysis_taptap.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────────
FLOWS_PATH = Path("data/raw/taptap/flows_hourly.csv")
POIS_PATH  = Path("data/raw/taptap/source_poibrandsesp_tags_202604170958.csv")
OUT_DIR    = Path("analysis/imgs/taptap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Carga de datos ─────────────────────────────────────────────────────────────
print("Cargando datos...")
flows = pd.read_csv(FLOWS_PATH)
flows["hour"] = pd.to_datetime(flows["hour"])
flows["poi_id"] = flows["poi_id"].astype(str)

pois = pd.read_csv(POIS_PATH)
pois["id"] = pois["id"].astype(str)

# Merge para tener categoría en flows
flows = flows.merge(pois[["id", "name", "tier1_category"]], 
                    left_on="poi_id", right_on="id", how="left")

print(f"  Entradas de flujo:  {len(flows):,}")
print(f"  POIs con datos:     {flows['poi_id'].nunique():,} / {len(pois):,}")
print(f"  Periodo:            {flows['hour'].min()} → {flows['hour'].max()}")

# ── Construir matriz completa [horas × POIs] ───────────────────────────────────
print("\nConstruyendo matriz hora × POI...")
all_hours = pd.date_range("2025-09-01 00:00", "2025-09-30 23:00", freq="h")
all_pois  = pois["id"].tolist()

pivot = flows.pivot_table(index="hour", columns="poi_id",
                          values="visit_count", aggfunc="sum", fill_value=0)
pivot = pivot.reindex(index=all_hours, columns=all_pois, fill_value=0).fillna(0)
matrix = pivot.values.astype(float)  # [720, 3225]
print(f"  Matriz shape: {matrix.shape}  → [horas, POIs]")

# ── 1. RESUMEN GENERAL ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== RESUMEN GENERAL ===")
print("="*60)
n_zeros = (matrix == 0).sum()
print(f"Shape:              {matrix.shape}  → [horas, POIs]")
print(f"Total celdas:       {matrix.size:,}")
print(f"Celdas con valor 0: {n_zeros:,} ({n_zeros/matrix.size*100:.1f}%)")
print(f"Celdas con valor >0:{matrix.size-n_zeros:,} ({(matrix.size-n_zeros)/matrix.size*100:.1f}%)")
print(f"\nVisitas totales:    {matrix.sum():,.0f}")
print(f"Media por celda:    {matrix.mean():.2f}")
nonzero = matrix[matrix > 0]
print(f"Media (no cero):    {nonzero.mean():.2f}")
print(f"Mediana (no cero):  {np.median(nonzero):.2f}")
print(f"Máximo:             {matrix.max():.0f}")

# ── 2. ANÁLISIS DE POIs ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== ANÁLISIS DE POIs ===")
print("="*60)

flujo_medio_poi = matrix.mean(axis=0)  # media horaria por POI
visitas_poi     = matrix.sum(axis=0)   # visitas totales por POI

print(f"POIs con alguna visita:       {(visitas_poi > 0).sum():,}")
print(f"POIs sin ninguna visita:      {(visitas_poi == 0).sum():,}")
print(f"\nVisitas totales por POI:")
print(f"  Media:    {visitas_poi.mean():.0f}")
print(f"  Mediana:  {np.median(visitas_poi):.0f}")
print(f"  Mínimo:   {visitas_poi.min():.0f}")
print(f"  Máximo:   {visitas_poi.max():.0f}")
for p in [75, 90, 95, 99]:
    print(f"  p{p}:      {np.percentile(visitas_poi, p):.0f}")

print(f"\n--- POIs por nivel de actividad (visitas totales sept.) ---")
umbrales = [1, 100, 500, 1000, 5000, 10000]
for u in umbrales:
    n = (visitas_poi >= u).sum()
    print(f"  ≥{u:6d} visitas: {n:4d} POIs ({n/len(all_pois)*100:.1f}%)")

print(f"\n--- Top 10 POIs más visitados ---")
top10_idx = np.argsort(visitas_poi)[::-1][:10]
for rank, idx in enumerate(top10_idx, 1):
    poi_id = all_pois[idx]
    poi_info = pois[pois["id"] == poi_id].iloc[0]
    print(f"  #{rank:2d}  {int(visitas_poi[idx]):8,} visitas — {poi_info['tier1_category']:<15} — {poi_info['name'][:40]}")

# ── 3. ANÁLISIS POR CATEGORÍA ──────────────────────────────────────────────────
print("\n" + "="*60)
print("=== ANÁLISIS POR CATEGORÍA ===")
print("="*60)

cat_stats = flows.groupby("tier1_category").agg(
    num_pois     = ("poi_id", "nunique"),
    total_visitas= ("visit_count", "sum"),
    media_hora   = ("visit_count", "mean"),
).sort_values("total_visitas", ascending=False)
cat_stats["visitas_por_poi"] = (cat_stats["total_visitas"] / cat_stats["num_pois"]).round(0)
print(cat_stats.to_string())

# ── 4. PATRONES TEMPORALES ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== PATRONES TEMPORALES ===")
print("="*60)

flujo_hora = matrix.sum(axis=1)  # visitas totales por hora

# Por hora del día
hora_dia = pd.Series(flujo_hora, index=all_hours)
perfil_horario = hora_dia.groupby(hora_dia.index.hour).mean()
print(f"\nHora con más visitas:  {perfil_horario.idxmax():02d}:00h ({perfil_horario.max():.0f} visitas/hora media)")
print(f"Hora con menos visitas: {perfil_horario.idxmin():02d}:00h ({perfil_horario.min():.0f} visitas/hora media)")

# Por día de la semana
orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
perfil_dow = hora_dia.groupby(hora_dia.index.day_name()).mean().reindex(orden_dias)
print(f"\nVisitas medias por día de la semana:")
for dia, val in perfil_dow.items():
    print(f"  {dia:<12}: {val:,.0f}")

# Por día del mes
flujo_diario = hora_dia.resample("D").sum()
print(f"\nDía con más visitas:   {flujo_diario.idxmax().strftime('%Y-%m-%d')} ({flujo_diario.max():,.0f})")
print(f"Día con menos visitas: {flujo_diario.idxmin().strftime('%Y-%m-%d')} ({flujo_diario.min():,.0f})")

# ── 5. GRÁFICOS ────────────────────────────────────────────────────────────────
print("\nGenerando gráficos...")
COLORS = {"blue": "#378ADD", "red": "#D85A30", "green": "#1D9E75", "orange": "#BA7517"}

# ── Fig 1: Análisis de POIs ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Análisis de POIs — TapTap Madrid, Septiembre 2025", 
             fontsize=14, fontweight="bold")

# 1a. Distribución visitas por POI (no cero)
active_visits = visitas_poi[visitas_poi > 0]
axes[0,0].hist(active_visits, bins=50, color=COLORS["blue"], edgecolor="white", linewidth=0.5)
axes[0,0].axvline(active_visits.mean(), color=COLORS["red"], linestyle="--", linewidth=1.5,
                  label=f"Media: {active_visits.mean():.0f}")
axes[0,0].axvline(np.median(active_visits), color=COLORS["green"], linestyle="--", linewidth=1.5,
                  label=f"Mediana: {np.median(active_visits):.0f}")
axes[0,0].set_title("Distribución visitas totales por POI (activos)")
axes[0,0].set_xlabel("Visitas en septiembre")
axes[0,0].set_ylabel("Número de POIs")
axes[0,0].legend(fontsize=9)

# 1b. Flujo medio horario por POI (ordenado)
flujo_sorted = np.sort(flujo_medio_poi)[::-1]
axes[0,1].plot(flujo_sorted, color=COLORS["blue"], linewidth=1)
axes[0,1].axhline(flujo_medio_poi[flujo_medio_poi>0].mean(), color=COLORS["red"], 
                  linestyle="--", linewidth=1.5,
                  label=f"Media activos: {flujo_medio_poi[flujo_medio_poi>0].mean():.2f}")
axes[0,1].set_title("Flujo medio horario por POI (ordenado)")
axes[0,1].set_xlabel("POI (ordenado por actividad)")
axes[0,1].set_ylabel("Visitas/hora media")
axes[0,1].legend(fontsize=9)

# 1c. Visitas por categoría
cat_total = flows.groupby("tier1_category")["visit_count"].sum().sort_values(ascending=True)
axes[1,0].barh(cat_total.index, cat_total.values, color=COLORS["blue"], 
               edgecolor="white", linewidth=0.5)
axes[1,0].set_title("Visitas totales por categoría")
axes[1,0].set_xlabel("Visitas totales")

# 1d. POIs con datos suficientes según umbral
umbrales_plot = [1, 50, 100, 250, 500, 1000, 2500, 5000]
n_pois_plot = [(visitas_poi >= u).sum() for u in umbrales_plot]
axes[1,1].plot(umbrales_plot, n_pois_plot, color=COLORS["blue"], marker="o", linewidth=2)
axes[1,1].axhline(len(all_pois), color="gray", linestyle=":", alpha=0.5, label=f"Total POIs: {len(all_pois)}")
axes[1,1].set_title("POIs con suficientes visitas según umbral")
axes[1,1].set_xlabel("Umbral mínimo de visitas (sept.)")
axes[1,1].set_ylabel("Número de POIs")
axes[1,1].legend(fontsize=9)
axes[1,1].set_xscale("log")

plt.tight_layout()
plt.savefig(OUT_DIR / "taptap_analisis_pois.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: {OUT_DIR}/taptap_analisis_pois.png")

# ── Fig 2: Patrones temporales ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Patrones temporales — TapTap Madrid, Septiembre 2025",
             fontsize=14, fontweight="bold")

# 2a. Perfil horario (hora del día)
axes[0,0].bar(perfil_horario.index, perfil_horario.values, 
              color=COLORS["blue"], edgecolor="white", linewidth=0.3)
axes[0,0].set_title("Visitas medias por hora del día")
axes[0,0].set_xlabel("Hora")
axes[0,0].set_ylabel("Visitas medias")
axes[0,0].set_xticks(range(0, 24, 2))

# 2b. Perfil día de la semana
colors_dow = [COLORS["orange"] if d in ["Saturday","Sunday"] else COLORS["blue"] 
              for d in orden_dias]
axes[0,1].bar(range(7), perfil_dow.values, color=colors_dow, edgecolor="white", linewidth=0.3)
axes[0,1].set_xticks(range(7))
axes[0,1].set_xticklabels(["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"])
axes[0,1].set_title("Visitas medias por día de la semana")
axes[0,1].set_ylabel("Visitas medias por hora")

# 2c. Serie temporal diaria
axes[1,0].plot(flujo_diario.index, flujo_diario.values, 
               color=COLORS["blue"], linewidth=1, alpha=0.9)
axes[1,0].axhline(flujo_diario.mean(), color=COLORS["red"], linestyle="--", linewidth=1.5,
                  label=f"Media: {flujo_diario.mean():,.0f}")
# Marcar fines de semana
for date in flujo_diario.index:
    if date.weekday() >= 5:
        axes[1,0].axvspan(date, date + pd.Timedelta(days=1), 
                          alpha=0.15, color=COLORS["orange"])
axes[1,0].set_title("Visitas diarias totales (sombreado = fin de semana)")
axes[1,0].set_xlabel("Fecha")
axes[1,0].set_ylabel("Visitas totales")
axes[1,0].legend(fontsize=9)
axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
axes[1,0].xaxis.set_major_locator(mdates.WeekdayLocator())
plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=30)

# 2d. Heatmap hora del día × día de la semana
heatmap_dow = np.zeros((24, 7))
for h in range(24):
    for d in range(7):
        mask = (all_hours.hour == h) & (all_hours.weekday == d)
        heatmap_dow[h, d] = matrix[mask, :].sum(axis=1).mean() if mask.any() else 0

im = axes[1,1].imshow(heatmap_dow, aspect="auto", cmap="YlOrRd", origin="upper")
axes[1,1].set_title("Heatmap: hora del día × día de la semana")
axes[1,1].set_xlabel("Día de la semana")
axes[1,1].set_ylabel("Hora del día")
axes[1,1].set_xticks(range(7))
axes[1,1].set_xticklabels(["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"])
axes[1,1].set_yticks(range(0, 24, 2))
plt.colorbar(im, ax=axes[1,1], label="Visitas medias")

plt.tight_layout()
plt.savefig(OUT_DIR / "taptap_patrones_temporales.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: {OUT_DIR}/taptap_patrones_temporales.png")

# ── Fig 3: Análisis de flujos (estilo analysis_flow_npy) ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Análisis de la matriz de flujos — TapTap Madrid, Septiembre 2025",
             fontsize=14, fontweight="bold")

# 3a. Distribución de valores no cero
axes[0,0].hist(nonzero.flatten(), bins=60, color=COLORS["blue"], 
               edgecolor="white", linewidth=0.3)
axes[0,0].axvline(nonzero.mean(), color=COLORS["red"], linestyle="--", linewidth=1.5,
                  label=f"Media: {nonzero.mean():.1f}")
axes[0,0].axvline(np.median(nonzero), color=COLORS["green"], linestyle="--", linewidth=1.5,
                  label=f"Mediana: {np.median(nonzero):.1f}")
axes[0,0].set_title("Distribución de flujos (valores > 0)")
axes[0,0].set_xlabel("Visitas por hora")
axes[0,0].set_ylabel("Frecuencia")
axes[0,0].legend(fontsize=9)
axes[0,0].set_yscale("log")

# 3b. Flujo medio por POI (ordenado)
flujo_sorted_all = np.sort(flujo_medio_poi)[::-1]
axes[0,1].plot(flujo_sorted_all, color=COLORS["blue"], linewidth=1)
axes[0,1].axhline(flujo_medio_poi.mean(), color=COLORS["red"], linestyle="--", linewidth=1.5,
                  label=f"Media: {flujo_medio_poi.mean():.2f}")
axes[0,1].set_title("Flujo medio horario por POI (ordenado)")
axes[0,1].set_xlabel("POI (ordenado por actividad)")
axes[0,1].set_ylabel("Visitas/hora media")
axes[0,1].legend(fontsize=9)

# 3c. Serie temporal del flujo medio horario
flujo_hora_media = matrix.mean(axis=1)
axes[1,0].plot(range(len(flujo_hora_media)), flujo_hora_media,
               color=COLORS["blue"], linewidth=0.6, alpha=0.8)
axes[1,0].axhline(flujo_hora_media.mean(), color=COLORS["red"], linestyle="--", linewidth=1.5,
                  label=f"Media: {flujo_hora_media.mean():.2f}")
axes[1,0].set_title("Flujo medio por hora (promedio sobre POIs)")
axes[1,0].set_xlabel("Hora (0 = 2025-09-01 00:00)")
axes[1,0].set_ylabel("Visitas medias")
axes[1,0].legend(fontsize=9)

# 3d. Heatmap top 50 POIs × 30 días
flujo_diario_poi = matrix.reshape(30, 24, len(all_pois)).sum(axis=1)  # [30, 3225]
top50_idx = np.argsort(flujo_diario_poi.mean(axis=0))[::-1][:50]
heatmap_data = flujo_diario_poi[:, top50_idx].T  # [50, 30]
im = axes[1,1].imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
axes[1,1].set_title("Heatmap: top 50 POIs × 30 días")
axes[1,1].set_xlabel("Día de septiembre")
axes[1,1].set_ylabel("POI (ordenado por actividad)")
plt.colorbar(im, ax=axes[1,1], label="Visitas diarias")

plt.tight_layout()
plt.savefig(OUT_DIR / "taptap_analisis_flujos.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: {OUT_DIR}/taptap_analisis_flujos.png")

print("\n¡Análisis completado\!")
print(f"Imágenes guardadas en {OUT_DIR}/")
