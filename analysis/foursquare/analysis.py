import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# =============================================================================
# 0. CARGA DE DATOS
# =============================================================================
 
df = pd.read_csv('data/raw/dataset_TSMC2014_NYC.csv')
df['datetime']   = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S +0000 %Y')
df['date']       = df['datetime'].dt.date
df['dayofweek']  = df['datetime'].dt.day_name()
 
 
# =============================================================================
# 1. RESUMEN GENERAL
# =============================================================================
 
print("=" * 60)
print("=== RESUMEN GENERAL ===")
print("=" * 60)
print(f"Check-ins totales:  {len(df)}")
print(f"Usuarios únicos:    {df['userId'].nunique()}")
print(f"POIs únicos:        {df['venueId'].nunique()}")
print(f"Número de días:     {df['date'].nunique()}")
print(f"Fecha de inicio:    {df['date'].min()}")
print(f"Fecha de fin:       {df['date'].max()}")
 
 
# =============================================================================
# 2. ANÁLISIS DE USUARIOS
# =============================================================================
 
print("\n" + "=" * 60)
print("=== ANÁLISIS DE USUARIOS ===")
print("=" * 60)
 
user_counts = df.groupby('userId').size().sort_values(ascending=False)
 
print(user_counts.describe().to_string())
print(f"\nPercentil 90: {user_counts.quantile(0.90):.0f}")
print(f"Percentil 95: {user_counts.quantile(0.95):.0f}")
print(f"Percentil 99: {user_counts.quantile(0.99):.0f}")
 
print(f"\nTop 10 usuarios más activos:")
print(user_counts.head(10).to_string())
 
# ── Gráfico usuarios ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Análisis de usuarios — Foursquare NYC', fontsize=14, fontweight='bold')
 
axes[0].hist(user_counts, bins=50, color='#378ADD', edgecolor='white', linewidth=0.5)
axes[0].set_title('Distribución check-ins por usuario')
axes[0].set_xlabel('Check-ins')
axes[0].set_ylabel('Número de usuarios')
 
axes[1].boxplot(user_counts.values, patch_artist=True,
                boxprops=dict(facecolor='#378ADD', alpha=0.6),
                medianprops=dict(color='#1D9E75', linewidth=2),
                whiskerprops=dict(color='#378ADD'),
                capprops=dict(color='#378ADD'),
                flierprops=dict(marker='o', markerfacecolor='#D85A30', markersize=4))
axes[1].set_title('Boxplot check-ins por usuario')
axes[1].set_xticks([])
 
plt.tight_layout()
# plt.savefig('analisis_usuarios.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado en analisis_usuarios.png")
 
 
# =============================================================================
# 3. ANÁLISIS DE POIs
# =============================================================================
 
print("\n" + "=" * 60)
print("=== ANÁLISIS DE POIs ===")
print("=" * 60)
 
poi_counts = df['venueId'].value_counts()
 
print(poi_counts.describe().to_string())
print(f"\nMedia visitas/POI:   {poi_counts.mean():.1f}")
print(f"Mediana visitas/POI: {poi_counts.median():.1f}")
 
print(f"\n--- Umbrales de calidad ---")
for umbral in [5, 10, 20, 50, 100, 200]:
    n       = (poi_counts >= umbral).sum()
    visitas = poi_counts[poi_counts >= umbral].sum()
    print(f"  ≥{umbral:4d} visitas: {n:5d} POIs → {visitas/len(df)*100:.1f}% de check-ins")
 
print(f"\n--- Concentración por top-N POIs ---")
for n in [10, 100, 500, 1000, 1500, 2000, 3000]:
    visitas = poi_counts.head(n).sum()
    print(f"  Top {n:5d} POIs → {visitas:6d} visitas ({visitas/len(df)*100:.1f}% del total)")
 
print(f"\n--- Top 10 POIs más visitados ---")
for i, (venue_id, count) in enumerate(poi_counts.head(10).items(), 1):
    cat = df[df['venueId'] == venue_id]['venueCategory'].iloc[0]
    print(f"  #{i:2d}  {count:5d} visitas — {cat:<30} (id: {venue_id})")
 
print(f"\n--- Top 10 categorías por número de visitas ---")
cat_stats = df.groupby('venueCategory').agg(
    num_pois    = ('venueId', 'nunique'),
    num_visitas = ('venueId', 'count')
).sort_values('num_visitas', ascending=False)
cat_stats['visitas_por_poi'] = (cat_stats['num_visitas'] / cat_stats['num_pois']).round(1)
cat_stats['pct_visitas']     = (cat_stats['num_visitas'] / len(df) * 100).round(2)
print(cat_stats.head(10).to_string())
print(f"\nTotal categorías únicas: {len(cat_stats)}")
 
# ── Gráfico POIs ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis de POIs — Foursquare NYC', fontsize=14, fontweight='bold')
 
# 1. Histograma de visitas por POI
ax1 = axes[0, 0]
bins   = [1, 2, 6, 11, 21, 51, 101, 201, poi_counts.max()+1]
labels = ['1', '2-5', '6-10', '11-20', '21-50', '51-100', '101-200', '>200']
counts_hist = [((poi_counts >= bins[i]) & (poi_counts < bins[i+1])).sum() for i in range(len(bins)-1)]
ax1.bar(labels, counts_hist, color='#378ADD', edgecolor='white', linewidth=0.5)
ax1.set_title('Distribución visitas por POI')
ax1.set_xlabel('Visitas')
ax1.set_ylabel('Número de POIs')
ax1.tick_params(axis='x', rotation=30)
 
# 2. Percentiles (escala log)
ax2 = axes[0, 1]
percentiles = [25, 50, 75, 90, 95, 99]
values      = [poi_counts.quantile(p/100) for p in percentiles]
colors      = ['#1D9E75','#1D9E75','#1D9E75','#BA7517','#BA7517','#D85A30']
bars = ax2.bar([f'p{p}' for p in percentiles], values, color=colors, edgecolor='white', linewidth=0.5)
ax2.set_yscale('log')
ax2.set_title('Percentiles visitas por POI (escala log)')
ax2.set_ylabel('Visitas (log)')
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
             f'{int(val)}', ha='center', va='bottom', fontsize=9)
 
# 3. Top 10 categorías
ax3 = axes[1, 0]
top_cat    = df.groupby('venueCategory')['venueId'].count().sort_values(ascending=False).head(10)
cat_labels = [c[:20] for c in top_cat.index]
ax3.barh(cat_labels[::-1], top_cat.values[::-1], color='#378ADD', edgecolor='white', linewidth=0.5)
ax3.set_title('Top 10 categorías por visitas')
ax3.set_xlabel('Número de visitas')
 
# 4. Trade-off umbral mínimo de visitas
ax4      = axes[1, 1]
umbrales = [5, 10, 20, 50, 100, 200]
n_pois   = [(poi_counts >= u).sum() for u in umbrales]
pct_vis  = [poi_counts[poi_counts >= u].sum() / len(df) * 100 for u in umbrales]
ax4_twin = ax4.twinx()
ax4.plot(umbrales, n_pois, color='#378ADD', marker='o', linewidth=2, label='POIs')
ax4_twin.plot(umbrales, pct_vis, color='#D85A30', marker='s', linewidth=2,
              linestyle='--', label='% check-ins')
ax4.set_title('POIs y check-ins según umbral mínimo')
ax4.set_xlabel('Umbral mínimo de visitas')
ax4.set_ylabel('Número de POIs', color='#378ADD')
ax4_twin.set_ylabel('% check-ins capturados', color='#D85A30')
ax4.tick_params(axis='y', labelcolor='#378ADD')
ax4_twin.tick_params(axis='y', labelcolor='#D85A30')
ax4.axvline(x=10, color='gray', linestyle=':', alpha=0.7)
ax4.text(10.5, max(n_pois)*0.8, 'config\nactual', fontsize=8, color='gray')
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
 
plt.tight_layout()
# plt.savefig('analisis_pois.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado en analisis_pois.png")
 
 
# =============================================================================
# 4. ANÁLISIS DE EVENTOS POR DÍA
# =============================================================================
 
print("\n" + "=" * 60)
print("=== ANÁLISIS DE EVENTOS POR DÍA ===")
print("=" * 60)
 
eventos_por_dia = df.groupby('date').size()
 
print(f"Total días:  {len(eventos_por_dia)}")
print(f"Media:       {eventos_por_dia.mean():.1f}")
print(f"Mediana:     {eventos_por_dia.median():.1f}")
print(f"Std:         {eventos_por_dia.std():.1f}")
print(f"Mínimo:      {eventos_por_dia.min()}")
print(f"Máximo:      {eventos_por_dia.max()}")
print(f"\nDías con < 100 eventos: {(eventos_por_dia < 100).sum()}")
print(f"Días con < 500 eventos: {(eventos_por_dia < 500).sum()}")
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p:3d}: {eventos_por_dia.quantile(p/100):.0f}")
 
print(f"\n--- Eventos por día de la semana ---")
order      = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_stats  = df.groupby('dayofweek').size().reindex(order)
for day, count in dow_stats.items():
    print(f"  {day:<12}: {count:6d} eventos ({count/len(df)*100:.1f}%)")
 
print(f"\n--- Días anómalos (< 100 eventos) ---")
events_per_day = df.groupby('date').size().reset_index(name='events').sort_values('date')
anomalos       = events_per_day[events_per_day['events'] < 100]
print(anomalos.to_string(index=False))
 
# ── Gráfico eventos por día ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Análisis de eventos por día — Foursquare NYC', fontsize=14, fontweight='bold')
 
# 1. Histograma
axes[0].hist(eventos_por_dia, bins=30, color='#378ADD', edgecolor='white', linewidth=0.5)
axes[0].axvline(eventos_por_dia.mean(),   color='#D85A30', linestyle='--', linewidth=1.5,
                label=f'Media: {eventos_por_dia.mean():.0f}')
axes[0].axvline(eventos_por_dia.median(), color='#1D9E75', linestyle='--', linewidth=1.5,
                label=f'Mediana: {eventos_por_dia.median():.0f}')
axes[0].set_title('Distribución eventos por día')
axes[0].set_xlabel('Eventos')
axes[0].set_ylabel('Número de días')
axes[0].legend(fontsize=9)
 
# 2. Serie temporal
axes[1].plot(range(len(eventos_por_dia)), eventos_por_dia.values,
             color='#378ADD', linewidth=0.8, alpha=0.8)
axes[1].axhline(eventos_por_dia.mean(), color='#D85A30', linestyle='--', linewidth=1,
                label=f'Media: {eventos_por_dia.mean():.0f}')
axes[1].set_title('Eventos por día a lo largo del tiempo')
axes[1].set_xlabel('Día (0-251)')
axes[1].set_ylabel('Número de eventos')
axes[1].legend(fontsize=9)
 
# 3. Boxplot
bp = axes[2].boxplot(eventos_por_dia.values, patch_artist=True,
                     boxprops=dict(facecolor='#378ADD', alpha=0.6),
                     medianprops=dict(color='#1D9E75', linewidth=2),
                     whiskerprops=dict(color='#378ADD'),
                     capprops=dict(color='#378ADD'),
                     flierprops=dict(marker='o', markerfacecolor='#D85A30', markersize=4))
axes[2].set_title('Boxplot eventos por día')
axes[2].set_ylabel('Número de eventos')
axes[2].set_xticks([])
for p, val in [(25, eventos_por_dia.quantile(0.25)),
               (75, eventos_por_dia.quantile(0.75)),
               (99, eventos_por_dia.quantile(0.99))]:
    axes[2].axhline(val, color='gray', linestyle=':', alpha=0.5)
    axes[2].text(1.1, val, f'p{p}: {val:.0f}', fontsize=8, color='gray', va='center')
 
plt.tight_layout()
# plt.savefig('analisis_eventos_dia.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado en analisis_eventos_dia.png")

# =============================================================================
# 5. TRAYECTORIAS ÚTILES PARA CONFIG B (top 1500 POIs)
# =============================================================================

trayectorias = df.groupby(['userId', 'date'])['venueId'].apply(list).reset_index()
trayectorias.columns = ['userId', 'date', 'pois']

poi_counts = df['venueId'].value_counts()
pois_top1500 = set(poi_counts.head(1500).index)

traj_validas_b = trayectorias['pois'].apply(
    lambda x: len(set(x) & pois_top1500) >= 2
).sum()

print(f"\nConfig B (1500 POIs): {traj_validas_b} trayectorias útiles "
      f"({traj_validas_b/len(trayectorias)*100:.1f}%)")


