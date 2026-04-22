import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 6. ANÁLISIS GEOESPACIAL
# =============================================================================

df = pd.read_csv('data/raw/dataset_TSMC2014_NYC.csv')
df['datetime'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S +0000 %Y')

# POIs únicos con sus coordenadas
pois = df.drop_duplicates('venueId')[['venueId', 'latitude', 'longitude', 'venueCategory']].copy()
poi_counts = df['venueId'].value_counts()
pois['visitas'] = pois['venueId'].map(poi_counts)

print("=" * 60)
print("=== ANÁLISIS GEOESPACIAL ===")
print("=" * 60)
print(f"Rango latitud:  {pois['latitude'].min():.4f} → {pois['latitude'].max():.4f}")
print(f"Rango longitud: {pois['longitude'].min():.4f} → {pois['longitude'].max():.4f}")
print(f"\nPOIs fuera del bbox de NYC (posibles errores de coordenadas):")
# NYC bbox aproximado
nyc = pois[
    (pois['latitude']  < 40.4) | (pois['latitude']  > 40.95) |
    (pois['longitude'] < -74.3) | (pois['longitude'] > -73.6)
]
print(f"  {len(nyc)} POIs fuera del bounding box de NYC")
if len(nyc) > 0:
    print(nyc[['venueId', 'latitude', 'longitude', 'venueCategory', 'visitas']].to_string())

# ── Gráfico ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Análisis geoespacial — Foursquare NYC', fontsize=14, fontweight='bold')

# 1. Todos los POIs
axes[0].scatter(pois['longitude'], pois['latitude'],
                s=1, alpha=0.3, color='#378ADD')
axes[0].set_title(f'Todos los POIs ({len(pois):,})')
axes[0].set_xlabel('Longitud')
axes[0].set_ylabel('Latitud')

# 2. Top 1500 POIs (tamaño proporcional a visitas)
top1500 = pois.nlargest(1500, 'visitas')
scatter = axes[1].scatter(top1500['longitude'], top1500['latitude'],
                          s=top1500['visitas']/top1500['visitas'].max()*50,
                          alpha=0.5, color='#D85A30')
axes[1].set_title('Top 1.500 POIs (tamaño ∝ visitas)')
axes[1].set_xlabel('Longitud')
axes[1].set_ylabel('Latitud')

plt.tight_layout()
plt.savefig('analysis/imgs/analisis_geoespacial.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado en analysis/imgs/analisis_geoespacial.png")