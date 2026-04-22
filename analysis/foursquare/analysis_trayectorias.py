import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  ANÁLISIS DE TRAYECTORIAS
# =============================================================================

df = pd.read_csv('data/raw/dataset_TSMC2014_NYC.csv')
df['datetime'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S +0000 %Y')
df['date']     = df['datetime'].dt.date

# Agrupar por user y día y tomar todos los pois visitados por el user en ese día (su trayectoria)
trayectorias = df.groupby(['userId', 'date'])['venueId'].apply(list).reset_index()
trayectorias.columns = ['userId', 'date', 'pois']
trayectorias['longitud'] = trayectorias['pois'].apply(len)
trayectorias['pois_unicos'] = trayectorias['pois'].apply(lambda x: len(set(x)))


# --- Estadísticas generales -----------------------------

print("=" * 60)
print("=== ANÁLISIS DE TRAYECTORIAS ===")
print("=" * 60)
print(f"Total trayectorias (usuario-día): {len(trayectorias)}")
print(f"Media trayectorias por usuario:   {len(trayectorias)/df['userId'].nunique():.1f}")
print(f"Media trayectorias por día:       {len(trayectorias)/df['date'].nunique():.1f}")

print(f"\n--- Longitud de trayectorias (check-ins totales/día) ---")
print(trayectorias['longitud'].describe().to_string())

print(f"\n--- POIs únicos por trayectoria ---")
print(trayectorias['pois_unicos'].describe().to_string())

print(f"\nPercentiles (POIs únicos por trayectoria):")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  p{p:3d}: {trayectorias['pois_unicos'].quantile(p/100):.0f}")

print(f"\nTrayectorias con solo 1 POI único: {(trayectorias['pois_unicos'] == 1).sum()} "
      f"({(trayectorias['pois_unicos'] == 1).mean()*100:.1f}%)")
print(f"Trayectorias con 2-5 POIs únicos: {((trayectorias['pois_unicos'] >= 2) & (trayectorias['pois_unicos'] <= 5)).sum()} "
      f"({((trayectorias['pois_unicos'] >= 2) & (trayectorias['pois_unicos'] <= 5)).mean()*100:.1f}%)")
print(f"Trayectorias con >5 POIs únicos:  {(trayectorias['pois_unicos'] > 5).sum()} "
      f"({(trayectorias['pois_unicos'] > 5).mean()*100:.1f}%)")



# --- Relación con umbral de POIs -----------------------------

print(f"\n--- Cobertura de trayectorias según umbral de POIs ---")
poi_counts = df['venueId'].value_counts()
for umbral, n_pois in [(50, 653), (20, 2413), (10, 5135)]:
    pois_validos = set(poi_counts[poi_counts >= umbral].index)
    # Trayectorias con al menos 2 POIs válidos (necesario para construir aristas en TTG)
    traj_validas = trayectorias['pois'].apply(
        lambda x: len(set(x) & pois_validos) >= 2
    ).sum()
    print(f"  ≥{umbral:3d} visitas ({n_pois:5d} POIs): "
          f"{traj_validas} trayectorias útiles para TTG "
          f"({traj_validas/len(trayectorias)*100:.1f}%)")
    

# --- Gráficos -----------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Análisis de trayectorias — Foursquare NYC', fontsize=14, fontweight='bold')

# 1. Distribución de POIs únicos por trayectoria
axes[0].hist(trayectorias['pois_unicos'], bins=30,
             color='#378ADD', edgecolor='white', linewidth=0.5)
axes[0].axvline(trayectorias['pois_unicos'].mean(), color='#D85A30',
                linestyle='--', linewidth=1.5,
                label=f"Media: {trayectorias['pois_unicos'].mean():.1f}")
axes[0].axvline(trayectorias['pois_unicos'].median(), color='#1D9E75',
                linestyle='--', linewidth=1.5,
                label=f"Mediana: {trayectorias['pois_unicos'].median():.0f}")
axes[0].set_title('POIs únicos por trayectoria')
axes[0].set_xlabel('Nº de POIs únicos')
axes[0].set_ylabel('Nº de trayectorias')
axes[0].legend(fontsize=9)

# 2. Distribución de trayectorias por usuario
traj_por_usuario = trayectorias.groupby('userId').size()
axes[1].hist(traj_por_usuario, bins=30,
             color='#378ADD', edgecolor='white', linewidth=0.5)
axes[1].axvline(traj_por_usuario.mean(), color='#D85A30',
                linestyle='--', linewidth=1.5,
                label=f"Media: {traj_por_usuario.mean():.1f}")
axes[1].axvline(traj_por_usuario.median(), color='#1D9E75',
                linestyle='--', linewidth=1.5,
                label=f"Mediana: {traj_por_usuario.median():.0f}")
axes[1].set_title('Trayectorias por usuario')
axes[1].set_xlabel('Nº de días con actividad')
axes[1].set_ylabel('Nº de usuarios')
axes[1].legend(fontsize=9)

# 3. Longitud media de trayectoria por día (evolución temporal)
traj_diaria = trayectorias.groupby('date')['pois_unicos'].mean()
axes[2].plot(range(len(traj_diaria)), traj_diaria.values,
             color='#378ADD', linewidth=0.8, alpha=0.8)
axes[2].axhline(traj_diaria.mean(), color='#D85A30',
                linestyle='--', linewidth=1,
                label=f"Media: {traj_diaria.mean():.1f}")
axes[2].set_title('Longitud media de trayectoria por día')
axes[2].set_xlabel('Día (0-251)')
axes[2].set_ylabel('POIs únicos medios')
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('analysis/imgs/analisis_trayectorias.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado en analisis_trayectorias.png")