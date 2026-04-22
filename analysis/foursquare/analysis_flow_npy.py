import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 7. ANÁLISIS DE FLOW.NPY
# =============================================================================

flow = np.load('data/processed/flows/NYC/flow.npy')  # [252, 1500]

print("=" * 60)
print("=== ANÁLISIS DE FLOW.NPY ===")
print("=" * 60)
print(f"Shape:        {flow.shape}  → [días, POIs]")
print(f"Total celdas: {flow.size}")
print(f"\n--- Estadísticas globales ---")
print(f"Media:        {flow.mean():.4f}")
print(f"Mediana:      {np.median(flow):.4f}")
print(f"Std:          {flow.std():.4f}")
print(f"Mínimo:       {flow.min():.4f}")
print(f"Máximo:       {flow.max():.4f}")

print(f"\n--- Ceros ---")
n_zeros = (flow == 0).sum()
print(f"Celdas con valor 0:     {n_zeros} ({n_zeros/flow.size*100:.1f}%)")
print(f"Celdas con valor > 0:   {flow.size - n_zeros} ({(flow.size-n_zeros)/flow.size*100:.1f}%)")

print(f"\n--- Distribución de valores no cero ---")
nonzero = flow[flow > 0]
print(f"Media (no cero):    {nonzero.mean():.4f}")
print(f"Mediana (no cero):  {np.median(nonzero):.4f}")
print(f"Std (no cero):      {nonzero.std():.4f}")
print(f"Máximo:             {nonzero.max():.4f}")
for p in [75, 90, 95, 99]:
    print(f"  p{p}: {np.percentile(nonzero, p):.4f}")

print(f"\n--- POIs por nivel de actividad ---")
flujo_medio_poi = flow.mean(axis=0)  # media por POI a lo largo del tiempo
print(f"POIs con flujo medio = 0:      {(flujo_medio_poi == 0).sum()}")
print(f"POIs con flujo medio < 0.1:    {(flujo_medio_poi < 0.1).sum()}")
print(f"POIs con flujo medio 0.1-0.5:  {((flujo_medio_poi >= 0.1) & (flujo_medio_poi < 0.5)).sum()}")
print(f"POIs con flujo medio 0.5-1:    {((flujo_medio_poi >= 0.5) & (flujo_medio_poi < 1)).sum()}")
print(f"POIs con flujo medio > 1:      {(flujo_medio_poi >= 1).sum()}")

print(f"\n--- Variabilidad temporal ---")
flujo_medio_dia = flow.mean(axis=1)  # media por día a lo largo de los POIs
print(f"Media diaria:   {flujo_medio_dia.mean():.4f}")
print(f"Std diaria:     {flujo_medio_dia.std():.4f}")
print(f"Mínimo diario:  {flujo_medio_dia.min():.4f}")
print(f"Máximo diario:  {flujo_medio_dia.max():.4f}")

# =============================================================================
# GRÁFICOS
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis de flow.npy — Foursquare NYC (1500 POIs)', 
             fontsize=14, fontweight='bold')

# 1. Distribución de valores no cero
axes[0, 0].hist(nonzero, bins=50, color='#378ADD', edgecolor='white', linewidth=0.5)
axes[0, 0].axvline(nonzero.mean(), color='#D85A30', linestyle='--', linewidth=1.5,
                   label=f'Media: {nonzero.mean():.2f}')
axes[0, 0].axvline(np.median(nonzero), color='#1D9E75', linestyle='--', linewidth=1.5,
                   label=f'Mediana: {np.median(nonzero):.2f}')
axes[0, 0].set_title('Distribución de flujos (valores > 0)')
axes[0, 0].set_xlabel('Flujo')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].legend(fontsize=9)

# 2. Flujo medio por POI (ordenado)
flujo_sorted = np.sort(flujo_medio_poi)[::-1]
axes[0, 1].plot(flujo_sorted, color='#378ADD', linewidth=1)
axes[0, 1].axhline(flujo_medio_poi.mean(), color='#D85A30', linestyle='--',
                   linewidth=1.5, label=f'Media: {flujo_medio_poi.mean():.3f}')
axes[0, 1].set_title('Flujo medio por POI (ordenado)')
axes[0, 1].set_xlabel('POI (ordenado por flujo)')
axes[0, 1].set_ylabel('Flujo medio')
axes[0, 1].legend(fontsize=9)

# 3. Serie temporal del flujo medio diario
axes[1, 0].plot(range(len(flujo_medio_dia)), flujo_medio_dia,
                color='#378ADD', linewidth=0.8, alpha=0.8)
axes[1, 0].axhline(flujo_medio_dia.mean(), color='#D85A30', linestyle='--',
                   linewidth=1.5, label=f'Media: {flujo_medio_dia.mean():.4f}')
axes[1, 0].set_title('Flujo medio diario (promedio sobre POIs)')
axes[1, 0].set_xlabel('Día (0-251)')
axes[1, 0].set_ylabel('Flujo medio')
axes[1, 0].legend(fontsize=9)

# 4. Heatmap (muestra de 100 POIs más activos x todos los días)
top100_idx = np.argsort(flujo_medio_poi)[::-1][:100]
heatmap_data = flow[:, top100_idx].T  # [100, 252]
im = axes[1, 1].imshow(heatmap_data, aspect='auto', cmap='YlOrRd',
                        interpolation='nearest')
axes[1, 1].set_title('Heatmap: top 100 POIs × 252 días')
axes[1, 1].set_xlabel('Día (0-251)')
axes[1, 1].set_ylabel('POI (ordenado por actividad)')
plt.colorbar(im, ax=axes[1, 1], label='Flujo')

plt.tight_layout()
plt.savefig('analysis/imgs/analisis_flow.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado en analysis/imgs/analisis_flow.png")