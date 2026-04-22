"""
aggregate_flows.py
==================
Genera la serie temporal de flujos agregados por POI y día 
(necesarios para Graph WaveNet) a partir del CSV crudo de Foursquare. 

Uso:
    python preprocessing/aggregate_flows.py --config config_A_653
    python preprocessing/aggregate_flows.py --config config_B_1500
    python preprocessing/aggregate_flows.py --config config_C_2413

Salida (en data/processed/flows/NYC/<config>/):
    flow.npy        → matriz [num_days, num_pois]
    flow_dates.npy  → array de fechas (strings)
    poi_to_idx.npy  → mapping venueId → poi_id
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------
# RUTAS
# ---------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_CSV   = os.path.join(PROJECT_DIR, "data", "raw", "dataset_TSMC2014_NYC.csv")

# ---------------------
# ARGUMENTOS
# ---------------------
_NUM_POIS_MAP = {
    "config_A_653"  : 653,
    "config_B_1500" : 1500,
    "config_C_2413" : 2413,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=list(_NUM_POIS_MAP.keys()),
    help="Configuración de POIs a usar"
)
args = parser.parse_args()

TOP_N_POIS = _NUM_POIS_MAP[args.config]
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "processed", "flows", "NYC", args.config)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"Configuración: {args.config} ({TOP_N_POIS} POIs)")
print(f"Salida en:     {OUTPUT_DIR}")


def main():
    print("\nLeyendo CSV...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Total check-ins: {len(df)}")

    # 1. Parsear timestamp y extraer fecha
    print("Parseando timestamps...")
    df['datetime'] = pd.to_datetime(
        df['utcTimestamp'],
        format='%a %b %d %H:%M:%S +0000 %Y'
    )
    df['date'] = df['datetime'].dt.date
    print(f"  Rango de fechas: {df['date'].min()} -> {df['date'].max()}")

    # 2. Eliminar último día (16 feb 2013, día incompleto)
    last_day = df['date'].max()
    df = df[df['date'] != last_day].copy()
    print(f"  Eliminado último día ({last_day}): {len(df)} check-ins restantes")

    # 3. Filtrar top N POIs
    print(f"Filtrando top {TOP_N_POIS} POIs más visitados...")
    top_venues = df['venueId'].value_counts().head(TOP_N_POIS).index
    df         = df[df['venueId'].isin(top_venues)].copy()
    print(f"  Check-ins tras filtrar: {len(df)}")

    # 4. Asignar poi_id numérico
    print("Asignando IDs numéricos a venues...")
    unique_venues = df['venueId'].unique()
    venue_to_id   = {v: i for i, v in enumerate(unique_venues)}
    df['poi_id']  = df['venueId'].map(venue_to_id)
    num_pois      = len(unique_venues)
    print(f"  Número de POIs: {num_pois}")

    # 5. Construir serie temporal de flujos
    print("Construyendo serie temporal de flujos...")

    # Recoger y ordenar fechas
    all_dates   = sorted(df['date'].unique()) 
    num_days    = len(all_dates)

    # Asignar índice a cada día
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    print(f"  Número de días: {num_days}")

    # Crear matriz vacía
    flow = np.zeros((num_days, num_pois), dtype=np.float32)
    df['day_idx'] = df['date'].map(date_to_idx)

    # Contar check-ins por día
    counts = df.groupby(['day_idx', 'poi_id']).size().reset_index(name='count')

    # Rellenar matriz
    flow[counts['day_idx'].astype(int).values, 
     counts['poi_id'].astype(int).values] = counts['count'].values

    print(f"  Shape flow:              {flow.shape}")
    print(f"  Flujo medio por POI/día: {flow.mean():.4f}")
    print(f"  Flujo máximo:            {flow.max():.0f}")
    print(f"  Porcentaje de ceros:     {(flow == 0).mean()*100:.1f}%")

    # 6. Guardar
    print("\nGuardando ficheros...")
    np.save(os.path.join(OUTPUT_DIR, "flow.npy"),
            flow)
    np.save(os.path.join(OUTPUT_DIR, "flow_dates.npy"),
            np.array([str(d) for d in all_dates]))
    np.save(os.path.join(OUTPUT_DIR, "poi_to_idx.npy"),
            np.array(list(venue_to_id.items())))

    print(f"\n✓ Completado.")
    print(f"  flow.npy:       {flow.shape}")
    print(f"  flow_dates.npy: {num_days} fechas")
    print(f"  poi_to_idx.npy: {num_pois} POIs")


if __name__ == "__main__":
    main()