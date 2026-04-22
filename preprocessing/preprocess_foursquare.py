"""
preprocess_foursquare.py
========================
Genera las trayectorias y coordenadas GPS para STAR.

Uso:
    python preprocessing/preprocess_foursquare.py --config config_A_653
    python preprocessing/preprocess_foursquare.py --config config_B_1500
    python preprocessing/preprocess_foursquare.py --config config_C_2413

Salida (en data/processed/tra0.7-val0.1-test0.2/min_len_8/NYC/<config>/):
    train.txt, train_t.txt
    val.txt,   val_t.txt
    test.txt,  test_t.txt
    gps.npy
    start.npy, train_len.npy
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os

# ---------------------
# PARAMETROS FIJOS
# ---------------------

MIN_SEQ_LEN = 8    # longitud mínima de una trayectoría para que sea válida
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
TEST_RATIO  = 0.2
CITY        = "NYC"

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
args   = parser.parse_args()

TOP_N_POIS = _NUM_POIS_MAP[args.config]
OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "data", "processed",
    f"tra{TRAIN_RATIO}-val{VAL_RATIO}-test{TEST_RATIO}",
    f"min_len_{MIN_SEQ_LEN}", CITY, args.config
)

print(f"Configuración: {args.config} ({TOP_N_POIS} POIs)")
print(f"Salida en:     {OUTPUT_DIR}")

# ---------------------
# PROCESAMIENTO
# ---------------------
print("\nLeyendo CSV...")
df = pd.read_csv(INPUT_CSV)
print(f"  Total de check-ins: {len(df)}")

# 1. Parsear timestamp
print("Parseando timestamps...")
df['datetime'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S +0000 %Y')
df['hour']     = df['datetime'].dt.hour
df['date']     = df['datetime'].dt.date

# 2. Eliminar último día (16 feb 2013, día incompleto)
last_day = df['date'].max()
df = df[df['date'] != last_day].copy()
print(f"  Eliminado último día ({last_day}): {len(df)} check-ins restantes")

# 3. Filtrar top N POIs con más visitas
print(f"Filtrando top {TOP_N_POIS} POIs más visitados...")
top_venues = df['venueId'].value_counts().head(TOP_N_POIS).index
df         = df[df['venueId'].isin(top_venues)]
print(f"  Check-ins tras filtrar: {len(df)}")

# 4. Asignar IDs numéricos
print("Asignando IDs numéricos a venues...")
unique_venues = df['venueId'].unique()
venue_to_id   = {v: i for i, v in enumerate(unique_venues)}
df['poi_id']  = df['venueId'].map(venue_to_id)
num_locs      = len(unique_venues)
print(f"  Número de localizaciones únicas: {num_locs}")

# 5. Construir gps.npy
print("Construyendo gps.npy...")
gps_df = df.groupby('poi_id')[['latitude', 'longitude']].first().sort_index()
gps    = gps_df.to_numpy()
print(f"  Shape de gps: {gps.shape}")

# 6. Construir trayectorias
print("Construyendo trayectorias por usuario y día...")
df_sorted    = df.sort_values(['userId', 'date', 'datetime'])
trajectories = []
for (user, date), group in df_sorted.groupby(['userId', 'date']):
    poi_seq  = group['poi_id'].tolist()
    hour_seq = group['hour'].tolist()
    if len(poi_seq) >= MIN_SEQ_LEN:
        trajectories.append((poi_seq, hour_seq))
print(f"  Trayectorias válidas (>= {MIN_SEQ_LEN} visitas): {len(trajectories)}")

# 7. Split train / val / test
n       = len(trajectories)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)
train   = trajectories[:n_train]
val     = trajectories[n_train:n_train + n_val]
test    = trajectories[n_train + n_val:]
print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# 8. Guardar
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def save_sequences(seqs, path_locs, path_times):
    with open(path_locs, 'w') as fl, open(path_times, 'w') as ft:
        for poi_seq, hour_seq in seqs:
            fl.write(' '.join(map(str, poi_seq))  + '\n')
            ft.write(' '.join(map(str, hour_seq)) + '\n')

print("Guardando ficheros...")
save_sequences(train, f"{OUTPUT_DIR}/train.txt",  f"{OUTPUT_DIR}/train_t.txt")
save_sequences(val,   f"{OUTPUT_DIR}/val.txt",    f"{OUTPUT_DIR}/val_t.txt")
save_sequences(test,  f"{OUTPUT_DIR}/test.txt",   f"{OUTPUT_DIR}/test_t.txt")
np.save(f"{OUTPUT_DIR}/gps.npy", gps)

train_seqs = [s[0] for s in train]
starts     = np.array([s[0] for s in train_seqs])
lengths    = np.array([len(s) for s in train_seqs])
np.save(f"{OUTPUT_DIR}/start.npy",     starts)
np.save(f"{OUTPUT_DIR}/train_len.npy", lengths)

print(f"\n✓ Preprocesamiento completado.")
print(f"  Ficheros guardados en: {OUTPUT_DIR}")
print(f"  num_locs = {num_locs}")
print(f"  gps shape = {gps.shape}")