import pandas as pd
import numpy as np
from pathlib import Path

# ── Parámetros ──────────────────────────────────────────────
MIN_SEQ_LEN   = 8      # trayectorias con menos visitas se descartan
TRAIN_RATIO   = 0.7
VAL_RATIO     = 0.1
TEST_RATIO    = 0.2
CITY          = "NYC"
INPUT_CSV     = "dataset_TSMC2014_NYC.csv"
OUTPUT_DIR    = f"/home/gloria/tfg/gen-val-data/tra{TRAIN_RATIO}-val{VAL_RATIO}-test{TEST_RATIO}/min_len_{MIN_SEQ_LEN}/{CITY}"
# ────────────────────────────────────────────────────────────

print("Leyendo CSV...")
df = pd.read_csv(INPUT_CSV)
print(f"  Total de check-ins: {len(df)}")

# 1. Parsear timestamp y extraer hora del día (slot 0-23)
print("Parseando timestamps...")
df['datetime'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S +0000 %Y')
df['hour']     = df['datetime'].dt.hour
df['date']     = df['datetime'].dt.date

# 2. Asignar IDs numéricos consecutivos a los venueIds
# Filtrar: quedarse solo con los top N POIs más visitados
TOP_N_POIS = 1500  # similar al num_locs de STAR para NYC (1341)

print("Filtrando top POIs más visitados...")
top_venues = df['venueId'].value_counts().head(TOP_N_POIS).index
df = df[df['venueId'].isin(top_venues)]
print(f"  Check-ins tras filtrar: {len(df)}")

print("Asignando IDs numéricos a venues...")
unique_venues = df['venueId'].unique()
venue_to_id   = {v: i for i, v in enumerate(unique_venues)}
df['poi_id']  = df['venueId'].map(venue_to_id)
num_locs      = len(unique_venues)
print(f"  Número de localizaciones únicas: {num_locs}")

# 3. Construir gps.npy: coordenadas ordenadas por poi_id
print("Construyendo gps.npy...")
gps_df = df.groupby('poi_id')[['latitude', 'longitude']].first().sort_index()
gps    = gps_df.to_numpy()  # shape: [num_locs, 2]
print(f"  Shape de gps: {gps.shape}")

# 4. Agrupar por usuario y día → trayectorias
print("Construyendo trayectorias por usuario y día...")
df_sorted = df.sort_values(['userId', 'date', 'datetime'])
trajectories = []
for (user, date), group in df_sorted.groupby(['userId', 'date']):
    poi_seq  = group['poi_id'].tolist()
    hour_seq = group['hour'].tolist()
    if len(poi_seq) >= MIN_SEQ_LEN:
        trajectories.append((poi_seq, hour_seq))

print(f"  Trayectorias válidas (>= {MIN_SEQ_LEN} visitas): {len(trajectories)}")

# 5. Split train / val / test
n       = len(trajectories)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

train = trajectories[:n_train]
val   = trajectories[n_train:n_train + n_val]
test  = trajectories[n_train + n_val:]
print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# 6. Guardar ficheros
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def save_sequences(seqs, path_locs, path_times):
    with open(path_locs,  'w') as fl, open(path_times, 'w') as ft:
        for poi_seq, hour_seq in seqs:
            fl.write(' '.join(map(str, poi_seq))  + '\n')
            ft.write(' '.join(map(str, hour_seq)) + '\n')

print("Guardando ficheros...")
save_sequences(train, f"{OUTPUT_DIR}/train.txt",  f"{OUTPUT_DIR}/train_t.txt")
save_sequences(val,   f"{OUTPUT_DIR}/val.txt",    f"{OUTPUT_DIR}/val_t.txt")
save_sequences(test,  f"{OUTPUT_DIR}/test.txt",   f"{OUTPUT_DIR}/test_t.txt")

np.save(f"{OUTPUT_DIR}/gps.npy", gps)

# 7. Guardar también start.npy y train_len.npy que necesita main.py
train_seqs = [s[0] for s in train]
starts     = np.array([s[0] for s in train_seqs])
lengths    = np.array([len(s) for s in train_seqs])
np.save(f"{OUTPUT_DIR}/start.npy",     starts)
np.save(f"{OUTPUT_DIR}/train_len.npy", lengths)

print(f"\n✓ Preprocesamiento completado.")
print(f"  Ficheros guardados en: {OUTPUT_DIR}")
print(f"  num_locs = {num_locs}  (añade esto a params_map en stg_gen.py y main.py)")
print(f"  gps shape = {gps.shape}")