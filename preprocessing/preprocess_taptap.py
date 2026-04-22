"""
preprocess_taptap.py
====================
Genera los ficheros procesados para el pipeline ST-GNN a partir de los
datos de TapTap Digital (Predicio + POIs Madrid).

Equivalente a aggregate_flows.py + preprocess_foursquare.py para Foursquare,
pero adaptado a granularidad horaria y al formato de datos TapTap.

Uso:
    python preprocessing/preprocess_taptap.py

Salida (en data/processed/flows/Madrid/):
    flow.npy        → matriz de flujos [720 horas, 3225 POIs] float32
    flow_hours.npy  → array de timestamps (strings) [720]
    poi_to_idx.npy  → mapping poi_id (str) → índice numérico [3225, 2]
    gps.npy         → coordenadas GPS de cada POI [3225, 2] (lat, lon)

Salida (en data/processed/tra0.7-val0.1-test0.2/Madrid/):
    train.txt, train_t.txt  → trayectorias de train (poi_id, hora)
    val.txt,   val_t.txt    → trayectorias de val
    test.txt,  test_t.txt   → trayectorias de test
    gps.npy                 → coordenadas GPS (copia)
"""

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Rutas ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)

FLOWS_CSV    = os.path.join(PROJECT_DIR, "data", "raw", "taptap", "flows_hourly.csv")
POIS_CSV     = os.path.join(PROJECT_DIR, "data", "raw", "taptap", "source_poibrandsesp_tags_202604170958.csv")
VISITS_CSV   = os.path.join(PROJECT_DIR, "data", "raw", "taptap", "visits.csv")

FLOW_OUT_DIR = os.path.join(PROJECT_DIR, "data", "processed", "flows", "Madrid")
TRAJ_OUT_DIR = os.path.join(PROJECT_DIR, "data", "processed", "tra0.7-val0.1-test0.2", "Madrid")

TRAIN_RATIO  = 0.7
VAL_RATIO    = 0.1
TEST_RATIO   = 0.2
MIN_SEQ_LEN  = 2   # mínimo de POIs distintos para trayectoria útil (TTG)

Path(FLOW_OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(TRAJ_OUT_DIR).mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 1: flow.npy — matriz de flujos horarios [720, 3225]
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PARTE 1: Construyendo flow.npy")
print("=" * 60)

print("\nCargando POIs...")
pois_df = pd.read_csv(POIS_CSV)
pois_df["id"] = pois_df["id"].astype(str)
all_pois = pois_df["id"].tolist()

# Mapping poi_id (str) → índice numérico
poi_to_idx = {pid: i for i, pid in enumerate(all_pois)}
num_pois   = len(all_pois)
print(f"  {num_pois} POIs cargados")

print("\nCargando flows_hourly.csv...")
flows_df = pd.read_csv(FLOWS_CSV)
flows_df["hour"]   = pd.to_datetime(flows_df["hour"])
flows_df["poi_id"] = flows_df["poi_id"].astype(str)
print(f"  {len(flows_df):,} entradas de flujo")

# Construir índice temporal: 720 horas de sept 2025
all_hours  = pd.date_range("2025-09-01 00:00", "2025-09-30 23:00", freq="h")
hour_to_idx = {h: i for i, h in enumerate(all_hours)}
num_hours  = len(all_hours)
print(f"  {num_hours} timesteps horarios")

# Construir matriz [720, 3225]
print("\nConstruyendo matriz de flujos...")
flow = np.zeros((num_hours, num_pois), dtype=np.float32)

for _, row in flows_df.iterrows():
    h_idx = hour_to_idx.get(row["hour"])
    p_idx = poi_to_idx.get(row["poi_id"])
    if h_idx is not None and p_idx is not None:
        flow[h_idx, p_idx] = row["visit_count"]

print(f"  Shape: {flow.shape}")
print(f"  Flujo medio:         {flow.mean():.2f}")
print(f"  Flujo máximo:        {flow.max():.0f}")
print(f"  Porcentaje de ceros: {(flow == 0).mean()*100:.1f}%")

# Guardar
np.save(os.path.join(FLOW_OUT_DIR, "flow.npy"), flow)
np.save(os.path.join(FLOW_OUT_DIR, "flow_hours.npy"),
        np.array([str(h) for h in all_hours]))
np.save(os.path.join(FLOW_OUT_DIR, "poi_to_idx.npy"),
        np.array(list(poi_to_idx.items())))
print(f"\n✓ flow.npy guardado en {FLOW_OUT_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 2: gps.npy — coordenadas GPS de cada POI [3225, 2]
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PARTE 2: Construyendo gps.npy")
print("=" * 60)

gps = np.zeros((num_pois, 2), dtype=np.float32)
for _, row in pois_df.iterrows():
    idx = poi_to_idx.get(row["id"])
    if idx is not None:
        gps[idx, 0] = float(row["latitude"])
        gps[idx, 1] = float(row["longitude"])

print(f"  Shape: {gps.shape}")
print(f"  Lat: {gps[:,0].min():.4f} → {gps[:,0].max():.4f}")
print(f"  Lon: {gps[:,1].min():.4f} → {gps[:,1].max():.4f}")

np.save(os.path.join(FLOW_OUT_DIR, "gps.npy"), gps)
np.save(os.path.join(TRAJ_OUT_DIR, "gps.npy"), gps)
print(f"✓ gps.npy guardado")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 3: trayectorias para STAR (train/val/test)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PARTE 3: Construyendo trayectorias para STAR")
print("=" * 60)

print(f"\nLeyendo visits.csv en chunks (fichero grande)...")
# Agrupamos visitas por (device_aid, día) ordenadas por timestamp
# → cada grupo es una trayectoria de usuario en un día

# Split temporal de días para train/val/test
# Train: días 1-21 | Val: días 22-24 | Test: días 25-30
train_days = set(range(1, 22))   # sept 1-21
val_days   = set(range(22, 25))  # sept 22-24
test_days  = set(range(25, 31))  # sept 25-30

train_trajs, val_trajs, test_trajs = [], [], []

CHUNK = 2_000_000
user_day_visits = defaultdict(list)  # {(device_aid, day): [(ts, poi_idx)]}

total = 0
with open(VISITS_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    chunk  = []
    for row in reader:
        chunk.append(row)
        if len(chunk) >= CHUNK:
            for r in chunk:
                poi_idx = poi_to_idx.get(r["poi_id"])
                if poi_idx is None:
                    continue
                ts  = int(r["timestamp"])
                day = pd.Timestamp(ts, unit="s", tz="UTC").day
                user_day_visits[(r["device_aid"], day)].append((ts, poi_idx))
            total += len(chunk)
            chunk = []
            print(f"  {total/1_000_000:.0f}M visitas procesadas...", end="\r")
    # chunk final
    for r in chunk:
        poi_idx = poi_to_idx.get(r["poi_id"])
        if poi_idx is None:
            continue
        ts  = int(r["timestamp"])
        day = pd.Timestamp(ts, unit="s", tz="UTC").day
        user_day_visits[(r["device_aid"], day)].append((ts, poi_idx))
    total += len(chunk)

print(f"\n  Total visitas procesadas: {total:,}")
print(f"  Trayectorias (usuario, día): {len(user_day_visits):,}")

# Construir secuencias y asignar a split
for (user, day), visits in user_day_visits.items():
    visits_sorted = sorted(visits, key=lambda x: x[0])
    poi_seq  = [v[1] for v in visits_sorted]
    hour_seq = [pd.Timestamp(v[0], unit="s", tz="UTC").hour for v in visits_sorted]

    # Filtrar trayectorias con >= MIN_SEQ_LEN POIs distintos
    if len(set(poi_seq)) < MIN_SEQ_LEN:
        continue

    traj = (poi_seq, hour_seq)
    if day in train_days:
        train_trajs.append(traj)
    elif day in val_days:
        val_trajs.append(traj)
    elif day in test_days:
        test_trajs.append(traj)

print(f"\n  Trayectorias válidas (≥{MIN_SEQ_LEN} POIs distintos):")
print(f"    Train: {len(train_trajs):,}")
print(f"    Val:   {len(val_trajs):,}")
print(f"    Test:  {len(test_trajs):,}")

def save_trajectories(trajs, path_locs, path_times):
    with open(path_locs, "w") as fl, open(path_times, "w") as ft:
        for poi_seq, hour_seq in trajs:
            fl.write(" ".join(map(str, poi_seq))  + "\n")
            ft.write(" ".join(map(str, hour_seq)) + "\n")

print("\nGuardando trayectorias...")
save_trajectories(train_trajs, f"{TRAJ_OUT_DIR}/train.txt", f"{TRAJ_OUT_DIR}/train_t.txt")
save_trajectories(val_trajs,   f"{TRAJ_OUT_DIR}/val.txt",   f"{TRAJ_OUT_DIR}/val_t.txt")
save_trajectories(test_trajs,  f"{TRAJ_OUT_DIR}/test.txt",  f"{TRAJ_OUT_DIR}/test_t.txt")

# start.npy y train_len.npy para STAREmbedding
train_seqs = [t[0] for t in train_trajs]
np.save(f"{TRAJ_OUT_DIR}/start.npy",     np.array([s[0]    for s in train_seqs]))
np.save(f"{TRAJ_OUT_DIR}/train_len.npy", np.array([len(s)  for s in train_seqs]))

print(f"\n{'='*60}")
print("✓ PREPROCESAMIENTO COMPLETADO")
print(f"{'='*60}")
print(f"  flow.npy:    {flow.shape}  →  {FLOW_OUT_DIR}")
print(f"  gps.npy:     {gps.shape}")
print(f"  Trayectorias → {TRAJ_OUT_DIR}")
print(f"    train: {len(train_trajs):,} | val: {len(val_trajs):,} | test: {len(test_trajs):,}")