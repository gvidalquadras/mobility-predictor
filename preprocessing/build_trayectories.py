"""
build_trajectories.py
=====================
Construye las trayectorias para STAR a partir de visits_sorted.csv
(visits.csv ordenado por device_aid con Unix sort).

Procesa el fichero en streaming — un usuario a la vez — sin acumular
todo en RAM.

Uso:
    python preprocessing/build_trajectories.py

Salida (en data/processed/tra0.7-val0.1-test0.2/Madrid/):
    train.txt, train_t.txt
    val.txt,   val_t.txt
    test.txt,  test_t.txt
    start.npy, train_len.npy
"""

import csv
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Rutas ──────────────────────────────────────────────────────────────────────
import os
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

SORTED_CSV  = os.path.join(PROJECT_DIR, "data", "raw", "taptap", "visits_sorted.csv")
POIS_CSV    = os.path.join(PROJECT_DIR, "data", "raw", "taptap", "source_poibrandsesp_tags_202604170958.csv")
OUT_DIR     = os.path.join(PROJECT_DIR, "data", "processed", "tra0.7-val0.1-test0.2", "Madrid")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

MIN_SEQ_LEN = 2  # mínimo de POIs distintos por trayectoria

# Splits temporales (por día de septiembre)
TRAIN_DAYS = set(range(1, 22))   # días 1-21  → train
VAL_DAYS   = set(range(22, 25))  # días 22-24 → val
TEST_DAYS  = set(range(25, 31))  # días 25-30 → test

# ── Cargar mapping poi_id → índice ────────────────────────────────────────────
print("Cargando POIs...")
import pandas
pois_df = pandas.read_csv(POIS_CSV)
pois_df["id"] = pois_df["id"].astype(str)
poi_to_idx = {pid: i for i, pid in enumerate(pois_df["id"].tolist())}
print(f"  {len(poi_to_idx)} POIs")

# ── Procesar en streaming ─────────────────────────────────────────────────────
print(f"\nProcesando {SORTED_CSV} en streaming...")

train_out  = open(f"{OUT_DIR}/train.txt",   "w")
train_t    = open(f"{OUT_DIR}/train_t.txt", "w")
val_out    = open(f"{OUT_DIR}/val.txt",     "w")
val_t      = open(f"{OUT_DIR}/val_t.txt",   "w")
test_out   = open(f"{OUT_DIR}/test.txt",    "w")
test_t_out = open(f"{OUT_DIR}/test_t.txt",  "w")

files = {
    "train": (train_out, train_t),
    "val":   (val_out,   val_t),
    "test":  (test_out,  test_t_out),
}

train_seqs_meta = []  # para start.npy y train_len.npy

total_rows  = 0
total_trajs = {"train": 0, "val": 0, "test": 0}

def flush_user(user_days):
    """Procesa todas las visitas de un usuario y escribe trayectorias válidas."""
    for day, visits in user_days.items():
        if len(visits) == 0:
            continue
        # Ordenar por timestamp
        visits.sort(key=lambda x: x[0])
        poi_seq  = [v[1] for v in visits]
        hour_seq = [v[2] for v in visits]

        # Filtrar por POIs distintos
        if len(set(poi_seq)) < MIN_SEQ_LEN:
            continue

        # Asignar a split
        if day in TRAIN_DAYS:
            split = "train"
        elif day in VAL_DAYS:
            split = "val"
        elif day in TEST_DAYS:
            split = "test"
        else:
            continue

        fl, ft = files[split]
        fl.write(" ".join(map(str, poi_seq))  + "\n")
        ft.write(" ".join(map(str, hour_seq)) + "\n")
        total_trajs[split] += 1

        if split == "train":
            train_seqs_meta.append((poi_seq[0], len(poi_seq)))

current_user = None
current_days = defaultdict(list)  # {day: [(ts, poi_idx, hour)]}

with open(SORTED_CSV, newline="", encoding="utf-8") as f:
    fieldnames = ["device_aid", "poi_id", "poi_name", "poi_category", "timestamp", "hour"]
    reader = csv.DictReader(f, fieldnames=fieldnames)
    next(reader, None)  # saltar cualquier fila de cabecera que haya quedado
    for row in reader:
        total_rows += 1
        if total_rows % 5_000_000 == 0:
            n = sum(total_trajs.values())
            print(f"  {total_rows/1_000_000:.0f}M filas | {n:,} trayectorias", end="\r")

        aid    = row["device_aid"]
        poi_id = row["poi_id"]
        poi_idx = poi_to_idx.get(poi_id)
        if poi_idx is None:
            continue

        ts  = int(row["timestamp"])
        dt  = pd.Timestamp(ts, unit="s", tz="UTC")
        day = dt.day
        hour = dt.hour

        # Nuevo usuario → procesar el anterior
        if aid != current_user:
            if current_user is not None:
                flush_user(current_days)
            current_user = aid
            current_days = defaultdict(list)

        current_days[day].append((ts, poi_idx, hour))

# Último usuario
if current_user is not None:
    flush_user(current_days)

# Cerrar ficheros
for fl, ft in files.values():
    fl.close()
    ft.close()

print(f"\n  Total filas procesadas: {total_rows:,}")
print(f"\n  Trayectorias válidas (≥{MIN_SEQ_LEN} POIs distintos):")
print(f"    Train: {total_trajs['train']:,}")
print(f"    Val:   {total_trajs['val']:,}")
print(f"    Test:  {total_trajs['test']:,}")

# ── start.npy y train_len.npy ──────────────────────────────────────────────────
print("\nGuardando start.npy y train_len.npy...")
np.save(f"{OUT_DIR}/start.npy",     np.array([m[0] for m in train_seqs_meta]))
np.save(f"{OUT_DIR}/train_len.npy", np.array([m[1] for m in train_seqs_meta]))

print(f"\n✓ Trayectorias guardadas en {OUT_DIR}")

