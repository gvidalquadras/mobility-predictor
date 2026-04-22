"""
map_matching.py  (versión GPU)
Cruza los pings filtrados de Predicio con los POIs de TapTap usando PyTorch/CUDA.
Para cada ping, busca si hay un POI a menos de 100m. Si sí, registra la visita.
Agrega visitas por POI y por hora para construir la matriz de flujos.

Uso:
    python preprocessing/map_matching.py

Outputs:
    data/raw/taptap/visits.csv          ← visitas individuales
    data/raw/taptap/flows_hourly.csv    ← flujos agregados por POI y hora
"""

import csv
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# ── Configuración ──────────────────────────────────────────────────────────────
PINGS_PATH  = Path("data/raw/taptap/predicio_madrid_sept2025.csv")
POIS_PATH   = Path("data/raw/taptap/source_poibrandsesp_tags_202604170958.csv")
VISITS_PATH = Path("data/raw/taptap/visits.csv")
FLOWS_PATH  = Path("data/raw/taptap/flows_hourly.csv")

RADIUS_M    = 100.0    # radio de asignación en metros
CHUNK_SIZE  = 200_000  # pings por chunk (ajustar si hay OOM)

# Conversión grados → metros a ~40°N (Madrid)
LAT_M = 111_320.0
LON_M =  85_390.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Cargar POIs en GPU ─────────────────────────────────────────────────────────
print("\nCargando POIs...")
pois = []
with open(POIS_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pois.append({
            "id":       row["id"],
            "name":     row["name"],
            "category": row["tier1_category"],
            "lat":      float(row["latitude"]),
            "lon":      float(row["longitude"]),
        })

poi_ids   = [p["id"]       for p in pois]
poi_names = [p["name"]     for p in pois]
poi_cats  = [p["category"] for p in pois]

# Tensor de POIs en GPU [N_pois, 2] en metros
poi_coords = torch.tensor(
    [[p["lat"] * LAT_M, p["lon"] * LON_M] for p in pois],
    dtype=torch.float32, device=DEVICE
)
print(f"  {len(pois)} POIs cargados en {DEVICE}")

# ── Procesar pings ─────────────────────────────────────────────────────────────
print(f"\nProcesando pings (chunk={CHUNK_SIZE:,}, radio={RADIUS_M}m)...")

VISITS_PATH.parent.mkdir(parents=True, exist_ok=True)

total_pings  = 0
total_visits = 0
flows = {}  # {(poi_id, hour_str): count}

with open(PINGS_PATH, newline="", encoding="utf-8-sig", errors="replace") as ping_f, \
     open(VISITS_PATH, "w", newline="", encoding="utf-8") as vis_f:

    reader = csv.DictReader(ping_f)  # predicio_madrid_sept2025.csv usa comas
    writer = csv.writer(vis_f)
    writer.writerow(["device_aid", "poi_id", "poi_name", "poi_category", "timestamp", "hour"])

    chunk_rows = []
    chunk_lats = []
    chunk_lons = []
    chunk_ts   = []
    chunk_aids = []

    def process_chunk():
        global total_pings, total_visits

        if not chunk_rows:
            return

        # Tensor de pings en GPU [N_chunk, 2] en metros
        pts = torch.tensor(
            [[lat * LAT_M, lon * LON_M] for lat, lon in zip(chunk_lats, chunk_lons)],
            dtype=torch.float32, device=DEVICE
        )  # [N_chunk, 2]

        # Distancias: [N_chunk, N_pois]
        # Broadcasting: pts[:,None,:] - poi_coords[None,:,:]
        diff  = pts[:, None, :] - poi_coords[None, :, :]  # [N_chunk, N_pois, 2]
        dists = torch.sqrt((diff ** 2).sum(dim=2))         # [N_chunk, N_pois]

        # POI más cercano y su distancia
        min_dists, min_idxs = dists.min(dim=1)  # [N_chunk]

        min_dists_cpu = min_dists.cpu().numpy()
        min_idxs_cpu  = min_idxs.cpu().numpy()

        for i in range(len(chunk_rows)):
            total_pings += 1
            if min_dists_cpu[i] <= RADIUS_M:
                idx      = min_idxs_cpu[i]
                ts       = chunk_ts[i]
                dt       = datetime.fromtimestamp(ts, tz=timezone.utc)
                hour_str = dt.strftime("%Y-%m-%d %H:00")
                poi_id   = poi_ids[idx]

                writer.writerow([
                    chunk_aids[i],
                    poi_id,
                    poi_names[idx],
                    poi_cats[idx],
                    ts,
                    hour_str,
                ])

                flows[(poi_id, hour_str)] = flows.get((poi_id, hour_str), 0) + 1
                total_visits += 1

        chunk_rows.clear(); chunk_lats.clear(); chunk_lons.clear()
        chunk_ts.clear();   chunk_aids.clear()

    for row in reader:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            ts  = int(row["timestamp"])
        except (ValueError, KeyError):
            continue

        chunk_rows.append(row)
        chunk_lats.append(lat)
        chunk_lons.append(lon)
        chunk_ts.append(ts)
        chunk_aids.append(row["device_aid"])

        if len(chunk_rows) >= CHUNK_SIZE:
            process_chunk()
            print(f"  {total_pings/1_000_000:.1f}M pings | {total_visits:,} visitas", end="\r")

    process_chunk()  # chunk final

print(f"\n  Total pings:   {total_pings:,}")
print(f"  Total visitas: {total_visits:,}")
print(f"  Tasa asig.:    {100*total_visits/total_pings:.2f}%")

# ── Guardar flujos ─────────────────────────────────────────────────────────────
print(f"\nGuardando flujos en {FLOWS_PATH}...")
with open(FLOWS_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["poi_id", "hour", "visit_count"])
    for (poi_id, hour), count in sorted(flows.items()):
        writer.writerow([poi_id, hour, count])

print(f"  {len(flows):,} entradas guardadas")
print(f"\nOutputs:")
print(f"  {VISITS_PATH}  ({VISITS_PATH.stat().st_size/1e6:.1f} MB)")
print(f"  {FLOWS_PATH}   ({FLOWS_PATH.stat().st_size/1e6:.1f} MB)")
print("\n¡Map-matching completado\!")

