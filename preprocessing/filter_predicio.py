"""
filter_predicio.py
Descarga y filtra datos Predicio de S3, quedándose solo con pings
dentro de la Comunidad de Madrid con buena precisión GPS.

Uso:
    python filter_predicio.py

Output:
    data/raw/taptap/predicio_madrid_sept2025.csv
"""

import subprocess
import gzip
import csv
import io
import os
import sys
from pathlib import Path

# ── Configuración ──────────────────────────────────────────────────────────────
S3_BASE     = "s3://predicio.taptapdigital.com/ES/2025/09"
REGION      = "us-east-1"
OUTPUT_PATH = Path("data/raw/taptap/predicio_madrid_sept2025.csv")
TEMP_FILE   = Path("/tmp/predicio_temp.csv.gz")

# Bounding box Comunidad de Madrid
LAT_MIN, LAT_MAX =  40.02,  41.06
LON_MIN, LON_MAX =  -4.50,  -3.10

# Filtro de precisión GPS (metros) — solo pings con error < 100m
MAX_HORIZONTAL_ACCURACY = 100.0

# ── Setup ──────────────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def list_files_for_day(day: str) -> list[str]:
    """Lista los ficheros .csv.gz de un día dado (ej: '01')."""
    result = subprocess.run(
        ["aws", "s3", "ls", f"{S3_BASE}/{day}/", "--region", REGION],
        capture_output=True, text=True
    )
    files = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if parts and parts[-1].endswith(".csv.gz"):
            files.append(parts[-1])
    return files

def download_file(day: str, filename: str) -> bool:
    """Descarga un fichero de S3 a TEMP_FILE."""
    s3_path = f"{S3_BASE}/{day}/{filename}"
    result = subprocess.run(
        ["aws", "s3", "cp", s3_path, str(TEMP_FILE), "--region", REGION],
        capture_output=True, text=True
    )
    return result.returncode == 0

def filter_and_append(writer, is_first_file: bool) -> int:
    """Lee TEMP_FILE, filtra por Madrid + precisión, escribe al CSV de output."""
    count = 0
    with gzip.open(TEMP_FILE, "rt", encoding="utf-8", errors="replace") as gz:
        reader = csv.DictReader(gz, delimiter="\t")
        for row in reader:
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                acc = float(row["horizontal_accuracy"]) if row["horizontal_accuracy"] else 9999.0
            except (ValueError, KeyError):
                continue

            if (LAT_MIN <= lat <= LAT_MAX and
                LON_MIN <= lon <= LON_MAX and
                acc <= MAX_HORIZONTAL_ACCURACY):
                writer.writerow(row)
                count += 1
    return count

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    total_kept = 0
    total_files = 0

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as out_f:
        writer = None
        first_write = True

        for day_num in range(1, 31):
            day = f"{day_num:02d}"
            print(f"\n── Día {day} ──────────────────────────────")

            files = list_files_for_day(day)
            if not files:
                print(f"  Sin ficheros, saltando.")
                continue

            print(f"  {len(files)} ficheros encontrados")

            for fname in files:
                print(f"  Descargando {fname}...", end=" ", flush=True)
                if not download_file(day, fname):
                    print("ERROR al descargar, saltando.")
                    continue

                # Inicializar writer con cabecera del primer fichero
                if writer is None:
                    with gzip.open(TEMP_FILE, "rt", encoding="utf-8", errors="replace") as gz:
                        header_reader = csv.DictReader(gz, delimiter="\t")
                        fieldnames = header_reader.fieldnames
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()

                kept = filter_and_append(writer, first_write)
                first_write = False
                total_kept += kept
                total_files += 1
                TEMP_FILE.unlink(missing_ok=True)
                print(f"{kept:,} pings de Madrid guardados")

    print(f"\n{'='*50}")
    print(f"COMPLETADO")
    print(f"  Ficheros procesados: {total_files}")
    print(f"  Pings totales guardados: {total_kept:,}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Tamaño: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    main()