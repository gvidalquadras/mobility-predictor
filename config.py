"""
config.py
=========
Configuración centralizada del proyecto de predicción de movilidad urbana.

Para cambiar de experimento, modificar CONFIG_NAME:

  Foursquare NYC (validación):
    "config_A_653"        →  653 POIs (≥50 visitas),  granularidad diaria
    "config_B_1500"       → 1500 POIs (top 1500),      granularidad diaria
    "config_C_2413"       → 2413 POIs (≥20 visitas),  granularidad diaria

  TapTap Madrid (producción):
    "config_Madrid_3225"  → 3225 POIs (farmacias + bares + comercios, sept 2025)
                            Granularidad horaria: 720 timesteps (30 días × 24 h)
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Configuración activa  ← CAMBIAR ESTO PARA CADA EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_NAME = "config_Madrid_3225"
# Opciones NYC:     "config_A_653" | "config_B_1500" | "config_C_2413"
# Opciones Madrid:  "config_Madrid_3225"

_NUM_POIS_MAP = {
    "config_A_653"       : 653,
    "config_B_1500"      : 1500,
    "config_C_2413"      : 2413,
    "config_Madrid_3225" : 3225,
}
NUM_POIS = _NUM_POIS_MAP[CONFIG_NAME]

# ─────────────────────────────────────────────────────────────────────────────
# Rutas  (se seleccionan automáticamente según CONFIG_NAME)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

_IS_MADRID = CONFIG_NAME.startswith("config_Madrid")

if _IS_MADRID:
    DATA_CONFIG = {
        "city"        : "Madrid",
        "config_name" : CONFIG_NAME,
        "flow_path"   : os.path.join(PROJECT_DIR, "data", "processed", "flows", "Madrid", "flow.npy"),
        "dates_path"  : os.path.join(PROJECT_DIR, "data", "processed", "flows", "Madrid", "flow_hours.npy"),
        "graphs_dir"  : os.path.join(PROJECT_DIR, "data", "graphs", "Madrid"),
        "traj_dir"    : os.path.join(PROJECT_DIR, "data", "processed", "tra0.7-val0.1-test0.2", "Madrid"),
        "output_dir"  : os.path.join(PROJECT_DIR, "checkpoints", CONFIG_NAME),
        "logs_dir"    : os.path.join(PROJECT_DIR, "logs", CONFIG_NAME),
    }
else:
    DATA_CONFIG = {
        "city"        : "NYC",
        "config_name" : CONFIG_NAME,
        "flow_path"   : os.path.join(PROJECT_DIR, "data", "processed", "flows", "NYC", CONFIG_NAME, "flow.npy"),
        "dates_path"  : os.path.join(PROJECT_DIR, "data", "processed", "flows", "NYC", CONFIG_NAME, "flow_dates.npy"),
        "graphs_dir"  : os.path.join(PROJECT_DIR, "data", "graphs", "NYC", CONFIG_NAME),
        "traj_dir"    : os.path.join(PROJECT_DIR, "data", "processed",
                                     "tra0.7-val0.1-test0.2", "min_len_8", "NYC", CONFIG_NAME),
        "output_dir"  : os.path.join(PROJECT_DIR, "checkpoints", CONFIG_NAME),
        "logs_dir"    : os.path.join(PROJECT_DIR, "logs", CONFIG_NAME),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────────────────────
if _IS_MADRID:
    # Granularidad horaria: 720 timesteps (30 días × 24 h)
    # Ventana de entrada: 7 días (168 h) → predicción: 3 días (72 h)
    DATA = {
        "num_nodes"   : NUM_POIS,
        "history_len" : 168,   # 7 días × 24 h
        "horizon"     : 72,    # 3 días × 24 h
        "train_ratio" : 0.7,
        "val_ratio"   : 0.1,
        "test_ratio"  : 0.2,
    }
else:
    # Granularidad diaria: 251 timesteps
    DATA = {
        "num_nodes"   : NUM_POIS,
        "history_len" : 14,
        "horizon"     : 7,
        "train_ratio" : 0.7,
        "val_ratio"   : 0.1,
        "test_ratio"  : 0.2,
    }

# ─────────────────────────────────────────────────────────────────────────────
# STAREmbedding
# ─────────────────────────────────────────────────────────────────────────────
STAR = {
    "fea_dim"   : 32,
    "hid_dim"   : 32,
    "out_dim"   : 32,
    "layer_num" : 2,
    "head_num"  : 2,
}

# ─────────────────────────────────────────────────────────────────────────────
# GraphWaveNet
# ─────────────────────────────────────────────────────────────────────────────
GWN = {
    "in_dim"            : 1 + STAR["out_dim"],   # 1 flujo + 32 embedding = 33
    "out_dim"           : DATA["horizon"],
    "residual_channels" : 32,
    "dilation_channels" : 32,
    "skip_channels"     : 256,
    "end_channels"      : 512,
    "kernel_size"       : 2,
    "blocks"            : 4,
    "layers"            : 2,
    "dropout"           : 0.3,
}

# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
TRAIN = {
    "device"         : "cuda:0",
    "epochs"         : 100,
    "batch_size"     : 4,
    "lr"             : 0.001,
    "lr_decay"       : 0.97,
    "weight_decay"   : 0.0001,
    "grad_clip"      : 5.0,
    "patience"       : 15,
    "save_best"      : True,
}