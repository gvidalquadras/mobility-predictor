import argparse
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import logging
import scipy.sparse.linalg
import pandas as pd
import os
import sys
import socket
from scipy.stats import wasserstein_distance
import numba as nb
from scipy.sparse import csr_matrix, save_npz, load_npz

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import helpers

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuraciones disponibles 
_NUM_POIS_MAP = {
    "config_A_653"  : 653,
    "config_B_1500" : 1500,
    "config_C_2413" : 2413,
    "config_Madrid_3225" : 3225, 
}

# ---------------------
# ARGUMENTOS
# ---------------------

parser = argparse.ArgumentParser()
parser.add_argument('--cuda',       default="0",            type=str)
parser.add_argument('--data',       default='NYC',          type=str)
parser.add_argument('--config',     required=True,          type=str,
                    choices=list(_NUM_POIS_MAP.keys()),
                    help="Configuracion de POIs a usar")
parser.add_argument('--min_seq_len',default=8,              type=int)
parser.add_argument('--max_seq_len',default=24,             type=int)
parser.add_argument('--method',     default='STAR-TKDE',    type=str)
parser.add_argument('--train_ratio',default=0.7,            type=float)
parser.add_argument('--val_ratio',  default=0.1,            type=float)
parser.add_argument('--test_ratio', default=0.2,            type=float)
parser.add_argument('--eigen_dim',  default=32,             type=int)
parser.add_argument('--ratio',      default=0.005,          type=float)
parser.add_argument('--eigen',      action="store_true")
args = parser.parse_args()

# Parámetros derivados 
args.num_locs = _NUM_POIS_MAP[args.config]
args.device   = torch.device("cuda:" + args.cuda)
args.hostname = socket.gethostname()
args.cwd      = PROJECT_DIR

if args.data == 'Madrid':
    args.datapath = (
        f'{PROJECT_DIR}/data/processed/'
        f'tra{args.train_ratio}-val{args.val_ratio}-test{args.test_ratio}/'
        f'{args.data}'
    )
    args.propath = f'{PROJECT_DIR}/data/graphs/{args.data}'
else:
    args.datapath = (
        f'{PROJECT_DIR}/data/processed/'
        f'tra{args.train_ratio}-val{args.val_ratio}-test{args.test_ratio}/'
        f'min_len_{args.min_seq_len}/{args.data}/{args.config}'
    )
    args.propath = f'{PROJECT_DIR}/data/graphs/{args.data}/{args.config}'
Path(args.propath).mkdir(parents=True, exist_ok=True)

print(f"Configuracion: {args.config} ({args.num_locs} POIs)")
print(f"Datos en:      {args.datapath}")
print(f"Grafos en:     {args.propath}")

# ---------------------
# LOGGING
# ---------------------
log_dir    = f'{PROJECT_DIR}/logs/{args.config}'
log_prefix = f'{args.method}-{args.data}-{args.hostname}-gpu{args.cuda}'
Path(log_dir).mkdir(parents=True, exist_ok=True)
logger = helpers.set_logger(log_dir=log_dir, log_prefix=log_prefix)
logger.info(args)

# ---------------------
# CARGAR DATOS
# ---------------------
train_data = helpers.read_data_from_file(f'{args.datapath}/train.txt')
train_t    = helpers.read_data_from_file(f'{args.datapath}/train_t.txt')
gps        = np.load(f'{args.datapath}/gps.npy')
top = max(7, int(args.ratio * args.num_locs)) # Número de vecinos por nodo (máximo 7)


# ---------------------
# TTG
# ---------------------
if os.path.exists(f'{args.propath}/ttg.npz'):
    logger.info('load temporal transition graph...')
    ttg = load_npz(f'{args.propath}/ttg.npz')
else:
    # 1. Inicializar en cero
    ttg_all = np.zeros([args.num_locs, args.num_locs])

    # 2. Recorrer trayectorias y sumar 1 en la matriz 
    #    por cada par de POIs consecutivos
    for seq in train_data:
        for j in range(len(seq)-1):
            ttg_all[seq[j], seq[j+1]] += 1 # fuerza que la diagonal sea al menos 1 (self-loop con peso positivo)
    for i in range(args.num_locs):
        ttg_all[i, i] = max(1, ttg_all[i, i])
    save_npz(f'{args.propath}/ttg_all.npz', csr_matrix(ttg_all))

    # 3. Inicializar matriz
    _ttg         = np.zeros([args.num_locs, args.num_locs])
    _ttg_all     = ttg_all + 1e-10 # evita división por cero

    # 4. Normalizar por filas
    #    (se convierte en probabilidad de transición)
    ttg_all_norm = _ttg_all / _ttg_all.sum(axis=1)[:, None]

    # 5. Filtrado top-k
    for i in range(len(ttg_all_norm)):
        top_id = ttg_all_norm[i, :].argsort()[-top:]
        for id in top_id:
            _ttg[i, id] = ttg_all_norm[i, id]

    # 6. Conversión a formato disperso (ahorra RAM)
    ttg = csr_matrix(_ttg)

    # 7. Guardar
    save_npz(f'{args.propath}/ttg.npz', ttg)
    logger.info(f"save ttg.npz in {args.propath}")

logger.info(f"Sparsity of ttg: {(1.0*len(ttg.data)/(args.num_locs*args.num_locs)):.5f}")

# ---------------------
# SDG
# ---------------------
if os.path.exists(f'{args.propath}/sdg.npz'): 
    logger.info('load spatial distance graph...')
    sdg = load_npz(f'{args.propath}/sdg.npz')
else:
    # 1. Preparación de coordenadas
    latitude, longitude = gps[:, 0], gps[:, 1]
    lgti = np.radians(longitude)[:, np.newaxis]
    lati = np.radians(latitude)[:, np.newaxis]

    # 2. Cálculo de distancias (euclídea)
    @nb.njit(nogil=True, parallel=True)
    def np_distance(lgti, lati):
        distances = np.zeros((len(lgti), len(lgti))).astype(np.float32)
        for i in nb.prange(len(lgti)):
            gi, ai = lgti[i], lati[i]
            dlon   = gi - lgti
            dlat   = ai - lati
            dist   = dlon**2 + dlat**2 
            distances[i] = np.sqrt(np.reshape(dist, (len(dist))))
        return distances

    # 3. Crear y guardar matriz de distancias
    sdg_all = np_distance(lgti, lati)
    np.save(f'{args.propath}/sdg_all.npy', sdg_all)

    # 4. Inicializar matriz
    _sdg             = np.zeros([args.num_locs, args.num_locs])
    _sdg_all         = sdg_all + 1e-10

    # 5. Normalizar distancias
    sdg_all_norm     = _sdg_all / _sdg_all.sum(axis=1, keepdims=True)

    # 6. Transformar de distancia a similitud
    #    (distancia pequeña = gran similitud)
    sdg_all_sim      = 1 / sdg_all_norm
    sdg_all_sim_norm = sdg_all_sim / sdg_all_sim.sum(axis=1, keepdims=True)

    # 7. Filtrado de top vecinos
    for i in range(len(sdg_all_sim_norm)):
        top_id = sdg_all_sim_norm[i, :].argsort()[-top:]
        assert i in top_id
        for id in top_id:
            _sdg[i, id] = sdg_all_sim_norm[i, id]
    
    # 8. Formato disperso
    sdg = csr_matrix(_sdg)
    
    # 9. Guardar
    save_npz(f'{args.propath}/sdg.npz', sdg)
    logger.info(f"save sdg.npz in {args.propath}")

logger.info(f"Sparsity of sdg: {(1.0*len(sdg.data)/(args.num_locs*args.num_locs)):.5f}")

# ---------------------
# STG
# ---------------------
if os.path.exists(f'{args.propath}/stg.npz'):
    logger.info('load spatiotemporal graph...')
    stg = load_npz(f'{args.propath}/stg.npz')
else:
    # 1. Crear matriz donde filas son POIs y columnas franjas de tiempo
    poi_dis = np.zeros((args.num_locs, args.max_seq_len))
    for i in tqdm(range(len(train_data))):
        # Sumar 1 cada vez que el POI ha sido visitado en esa franja
        for poi, t in zip(train_data[i], train_t[i]):
            poi_dis[poi][t] += 1
    # 2. Laplace smoothing
    poi_dis = (poi_dis + 1) / (poi_dis.sum(axis=1, keepdims=True) + poi_dis.shape[1])

    # 3. Similitud usando Wasserstein distance
    _poi_sim = np.zeros((args.num_locs, args.num_locs))
    bins = np.arange(args.max_seq_len)
    for i in tqdm(range(args.num_locs)):
        # solo mitad superior de la matriz 
        # (distancia AB = distancia BA)
        for j in range(i+1, args.num_locs): 
            # (1 - distancia) convierte distancia en similitud
            _poi_sim[i, j] = max(0.0, 1 - wasserstein_distance(bins, bins, poi_dis[i], poi_dis[j]))

    # 4. Rellena la matriz con la mitad inferior, 
    #    haciéndola simétrica
    poi_sim_sym = _poi_sim + _poi_sim.T

    # 5. Sumar 1 a la diagonal 
    #    (cada POI tiene relacion maxima consigo mismo)
    stg_all     = poi_sim_sym + np.identity(args.num_locs)

    np.save(f'{args.propath}/stg_all.npy', stg_all)

    # 6. Inicializar matriz
    _stg = np.zeros((args.num_locs, args.num_locs))

    # 7. Filtrado top vecinos
    for i in range(stg_all.shape[0]):
        top_id = stg_all[i, :].argsort()[-top:]
        assert len(top_id) > 0
        for id in top_id:
            _stg[i, id] = stg_all[i, id]
    
    # 8. Conversión a formato disperso
    stg = csr_matrix(_stg)

    # 9. Guardar
    save_npz(f'{args.propath}/stg.npz', stg)
    logger.info(f"save stg.npz in {args.propath}")

logger.info(f"Sparsity of stg: {(1.0*len(stg.data)/(args.num_locs*args.num_locs)):.5f}")

# ── Guardar nodos y aristas ───────────────────────────────────────────────────
adj_l  = [ttg, sdg, stg]
name_l = ['ttg', 'sdg', 'stg']

for name, adj in zip(name_l, adj_l):

    # Extrae qué nodos están conectados
    e_src, e_dst = adj.nonzero() 
    # Extrae pesos
    weights      = adj.data
    # Crear tabla con columnas [origen, destino, peso]
    e_and_w      = np.hstack((e_src[:, None], e_dst[:, None], weights[:, None])) 

    # 1. Nodos
    # -----------------

    # Inicializar matriz de autovectores
    if args.eigen:
        edge_tuples   = [(int(u), int(v), {'weight': w}) for u, v, w in e_and_w]
        nx_g          = nx.Graph(edge_tuples)
        lap           = nx.laplacian_matrix(nx_g).asfptype()
        w, v          = scipy.sparse.linalg.eigs(lap, k=args.eigen_dim)
        eigen_vectors = np.real(v)
    else:
        eigen_vectors = np.zeros((args.num_locs, args.eigen_dim))

    central_dict = {}
    nodes        = pd.DataFrame(data=range(args.num_locs), columns=['node_id'])
    reorder_ev   = eigen_vectors[nodes['node_id'].tolist()] # asegurar que filas de autovectores 
                                                            # están en mismo orden que los node id
    
    # Guardar autovecores a diccionario
    for i in range(args.eigen_dim):
        central_dict[f'eigen_vec{i:03d}'] = reorder_ev[:, i]

    # Pasar datos del diccionario a la tabla
    for col in central_dict:
        nodes[col] = central_dict[col]

    # Guardar tabla
    nodes.to_csv(f'{args.propath}/fea_dim_{args.eigen_dim}_{name}_nodes.csv', index=None)

    # 2. Aristas
    # -----------------
    edges = pd.DataFrame(data=e_and_w, columns=['src', 'dst', 'weight'])
    edges.to_csv(f'{args.propath}/fea_dim_{args.eigen_dim}_{name}_edges.csv', index=None)
    logger.info(f'save {name} nodes and edges done.')
