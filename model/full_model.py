"""
full_model.py
=============
Modelo completo de predicción de flujos de movilidad urbana.

Combina STAREmbedding (módulo de embeddings espaciotemporales) con
GraphWaveNet (módulo de predicción temporal) en una arquitectura unificada.

"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import load_npz

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.embedding.star_embedding import STAREmbedding
from model.prediction.graph_wavenet import GraphWaveNet


# -----------------------------------
# Utilidades para cargar grafos
# -----------------------------------

def load_supports(graphs_dir: str, device: str = "cpu") -> list:
    """
    Carga los tres grafos de STAR (SDG, TTG, STG) como matrices de transición.

    Cada grafo .npz se convierte a tensor denso normalizado por filas,
    de forma que cada fila suma 1 (matriz de transición válida para difusión).

    Args:
        graphs_dir (str): ruta a la carpeta con los ficheros .npz
                        Ej: "data/graphs/NYC"
        device     (str): dispositivo de cómputo

    Returns:
        supports (list): [P_sdg, P_ttg, P_stg] tensores [N, N]
    """
    supports = []
    for name in ["sdg", "ttg", "stg"]:
        path = os.path.join(graphs_dir, f"{name}.npz")
        sp   = load_npz(path)
        arr  = sp.toarray().astype(np.float32)
        # Normalizar por filas -> matriz de transición
        row_sum = arr.sum(axis=1, keepdims=True) + 1e-10
        arr     = arr / row_sum
        supports.append(torch.tensor(arr).to(device))
    return supports


# ------------------
# Modelo completo
# ------------------

class FullModel(nn.Module):
    """
    Modelo completo: STAREmbedding (congelado) + GraphWaveNet.

    Args:
        propath           (str)  : ruta a los CSVs de grafos de STAR
                                   Ej: "data/graphs/NYC"
        supports          (list) : matrices de adyacencia [P_sdg, P_ttg, P_stg]
                                   generadas con load_supports()
        device            (str)  : dispositivo de cómputo. Default: "cpu"
        fea_dim           (int)  : dimensión features STAR. Default: 32
        hid_dim           (int)  : dimensión oculta STAR. Default: 32
        out_dim_emb       (int)  : dimensión embedding STAR. Default: 32
        layer_num         (int)  : capas GAT en STAR. Default: 2
        head_num          (int)  : attention heads en STAR. Default: 2
        out_dim_pred      (int)  : horizonte de predicción en días. Default: 30
        history_len       (int)  : días de historial. Default: 28
        residual_channels (int)  : canales residuales GWN. Default: 32
        dilation_channels (int)  : canales dilatados GWN. Default: 32
        skip_channels     (int)  : canales skip GWN. Default: 256
        end_channels      (int)  : canales finales GWN. Default: 512
        dropout           (float): dropout rate. Default: 0.3
        blocks            (int)  : bloques ST en GWN. Default: 4
        layers            (int)  : capas por bloque en GWN. Default: 2
    """
    def __init__(
        self,
        propath,
        supports,
        device            = "cpu",
        fea_dim           = 32,
        hid_dim           = 32,
        out_dim_emb       = 32,
        layer_num         = 2,
        head_num          = 2,
        out_dim_pred      = 7,
        history_len       = 14,
        residual_channels = 32,
        dilation_channels = 32,
        skip_channels     = 256,
        end_channels      = 512,
        dropout           = 0.3,
        blocks            = 4,
        layers            = 2,
    ):
        super(FullModel, self).__init__()

        self.device      = device
        self.history_len = history_len
        self.out_dim_emb = out_dim_emb

        # --- Módulo 1: STAREmbedding---
        self.star_embedding = STAREmbedding(
            propath   = propath,
            fea_dim   = fea_dim,
            hid_dim   = hid_dim,
            out_dim   = out_dim_emb,
            layer_num = layer_num,
            head_num  = head_num,
            device    = device,
        )
        print("  STAREmbedding inicializado (entrenamiento end-to-end).")

        # --- Módulo 2: GraphWaveNet ---
        # in_dim = 1 (flujo) + out_dim_emb (embedding STAR) = 33
        self.graph_wavenet = GraphWaveNet(
            device            = device,
            num_nodes         = supports[0].shape[0],
            dropout           = dropout,
            supports          = supports,
            gcn_bool          = True,
            addaptadj         = True,
            in_dim            = 1 + out_dim_emb,   # 33
            out_dim           = out_dim_pred,        # 30
            residual_channels = residual_channels,
            dilation_channels = dilation_channels,
            skip_channels     = skip_channels,
            end_channels      = end_channels,
            blocks            = blocks,
            layers            = layers,
        )
        print("  GraphWaveNet inicializado.")

    def forward(self, flow_history):
        """
        Args:
            flow_history (Tensor): [B, N, 1, T]
                B = batch size
                N = num_nodes (1500 POIs)
                1 = canal de flujo
                T = días de historial (28)

        Returns:
            predictions (Tensor): [B, N, out_dim]
                predicción de flujo para los próximos out_dim días
        """
        B, N, _, T = flow_history.shape

        # 1. Calcular embeddings de STAR 
        embeddings, _ = self.star_embedding() 

        # 2. Expandir embeddings a dimensión temporal
        # [N, 32] -> [1, N, 32, 1] -> [B, N, 32, T]
        emb = embeddings.unsqueeze(0).unsqueeze(-1) # [1, N, 32, 1]
        emb = emb.expand(B, -1, -1, T) # [B, N, 32, T]

        # 3. Concatenar flujos y embeddings
        # flow_history: [B, N, 1,  T]
        # emb:          [B, N, 32, T]
        # x:            [B, N, 33, T]
        x = torch.cat([flow_history, emb], dim=2)

        # 4. Predicción con GraphWaveNet 
        predictions = self.graph_wavenet(x) # [B, N, out_dim]

        return predictions

    def get_trainable_params(self):
        """Devuelve solo los parámetros entrenables (GraphWaveNet, STAR)."""
        return list(self.parameters())

    def count_params(self):
        """Muestra el número de parámetros de cada módulo."""
        star_params = sum(p.numel() for p in self.star_embedding.parameters())
        gwn_params  = sum(p.numel() for p in self.graph_wavenet.parameters())
        total       = star_params + gwn_params
        print(f"  STAREmbedding: {star_params:,} parámetros")
        print(f"  GraphWaveNet:  {gwn_params:,} parámetros")
        print(f"  Total:         {total:,} parámetros entrenables")


# ------------------
# Script de prueba
# ------------------

if __name__ == "__main__":

    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GRAPHS_DIR  = os.path.join(PROJECT_DIR, "data", "graphs", "NYC")
    DEVICE      = "cpu"

    print("Cargando grafos de STAR...")
    supports = load_supports(GRAPHS_DIR, device=DEVICE)
    print(f"  {len(supports)} grafos cargados: {supports[0].shape}")

    print("\nInicializando FullModel...")
    model = FullModel(
        propath  = GRAPHS_DIR,
        supports = supports,
        device   = DEVICE,
    ).to(DEVICE)

    print("\nParámetros del modelo:")
    model.count_params()

    print("\nForward pass de prueba...")
    B, N, T = 2, 1500, 28
    flow_history = torch.randn(B, N, 1, T).to(DEVICE)

    with torch.no_grad():
        predictions = model(flow_history)

    print(f"  Input shape:  {flow_history.shape}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Esperado:     torch.Size([{B}, {N}, 30])")
    print("\n✓ FullModel funciona correctamente.")