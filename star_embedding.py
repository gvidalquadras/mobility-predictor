"""
star_embedding.py
=================
Módulo de embeddings espaciotemporales basado en STAR-TKDE.

Implementa el Multi-channel Embedding Module que genera representaciones
ricas de cada localización combinando tres grafos:
  - SDG: Spatial Distance Graph (proximidad geográfica)
  - TTG: Temporal Transition Graph (patrones de transición)
  - STG: SpatioTemporal Graph (similitud de distribuciones de visita)

Cada grafo se procesa con un GAT independiente y los tres embeddings
resultantes se fusionan con SoftmaxAttention en un único vector por nodo.

El módulo es un nn.Module de PyTorch, entrenable end-to-end junto con
el módulo de predicción temporal.

Uso:
    embedding_module = STAREmbedding(
        propath="saved_data/NYC",
        fea_dim=32,
        hid_dim=32,
        out_dim=32,
        layer_num=2,
        head_num=2,
    )
    node_embeddings = embedding_module()  # shape: [num_nodes, out_dim]
"""

import torch
import torch.nn as nn
import dgl
import pandas as pd
import numpy as np
from graph_model import GAT, SoftmaxAttention


class STAREmbedding(nn.Module):
    """
    Multi-channel Embedding Module de STAR.

    Carga los tres grafos espaciotemporales preprocesados por stg_gen.py
    y aprende embeddings de nodos combinando GAT sobre cada grafo con
    fusión por SoftmaxAttention.

    Args:
        propath   (str): Ruta a la carpeta con los CSV de nodos y aristas.
                         Ej: "saved_data/NYC"
        fea_dim   (int): Dimensión de las features iniciales de los nodos
                         (eigen_dim de stg_gen.py). Default: 32.
        hid_dim   (int): Dimensión de las capas ocultas del GAT. Default: 32.
        out_dim   (int): Dimensión de los embeddings de salida. Default: 32.
        layer_num (int): Número de capas GAT. Default: 2.
        head_num  (int): Número de attention heads en el GAT. Default: 2.
        device    (str): Dispositivo de cómputo. Default: 'cpu'.
    """

    GRAPH_NAMES = ['sdg', 'ttg', 'stg']

    def __init__(
        self,
        propath:   str,
        fea_dim:   int = 32,
        hid_dim:   int = 32,
        out_dim:   int = 32,
        layer_num: int = 2,
        head_num:  int = 2,
        device:    str = 'cpu',
    ):
        super().__init__()
        self.propath   = propath
        self.fea_dim   = fea_dim
        self.out_dim   = out_dim
        self.device    = device

        # Cargar los tres grafos DGL desde los CSV
        self.graphs = self._load_graphs()
        num_nodes   = self.graphs[0].num_nodes()

        # Embedding inicial de nodos: aprendible desde cero
        # (los eigen vectors son todos cero, así que sustituimos por
        # una embedding matrix aprendible, igual que hace STAR internamente)
        self.node_embedding = nn.Embedding(num_nodes, fea_dim)

        # Un GAT independiente por cada grafo (SDG, TTG, STG)
        self.gat_layers = nn.ModuleList([
            GAT(
                in_size   = fea_dim,
                hid_size  = hid_dim,
                out_size  = out_dim,
                layer_num = layer_num,
                head_num  = head_num,
                edge_type = 'adjacent',
            )
            for _ in self.GRAPH_NAMES
        ])

        # Fusión de los tres embeddings en uno solo
        self.attention_fusion = SoftmaxAttention(
            feat_dim = out_dim,
            num      = len(self.GRAPH_NAMES),
        )

    def _load_graphs(self):
        """
        Carga los tres grafos desde los CSV generados por stg_gen.py.
        Devuelve una lista de grafos DGL: [g_sdg, g_ttg, g_stg].
        """
        graphs = []
        for name in self.GRAPH_NAMES:
            nodes_path = f"{self.propath}/fea_dim_{self.fea_dim}_{name}_nodes.csv"
            edges_path = f"{self.propath}/fea_dim_{self.fea_dim}_{name}_edges.csv"

            nodes_df = pd.read_csv(nodes_path)
            edges_df = pd.read_csv(edges_path)

            src     = torch.from_numpy(edges_df['src'].to_numpy().astype(np.int32))
            dst     = torch.from_numpy(edges_df['dst'].to_numpy().astype(np.int32))
            weights = torch.from_numpy(edges_df['weight'].to_numpy()).float()

            g = dgl.graph((src, dst), num_nodes=nodes_df.shape[0])
            g = dgl.add_self_loop(g)

            # Pesos de aristas: aristas originales + 1.0 para self-loops
            g.edata['weight'] = torch.cat([weights, torch.ones(nodes_df.shape[0])]).float()

            graphs.append(g.to(self.device))  # ← esto faltaba

        return graphs

    def forward(self):
        """
        Calcula los embeddings de todas las localizaciones.

        Returns:
            embeddings (Tensor): shape [num_nodes, out_dim]
                Embeddings finales de cada localización, que combinan
                información espacial, temporal y espaciotemporal.
            weights (Tensor): shape [num_nodes, 3]
                Pesos de atención asignados a cada grafo por nodo,
                útiles para interpretabilidad.
        """
        # Features iniciales: embedding aprendible por nodo
        num_nodes   = self.graphs[0].num_nodes()
        node_ids    = torch.arange(num_nodes, device=self.device)
        node_feats  = self.node_embedding(node_ids)  # [num_nodes, fea_dim]

        # Pasar las features por el GAT de cada grafo
        graph_embeddings = []
        for g, gat in zip(self.graphs, self.gat_layers):
            emb = gat(g, node_feats)          # [num_nodes, out_dim]
            graph_embeddings.append(emb)

        # Fusionar los tres embeddings con SoftmaxAttention
        fused, weights = self.attention_fusion(graph_embeddings)
        # fused:   [num_nodes, out_dim]
        # weights: [num_nodes, 3] → importancia de SDG, TTG, STG por nodo

        return fused, weights


# ──────────────────────────────────────────────────────────────────────────────
# Script de prueba: ejecutar directamente para verificar que funciona
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    PROPATH  = "saved_data/NYC"
    FEA_DIM  = 32
    HID_DIM  = 32
    OUT_DIM  = 32
    LAYER_NUM = 2
    HEAD_NUM  = 2
    DEVICE   = "cpu"   # cambiar a "cuda:0" si tienes GPU disponible

    print("Inicializando STAREmbedding...")
    model = STAREmbedding(
        propath   = PROPATH,
        fea_dim   = FEA_DIM,
        hid_dim   = HID_DIM,
        out_dim   = OUT_DIM,
        layer_num = LAYER_NUM,
        head_num  = HEAD_NUM,
        device    = DEVICE,
    )
    print(f"  Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")

    print("\nCalculando embeddings...")
    with torch.no_grad():
        embeddings, attn_weights = model()

    print(f"  Shape embeddings:    {embeddings.shape}")
    print(f"  Shape attn_weights:  {attn_weights.shape}")
    print(f"\nPesos de atención por grafo (media sobre todos los nodos):")
    mean_weights = attn_weights.mean(dim=0)
    for name, w in zip(STAREmbedding.GRAPH_NAMES, mean_weights):
        print(f"  {name.upper()}: {w.item():.4f}")

    print(f"\nEjemplo embedding nodo 0: {embeddings[0, :8].tolist()}")
    print("\n✓ STAREmbedding funciona correctamente.")