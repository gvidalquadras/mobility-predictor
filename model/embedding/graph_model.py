import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import pandas as pd
import dgl
from dgl.nn import GraphConv
from model.embedding.weighted_gatconv import WeightedGATConv
import numpy as np

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) para agregación de features de nodos en un grafo.
    Aplica capas de atención sobre el grafo para que cada nodo agregue información
    de sus vecinos de forma ponderada.

    En capas intermedias (no última), las salidas de las head_num cabezas se
    concatenan (flatten), multiplicando la dimensión por head_num. En la última
    capa se promedian (mean) en lugar de concatenar para obtener un único vector
    de dimensión out_size por nodo.

    Args:
        in_size   (int): dimensión de los features de entrada por nodo.
        hid_size  (int): dimensión de las capas ocultas intermedias.
        out_dim   (int): dimensión de los embeddings de salida por nodo.
        layer_num (int): número de capas GAT (0, 1 o ≥2).
        head_num  (int): número de cabezas de atención en capas intermedias.
                         La última capa siempre usa 1 cabeza.
        edge_type (str): tipo de arista del grafo (no usado internamente,
                         se mantiene por compatibilidad con la interfaz de STAR).

    Input:
        g (DGLGraph): grafo con aristas y pesos en g.edata['weight'].
        inputs (Tensor):   features iniciales de los nodos [num_nodes, in_size].

    Output:
        h (Tensor): embeddings de salida [num_nodes, out_size].
    """
    def __init__(self, in_size, hid_size, out_size, layer_num, head_num, edge_type):
        super().__init__()

        if layer_num == 0:
            self.gat_linear = nn.Linear(in_size, out_size)
        elif layer_num == 1:
            self.gat_layers = nn.ModuleList(
                [dglnn.GATConv(in_size, out_size, head_num, activation=F.elu)]
            )
            self.gat_linear = nn.Linear(out_size*head_num, out_size)
        else:
            self.gat_layers = nn.ModuleList()
            self.gat_layers.append(dglnn.GATConv(in_size, hid_size, head_num, activation=F.elu))
            for _ in range(layer_num - 2):
                self.gat_layers.append(dglnn.GATConv(hid_size*head_num, hid_size, head_num, activation=F.elu))
            self.gat_layers.append(dglnn.GATConv(hid_size*head_num, out_size, 1, activation=F.elu))
            self.gat_linear = nn.Identity()


    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == len(self.gat_layers) - 1: 
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h
    
class SoftmaxAttention(nn.Module):
    """
    Módulo de fusión por atención (combinar múltiples embeddings en uno).

    La fusión se hace en tres pasos:
        1. Transformación conjunta: los tres embeddings se concatenan y se pasan
            por una capa lineal con tanh para capturar interacciones entre ellos.
        2. Cálculo de pesos de atención: una capa lineal proyecta cada embedding
            transformado a un escalar (score). Los tres scores se normalizan con
            softmax para obtener pesos que suman 1 por nodo.
        3. Suma ponderada: cada embedding original se multiplica por su peso y
            se suman. El resultado se normaliza con LayerNorm.

    Los pesos de atención son distintos para cada nodo.

    Args:
        feat_dim (int): dimensión de cada embedding de entrada.
                        En STAREmbedding feat_dim=32.
        num      (int): número de embeddings a fusionar.
                        En STAREmbedding num=3 (SDG, TTG, STG).

    Parámetros aprendibles:
        trans (Linear): proyección [feat_dim*num → feat_dim*num] sin bias.
                        Captura interacciones entre los tres embeddings.
        query (Linear): proyección [feat_dim → 1] sin bias.
                        Calcula la importancia de cada grafo.
        layer_norm: normalización de la salida para estabilizar el entrenamiento.

    Input:
        embeds (list): lista de num tensores, cada uno de shape [N, feat_dim].
                        En STAREmbedding lista de tres tensores: [emb_sdg, emb_ttg, emb_stg].

    Output:
        ans (Tensor): embedding fusionado [N, feat_dim].
        weights (Tensor): pesos de atención por grafo [N, num].
                            weights[i] = [w_sdg, w_ttg, w_stg] para el nodo i,
                            donde w_sdg + w_ttg + w_stg = 1.
    """
    def __init__(self, feat_dim: int, num: int) -> None:
        super(SoftmaxAttention, self).__init__()
        self.trans = nn.Linear(feat_dim * num, feat_dim * num, bias=False)
        self.num = num
        self.query = nn.Linear(feat_dim, 1, bias=False)
        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, embeds: list) -> torch.Tensor:
        num = len(embeds)
        batch, dim = embeds[0].shape
        x = torch.stack(embeds, dim=1)
        trans_x = self.trans(x.view(batch, num * dim)).tanh()
        weights = self.query(trans_x.view(batch, num, dim))
        weights = torch.softmax(weights.view(batch, num), dim=1)
        ans = torch.bmm(weights.unsqueeze(1), x)
        ans = self.layer_norm(ans.sum(dim=1))
        return ans, weights