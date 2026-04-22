"""
graph_wavenet.py
================
Módulo de predicción temporal basado en Graph WaveNet (Wu et al., IJCAI 2019).
Adaptado para predicción de flujos de movilidad urbana en POIs y pantallas DOOH.

Diferencias respecto al paper original:
- in_dim=33:  1 canal de flujo + 32 dims de embedding STAR
                (original: 2 canales, velocidad + hora del día)
- out_dim=30: horizonte de 30 días
                (original: 12 pasos de 5 minutos = 1 hora)
- supports:   3 matrices (SDG, TTG, STG) de STAR
                (original: Pf y Pb, forward y backward de un grafo de tráfico)
- num_nodes:  1500 POIs NYC / adaptable a TapTap
                (original: 207 sensores METR-LA)

Arquitectura:
  Input [B, N, in_dim, T]
    → start_conv: proyección a residual_channels
    → blocks × layers capas ST:
        - Gated TCN (dilated causal conv): captura patrones temporales
        - GCN (difusión sobre grafos + matriz adaptativa): captura dependencias espaciales
        - Skip connection al output + residual connection local
    → suma de skip connections de todas las capas
    → end_conv_1, end_conv_2: proyección al horizonte de predicción
  Output [B, N, out_dim]

Referencia:
    Wu et al. "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
    IJCAI 2019. https://arxiv.org/abs/1906.00121
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ------------------------
# Bloques básicos
# ------------------------

class nconv(nn.Module):
    """
    Multiplicación nodo-grafo.
    Propaga las features de cada nodo a sus vecinos ponderado por A.

    Input:  x [B, C, N, T],  A [N, N]
    Output: [B, C, N, T]  (N transformado según A)
    """
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # para cada nodo destino w, agrega features
        # de todos los nodos origen v ponderados por A[v,w]
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    """
    Proyección lineal 1×1 sobre canales.
    Implementada como Conv2d con kernel (1,1) para eficiencia.

    Input:  [B, c_in,  N, T]
    Output: [B, c_out, N, T]
    """
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(
            c_in, c_out,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    """
    Graph Convolution Layer de Graph WaveNet (Ecuación 6 del paper).

    Para cada matriz de adyacencia en support, calcula difusión hasta
    'order' saltos y concatena todos los resultados. Luego proyecta
    a la dimensión de salida.

    Args:
        c_in        (int): canales de entrada
        c_out       (int): canales de salida
        dropout     (float): dropout rate
        support_len (int): número de matrices de adyacencia
        order       (int): número de saltos de difusión K
    """
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv   = nconv()
        # Dimensión de entrada tras concatenar todos los órdenes de difusión
        c_in_total   = (order * support_len + 1) * c_in
        self.mlp     = linear(c_in_total, c_out)
        self.dropout = dropout
        self.order   = order

    def forward(self, x, support):
        """
        Args:
            x       (Tensor): [B, C, N, T]
            support (list):   lista de matrices de adyacencia [N, N]
        Returns:
            h (Tensor): [B, c_out, N, T]
        """
        out = [x]   # k=0: features propias de cada nodo

        for a in support:
            # k=1: difusión un salto
            x1 = self.nconv(x, a)
            out.append(x1)
            # k=2,...,order: difusión múltiples saltos
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        # Concatenar todos los órdenes de difusión en dimensión de canales
        h = torch.cat(out, dim=1)           # [B, c_in*(order*support_len+1), N, T]
        h = self.mlp(h)                     # [B, c_out, N, T]
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# -----------------------
# Modelo completo
#------------------------

class GraphWaveNet(nn.Module):
    """
    Graph WaveNet para predicción de flujos de movilidad urbana.

    Combina:
        - Gated TCN con convoluciones dilatadas para dependencias temporales
        - GCN con matriz adaptativa para dependencias espaciales
        - Skip connections al output para capturar patrones a múltiples escalas

    Args:
        device             : dispositivo de cómputo (cpu / cuda)
        num_nodes    (int) : número de nodos (POIs + DOOH)
        dropout    (float) : dropout rate. Default: 0.3
        supports    (list) : lista de matrices de adyacencia predefinidas [N,N]
                            En tu caso: [P_sdg, P_ttg, P_stg]
        gcn_bool    (bool) : usar GCN. Default: True
        addaptadj   (bool) : añadir matriz adaptativa aprendible. Default: True
        aptinit            : inicialización de la matriz adaptativa. Default: None
        in_dim       (int) : canales de entrada.
                            33 = 1 flujo + 32 embedding STAR. Default: 33
        out_dim      (int) : horizonte de predicción en días. Default: 30
        residual_channels (int): canales en capas residuales. Default: 32
        dilation_channels (int): canales en capas dilatadas. Default: 32
        skip_channels     (int): canales en skip connections. Default: 256
        end_channels      (int): canales en capas finales. Default: 512
        kernel_size  (int) : tamaño del kernel temporal. Default: 2
        blocks       (int) : número de bloques ST. Default: 4
        layers       (int) : capas por bloque. Default: 2
    """
    def __init__(
        self,
        device,
        num_nodes,
        dropout           = 0.3,
        supports          = None,
        gcn_bool          = True,
        addaptadj         = True,
        aptinit           = None,
        in_dim            = 33,     # 1 flujo + 32 embedding STAR
        out_dim           = 30,     # horizonte 30 días
        residual_channels = 32,
        dilation_channels = 32,
        skip_channels     = 256,
        end_channels      = 512,
        kernel_size       = 2,
        blocks            = 4,
        layers            = 2,
    ):
        super(GraphWaveNet, self).__init__()

        self.dropout    = dropout
        self.blocks     = blocks
        self.layers     = layers
        self.gcn_bool   = gcn_bool
        self.addaptadj  = addaptadj

        # Listas de módulos por capa
        self.filter_convs   = nn.ModuleList()   # rama tanh del Gated TCN
        self.gate_convs     = nn.ModuleList()   # rama sigmoid del Gated TCN
        self.residual_convs = nn.ModuleList()   # 1×1 para residual connection
        self.skip_convs     = nn.ModuleList()   # 1×1 para skip connection
        self.bn             = nn.ModuleList()   # batch normalization
        self.gconv          = nn.ModuleList()   # graph convolution

        # Proyección inicial: in_dim -> residual_channels
        self.start_conv = nn.Conv2d(
            in_channels  = in_dim,
            out_channels = residual_channels,
            kernel_size  = (1, 1)
        )

        self.supports = supports

        # Calcular support_len: grafos predefinidos + adaptativo
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)  # SDG + TTG + STG = 3

        # Matriz adaptativa: E1 [N,10] y E2 [10,N] (E2 ya transpuesta)
        # Ã_adp = SoftMax(ReLU(E1 · E2))
        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                # Inicialización aleatoria (ecuación 5 del paper)
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10).to(device),
                    requires_grad=True
                ).to(device)
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes).to(device),
                    requires_grad=True
                ).to(device)
            else:
                # Inicialización con SVD de una matriz conocida
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
            self.supports_len += 1  # +1 por la matriz adaptativa

        # Construir capas ST: blocks × layers
        # Las dilations se doblan en cada capa: 1, 2, 1, 2, 1, 2, 1, 2
        # (con blocks=4 y layers=2)
        receptive_field   = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation     = 1
            for i in range(layers):

                # --- Gated TCN ---
                # Rama tanh: extrae información
                self.filter_convs.append(nn.Conv2d(
                    in_channels  = residual_channels,
                    out_channels = dilation_channels,
                    kernel_size  = (1, kernel_size),
                    dilation     = new_dilation
                ))
                # Rama sigmoid: controla qué información pasa (puerta)
                self.gate_convs.append(nn.Conv2d(
                    in_channels  = residual_channels,
                    out_channels = dilation_channels,
                    kernel_size  = (1, kernel_size),
                    dilation     = new_dilation
                ))

                # --- Connections ---
                # 1×1 para residual connection (mantiene dimensión)
                self.residual_convs.append(nn.Conv2d(
                    in_channels  = dilation_channels,
                    out_channels = residual_channels,
                    kernel_size  = (1, 1)
                ))
                # 1×1 para skip connection al output final
                self.skip_convs.append(nn.Conv2d(
                    in_channels  = dilation_channels,
                    out_channels = skip_channels,
                    kernel_size  = (1, 1)
                ))

                self.bn.append(nn.BatchNorm2d(residual_channels))

                # Actualizar dilation para la siguiente capa
                new_dilation     *= 2
                receptive_field  += additional_scope
                additional_scope *= 2

                # --- Graph Convolution ---
                if self.gcn_bool:
                    self.gconv.append(gcn(
                        c_in        = dilation_channels,
                        c_out       = residual_channels,
                        dropout     = dropout,
                        support_len = self.supports_len
                    ))

        # Capas de output: skip_channels -> end_channels -> out_dim
        self.end_conv_1 = nn.Conv2d(
            in_channels  = skip_channels,
            out_channels = end_channels,
            kernel_size  = (1, 1),
            bias         = True
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels  = end_channels,
            out_channels = out_dim,
            kernel_size  = (1, 1),
            bias         = True
        )

        self.receptive_field = receptive_field

    def forward(self, input):
        """
        Args:
            input (Tensor): [B, N, in_dim, T]
                B     = batch size
                N     = num_nodes (1500 POIs)
                in_dim = 33 (1 flujo + 32 embedding STAR)
                T     = pasos temporales históricos (28 días)

        Returns:
            output (Tensor): [B, N, out_dim]
                out_dim = 30 días predichos
        """
        # Reordenar a [B, in_dim, N, T] para Conv2d
        # (Conv2d opera sobre dim=1 como canales)
        x = input.permute(0, 2, 1, 3)   # [B, in_dim, N, T]

        # Padding si la secuencia es más corta que el campo receptivo
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0, 0, 0))

        # Proyección inicial
        x = self.start_conv(x)          # [B, residual_channels, N, T]
        skip = 0

        # Calcular matriz adaptativa una vez por forward pass
        # Ã_adp = SoftMax(ReLU(E1 · E2))  (Ecuación 5 del paper)
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp          = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # --- Capas ST (bloques × layers) ---
        for i in range(self.blocks * self.layers):
            residual = x

            # Gated TCN: h = tanh(filter) . sigmoid(gate)
            filter_ = self.filter_convs[i](residual)
            filter_ = torch.tanh(filter_)
            gate    = self.gate_convs[i](residual)
            gate    = torch.sigmoid(gate)
            x       = filter_ * gate                    # [B, dilation_channels, N, T']

            # Skip connection: acumula outputs de todas las capas
            s    = self.skip_convs[i](x)               # [B, skip_channels, N, T']
            try:
                # Recortar skip acumulado para que coincida en T
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip

            # Graph Convolution o residual simple
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x) # [B, residual_channels, N, T']

            # Residual connection: suma con input recortado en T
            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        # --- Output desde skip connections acumuladas ---
        x = F.relu(skip) # [B, skip_channels, N, 1]
        x = F.relu(self.end_conv_1(x)) # [B, end_channels, N, 1]
        x = self.end_conv_2(x) # [B, out_dim, N, 1]

        # Reordenar a [B, N, out_dim]
        x = x[:, :, :, -1]        # [B, out_dim, N]  toma el último paso temporal
        x = x.permute(0, 2, 1)    # [B, N, out_dim]
        
        return x


# ------------------------
# Script de prueba
# ------------------------

if __name__ == "__main__":
    import sys
    import os
    from scipy.sparse import load_npz

    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    GRAPHS_DIR  = os.path.join(PROJECT_DIR, "data", "graphs", "NYC")
    DEVICE      = "cpu"

    # Cargar grafos de STAR como matrices densas normalizadas
    def load_support(path):
        """Carga un grafo .npz y lo convierte a tensor normalizado por filas."""
        sp  = load_npz(path)
        arr = sp.toarray().astype(np.float32)
        # Normalizar por filas (matriz de transición)
        row_sum = arr.sum(axis=1, keepdims=True) + 1e-10
        arr     = arr / row_sum
        return torch.tensor(arr).to(DEVICE)

    print("Cargando grafos de STAR...")
    supports = [
        load_support(os.path.join(GRAPHS_DIR, "sdg.npz")),
        load_support(os.path.join(GRAPHS_DIR, "ttg.npz")),
        load_support(os.path.join(GRAPHS_DIR, "stg.npz")),
    ]
    print(f"  Grafos cargados: {len(supports)} × {supports[0].shape}")

    # Inicializar modelo
    print("\nInicializando GraphWaveNet...")
    model = GraphWaveNet(
        device    = DEVICE,
        num_nodes = 1500,
        supports  = supports,
        in_dim    = 33,
        out_dim   = 30,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros totales: {total_params:,}")
    print(f"  Campo receptivo:    {model.receptive_field} pasos")

    # Forward pass de prueba
    print("\nForward pass de prueba...")
    B, N, D, T = 2, 1500, 33, 28   # batch=2, 1500 POIs, 33 features, 28 días
    x_test = torch.randn(B, N, D, T).to(DEVICE)

    with torch.no_grad():
        out = model(x_test)

    print(f"  Input shape:  {x_test.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Esperado:     torch.Size([{B}, {N}, 30])")
    print("\n✓ GraphWaveNet funciona correctamente.")