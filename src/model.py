"""Models over pose-feature sequences.

Input  : (B, T, F)
Output : (B, C)
"""
from __future__ import annotations

import math

import torch
from torch import nn


# OpenPose BODY_25 skeletal edges. Used by EC3D for the 25-joint topology.
BODY25_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),                                  # nose - neck
    (1, 2), (2, 3), (3, 4),                  # neck - R_shoulder - R_elbow - R_wrist
    (1, 5), (5, 6), (6, 7),                  # neck - L_shoulder - L_elbow - L_wrist
    (1, 8),                                  # neck - mid_hip
    (8, 9), (9, 10), (10, 11),               # mid_hip - R_hip - R_knee - R_ankle
    (8, 12), (12, 13), (13, 14),             # mid_hip - L_hip - L_knee - L_ankle
    (0, 15), (15, 17),                       # nose - R_eye - R_ear
    (0, 16), (16, 18),                       # nose - L_eye - L_ear
    (14, 19), (19, 20), (14, 21),            # L_ankle - L_bigtoe - L_smalltoe; L_ankle - L_heel
    (11, 22), (22, 23), (11, 24),            # R_ankle - R_bigtoe - R_smalltoe; R_ankle - R_heel
)


def build_body25_adjacency(num_joints: int = 25) -> torch.Tensor:
    """Symmetric, normalised adjacency for BODY_25 with self-loops."""
    A = torch.zeros(num_joints, num_joints)
    for i, j in BODY25_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(num_joints)
    deg = A.sum(dim=1).clamp(min=1.0)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)


def _bfs_distance(adj: dict[int, set[int]], source: int, n: int) -> list[float]:
    """Graph distance from `source` to every joint (inf for disconnected)."""
    dist = [float("inf")] * n
    dist[source] = 0
    q = [source]
    while q:
        u = q.pop(0)
        for v in adj[u]:
            if dist[v] == float("inf"):
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def build_body25_partitioned_adjacency(num_joints: int = 25, center: int = 8) -> torch.Tensor:
    """Spatial-configuration partition strategy from ST-GCN (Yan 2018).

    For each joint i, neighbours are split by distance to the skeleton centroid
    (mid_hip = joint 8 for BODY_25):
      - partition 0 (root): self-loop only
      - partition 1 (centripetal): neighbour closer to centre than i
      - partition 2 (centrifugal): neighbour farther from centre than i
    Equidistant neighbours go in centripetal by convention.

    Returns (3, num_joints, num_joints) of D^-1/2 * A_k * D^-1/2 per partition.
    """
    adj_set: dict[int, set[int]] = {i: set() for i in range(num_joints)}
    for i, j in BODY25_EDGES:
        adj_set[i].add(j)
        adj_set[j].add(i)
    dist = _bfs_distance(adj_set, center, num_joints)

    A_root = torch.eye(num_joints)
    A_cent = torch.zeros(num_joints, num_joints)
    A_centrif = torch.zeros(num_joints, num_joints)
    for i in range(num_joints):
        for j in adj_set[i]:
            if dist[j] < dist[i]:
                A_cent[i, j] = 1.0
            elif dist[j] > dist[i]:
                A_centrif[i, j] = 1.0
            else:
                A_cent[i, j] = 1.0

    def normalise(A: torch.Tensor) -> torch.Tensor:
        deg = A.sum(dim=1).clamp(min=1.0)
        d_inv_sqrt = deg.pow(-0.5)
        return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)

    return torch.stack([normalise(A_root), normalise(A_cent), normalise(A_centrif)])


class BiLSTMHead(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        pooled = h.mean(dim=1)
        return self.head(self.dropout(pooled))


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerHead(nn.Module):
    """Small Transformer encoder over pose-feature sequences.

    Projects raw features to d_model, adds sinusoidal positional encoding,
    runs N encoder layers, mean-pools over time, then classifies.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = _SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.proj(x))
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(self.dropout(pooled))


class _GraphConv(nn.Module):
    """Single-partition graph conv along the joint dim.

    Input/output shape: (B, C, T, J).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 1x1 conv on channel dim, applied per (t, j) — same as a Linear over channels.
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C_out, T, J)
        # Aggregate over neighbours: y[..., j] = sum_k A[j, k] * x[..., k]
        return torch.einsum("bctk,jk->bctj", x, A)


class _STGCNBlock(nn.Module):
    """Graph conv + 1D temporal conv with residual, mirroring the GCB in the EC3D paper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 9,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        padding = temporal_kernel // 2
        self.gcn = _GraphConv(in_channels, out_channels)
        self.bn_g = nn.BatchNorm2d(out_channels)
        self.tcn = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(temporal_kernel, 1),
            padding=(padding, 0),
        )
        self.bn_t = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.bn_g(x)
        x = self.relu(x)
        x = self.tcn(x)
        x = self.bn_t(x)
        x = self.relu(x + res)
        return self.dropout(x)


class STGCNHead(nn.Module):
    """Stacked ST-GCN classifier over a (B, T, num_joints * channels_per_joint) tensor.

    Reshapes the flat input back to (B, C, T, J), runs N spatio-temporal blocks
    using the precomputed BODY_25 adjacency, global-pools over (T, J), classifies.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: int = 64,
        num_layers: int = 3,
        dropout: float = 0.4,
        num_joints: int = 25,
        temporal_kernel: int = 9,
    ) -> None:
        super().__init__()
        if n_features % num_joints != 0:
            raise ValueError(
                f"n_features={n_features} not divisible by num_joints={num_joints}; "
                "use feature_mode='positions' (75 = 25*3) for ST-GCN."
            )
        self.num_joints = num_joints
        self.register_buffer("A", build_body25_adjacency(num_joints))

        in_channels = n_features // num_joints
        channels = [in_channels] + [hidden] * num_layers
        self.blocks = nn.ModuleList(
            [
                _STGCNBlock(
                    channels[i], channels[i + 1],
                    temporal_kernel=temporal_kernel, dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) where F = num_joints * C_in
        B, T, _ = x.shape
        C = x.shape[-1] // self.num_joints
        # (B, T, J, C) -> (B, C, T, J)
        x = x.reshape(B, T, self.num_joints, C).permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x, self.A)
        # Global average pool over time and joints.
        x = x.mean(dim=(2, 3))
        return self.head(x)


class HybridSTGCNHead(nn.Module):
    """ST-GCN over positions + parallel temporal MLP over engineered features.

    Input layout: features are split into [first n_pose dims = position channels,
    remaining n_extra dims = engineered features (angles + geom)]. The ST-GCN
    branch runs on the pose channels; a small 1D conv stack runs on the extras
    in parallel; the two pooled embeddings are concatenated and classified.
    """

    def __init__(
        self,
        n_pose_features: int,
        n_extra_features: int,
        n_classes: int,
        stgcn_hidden: int = 64,
        stgcn_layers: int = 3,
        extra_hidden: int = 64,
        dropout: float = 0.4,
        num_joints: int = 25,
        temporal_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.n_pose_features = n_pose_features
        self.n_extra_features = n_extra_features
        self.stgcn = STGCNHead(
            n_pose_features, stgcn_hidden,   # use stgcn_hidden as the head output dim
            hidden=stgcn_hidden,
            num_layers=stgcn_layers,
            dropout=dropout,
            num_joints=num_joints,
            temporal_kernel=temporal_kernel,
        )
        # Replace stgcn's classification head with identity so we get the pooled vector.
        self.stgcn.head = nn.Identity()

        # Tiny temporal conv stack on the extra features.
        self.extra_stem = nn.Sequential(
            nn.Conv1d(n_extra_features, extra_hidden, kernel_size=temporal_kernel,
                      padding=temporal_kernel // 2),
            nn.BatchNorm1d(extra_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(extra_hidden, extra_hidden, kernel_size=temporal_kernel,
                      padding=temporal_kernel // 2),
            nn.BatchNorm1d(extra_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(stgcn_hidden + extra_hidden, stgcn_hidden + extra_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(stgcn_hidden + extra_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_pose + n_extra). Split.
        pose_x = x[..., : self.n_pose_features]
        extra_x = x[..., self.n_pose_features :]
        # ST-GCN expects (B, T, F). Output: (B, stgcn_hidden) after head=Identity.
        pose_emb = self.stgcn(pose_x)
        # extra stem expects (B, C, T). Output: (B, extra_hidden) after mean-pool.
        extra_h = self.extra_stem(extra_x.transpose(1, 2))
        extra_emb = extra_h.mean(dim=-1)
        merged = torch.cat([pose_emb, extra_emb], dim=-1)
        return self.classifier(merged)


class _MultiPartitionGraphConv(nn.Module):
    """ST-GCN graph conv with K learnable adjacency partitions.

    Input  : (B, C_in, T, J)
    Adj    : (K, J, J)
    Output : (B, C_out, T, J)
    """

    def __init__(self, in_channels: int, out_channels: int, n_partitions: int = 3) -> None:
        super().__init__()
        self.n_partitions = n_partitions
        # One channel projection per partition, packed into a single conv.
        self.proj = nn.Conv2d(in_channels, out_channels * n_partitions, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, _, T, J = x.shape
        h = self.proj(x)  # (B, K * C_out, T, J)
        h = h.view(B, self.n_partitions, -1, T, J)  # (B, K, C_out, T, J)
        # y[b, c, t, j] = sum_{k, m} A[k, j, m] * h[b, k, c, t, m]
        return torch.einsum("bkctm,kjm->bctj", h, A)


class _STGCNv2Block(nn.Module):
    """Faithful ST-GCN GCB block: multi-partition spatial + 1D temporal + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_partitions: int = 3,
        temporal_kernel: int = 9,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        padding = temporal_kernel // 2
        self.gcn = _MultiPartitionGraphConv(in_channels, out_channels, n_partitions)
        self.bn_g = nn.BatchNorm2d(out_channels)
        self.tcn = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(temporal_kernel, 1),
            padding=(padding, 0),
        )
        self.bn_t = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.bn_g(x)
        x = self.relu(x)
        x = self.tcn(x)
        x = self.bn_t(x)
        x = self.relu(x + res)
        return self.dropout(x)


class STGCNv2Head(nn.Module):
    """Multi-partition ST-GCN classification head — closer to the EC3D paper's GCB stack."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: int = 64,
        num_layers: int = 3,
        dropout: float = 0.4,
        num_joints: int = 25,
        temporal_kernel: int = 9,
        n_partitions: int = 3,
    ) -> None:
        super().__init__()
        if n_features % num_joints != 0:
            raise ValueError(
                f"n_features={n_features} not divisible by num_joints={num_joints}; "
                "use feature_mode='positions' (75 = 25*3) for ST-GCN."
            )
        self.num_joints = num_joints
        self.register_buffer("A", build_body25_partitioned_adjacency(num_joints))

        in_channels = n_features // num_joints
        channels = [in_channels] + [hidden] * num_layers
        self.blocks = nn.ModuleList(
            [
                _STGCNv2Block(
                    channels[i], channels[i + 1],
                    n_partitions=n_partitions,
                    temporal_kernel=temporal_kernel,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        C = x.shape[-1] // self.num_joints
        x = x.reshape(B, T, self.num_joints, C).permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x, self.A)
        x = x.mean(dim=(2, 3))
        return self.head(x)


class HybridSTGCNv2Head(nn.Module):
    """HybridSTGCN built on the multi-partition v2 backbone instead of v1."""

    def __init__(
        self,
        n_pose_features: int,
        n_extra_features: int,
        n_classes: int,
        stgcn_hidden: int = 64,
        stgcn_layers: int = 3,
        extra_hidden: int = 64,
        dropout: float = 0.4,
        num_joints: int = 25,
        temporal_kernel: int = 9,
        n_partitions: int = 3,
    ) -> None:
        super().__init__()
        self.n_pose_features = n_pose_features
        self.n_extra_features = n_extra_features
        self.stgcn = STGCNv2Head(
            n_pose_features, stgcn_hidden,
            hidden=stgcn_hidden,
            num_layers=stgcn_layers,
            dropout=dropout,
            num_joints=num_joints,
            temporal_kernel=temporal_kernel,
            n_partitions=n_partitions,
        )
        self.stgcn.head = nn.Identity()

        self.extra_stem = nn.Sequential(
            nn.Conv1d(n_extra_features, extra_hidden, kernel_size=temporal_kernel,
                      padding=temporal_kernel // 2),
            nn.BatchNorm1d(extra_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(extra_hidden, extra_hidden, kernel_size=temporal_kernel,
                      padding=temporal_kernel // 2),
            nn.BatchNorm1d(extra_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(stgcn_hidden + extra_hidden, stgcn_hidden + extra_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(stgcn_hidden + extra_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pose_x = x[..., : self.n_pose_features]
        extra_x = x[..., self.n_pose_features :]
        pose_emb = self.stgcn(pose_x)
        extra_h = self.extra_stem(extra_x.transpose(1, 2))
        extra_emb = extra_h.mean(dim=-1)
        merged = torch.cat([pose_emb, extra_emb], dim=-1)
        return self.classifier(merged)
