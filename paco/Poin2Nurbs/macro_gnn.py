import torch
import torch.nn as nn
import torch.nn.functional as F


class MacroGraphBlock(nn.Module):
    """
    Single Graph Attention / Message Passing layer over the K zone nodes.
    Computes spatial k-NN edges from 3D anchor coordinates, applies Edge-MLP
    with relative 3D positional encoding, and performs attention-weighted aggregation.
    """
    def __init__(self, embed_dim=128, k_neighbors=6):
        super().__init__()
        self.k = k_neighbors
        self.embed_dim = embed_dim

        # Input: [h_i (C), h_j - h_i (C), rel_pos (3), rel_dist (1)] -> 2*C + 4
        in_dim = embed_dim * 2 + 4
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.attn_proj = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h, anchors):
        """
        Args:
            h: Node features of shape (B, K, C)
            anchors: 3D anchor coordinates of shape (B, K, 3)
        Returns:
            Updated node features of shape (B, K, C)
        """
        B, K, C = h.shape

        # 1. Pairwise Euclidean distance between K anchors
        dists = torch.cdist(anchors, anchors)  # (B, K, K)

        # 2. k Nearest Neighbors
        knn_dists, knn_idx = torch.topk(dists, k=self.k, dim=-1, largest=False)  # (B, K, k)

        # 3. Gather neighbor features
        idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, C)  # (B, K, k, C)
        h_expanded = h.unsqueeze(1).expand(-1, K, -1, -1)          # (B, K, K, C)
        h_neighbors = torch.gather(h_expanded, 2, idx_expanded)    # (B, K, k, C)

        # 4. Gather neighbor anchor positions
        anc_idx = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)      # (B, K, k, 3)
        anc_expanded = anchors.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, K, 3)
        anc_neighbors = torch.gather(anc_expanded, 2, anc_idx)      # (B, K, k, 3)

        # 5. Relative 3D spatial features
        h_self = h.unsqueeze(2).expand(-1, -1, self.k, -1)         # (B, K, k, C)
        anc_self = anchors.unsqueeze(2).expand(-1, -1, self.k, -1) # (B, K, k, 3)
        rel_pos = anc_neighbors - anc_self                         # (B, K, k, 3)
        rel_dist = knn_dists.unsqueeze(-1)                         # (B, K, k, 1)

        # 6. Edge feature & message computation
        h_diff = h_neighbors - h_self                              # (B, K, k, C)
        edge_input = torch.cat([h_self, h_diff, rel_pos, rel_dist], dim=-1)
        messages = self.edge_mlp(edge_input)                       # (B, K, k, C)

        # 7. Attention-weighted aggregation across the k neighbors
        attn_logits = self.attn_proj(messages)                     # (B, K, k, 1)
        attn_weights = F.softmax(attn_logits, dim=2)               # (B, K, k, 1)
        aggregated = (messages * attn_weights).sum(dim=2)          # (B, K, C)

        # 8. Residual connection + LayerNorm
        return self.norm(h + aggregated)


class MacroGNN(nn.Module):
    """
    Tier 2: Macro Planner GNN over K=32 Zone Nodes.
    Coordinates boundary features, seam alignment, and topological curvature
    across adjacent 3D patches via 2-hop spatial message passing.
    """
    def __init__(self, embed_dim=128, k_neighbors=6, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            MacroGraphBlock(embed_dim=embed_dim, k_neighbors=k_neighbors)
            for _ in range(num_layers)
        ])

    def forward(self, zonal_embeddings, zone_anchors):
        """
        Args:
            zonal_embeddings: (B, K, embed_dim)
            zone_anchors: (B, K, 3)
        Returns:
            coordinated_plan: (B, K, embed_dim)
        """
        h = zonal_embeddings
        for layer in self.layers:
            h = layer(h, zone_anchors)
        return h
