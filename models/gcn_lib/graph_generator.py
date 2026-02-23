import torch
import torch.nn as nn
import torch.nn.functional as F

class GridGraphGenerator(nn.Module):
    """
    Generate graph nodes from CNN feature maps using Grid Pooling.
    """
    def __init__(self, in_channels, grid_size=(8, 8)):
        """
        Args:
            in_channels (int): Input feature dimension (e.g. 2048)
            grid_size (tuple): (H, W) of the target grid, e.g. (8, 8)
                               Input feature map will be adaptively pooled to this size.
        """
        super(GridGraphGenerator, self).__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.num_nodes = grid_size[0] * grid_size[1]
        
        # Adaptive pooling to ensure fixed number of nodes regardless of input size
        self.pool = nn.AdaptiveAvgPool2d(grid_size)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): CNN feature maps [B, C, H, W]
            
        Returns:
            nodes (torch.Tensor): Graph node features [B, num_nodes, C]
            adj (torch.Tensor): Adjacency matrix [B, num_nodes, num_nodes] (Optional, for now)
        """
        B, C, H, W = x.size()
        
        # 1. Grid Pooling -> [B, C, grid_h, grid_w]
        x_pooled = self.pool(x)
        
        # 2. Flatten to nodes -> [B, C, num_nodes]
        # Permute to [B, num_nodes, C] for standard GNN input
        nodes = x_pooled.view(B, C, -1).permute(0, 2, 1).contiguous()
        
        return nodes

    def get_adjacency_matrix(self, device):
        """
        Create a fixed 8-neighbor adjacency matrix for the grid.
        Returns:
            adj (torch.Tensor): [num_nodes, num_nodes]
        """
        H, W = self.grid_size
        num_nodes = self.num_nodes
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        
        for r in range(H):
            for c in range(W):
                curr_idx = r * W + c
                
                # Check 8 neighbors
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            neighbor_idx = nr * W + nc
                            adj[curr_idx, neighbor_idx] = 1.0
                            
        return adj
