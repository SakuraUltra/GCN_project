import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn_lib.graph_generator import GridGraphGenerator

def debug_grid_sizes():
    # 1. Simulate a ResNet feature map for a standard ReID image (256x128)
    # ResNet usually downsamples by 16x or 32x. 
    # Let's assume input (B, 2048, 16, 8) which matches ResNet50 output for 256x128
    batch_size = 2
    in_channels = 2048
    H_feat, W_feat = 16, 8
    
    dummy_feat_map = torch.randn(batch_size, in_channels, H_feat, W_feat)
    
    print(f"🔍 Input Feature Map Shape: {dummy_feat_map.shape} (Simulated ResNet50 Output)")
    print("-" * 60)

    # 2. Test configurations from your Checklist
    test_configs = [
        (4, 4),   # Very coarse
        (8, 8),   # Standard
        (12, 12), # Fine (Checklist item)
        (16, 8)   # Direct mapping (H_feat, W_feat)
    ]

    for grid_h, grid_w in test_configs:
        print(f"\n🧪 Testing Grid Size: {grid_h}x{grid_w}")
        
        # Instantiate Generator
        generator = GridGraphGenerator(in_channels, grid_size=(grid_h, grid_w))
        
        # Forward pass
        nodes = generator(dummy_feat_map)
        
        # Check Adjacency
        adj = generator.get_adjacency_matrix(device='cpu')
        num_edges = adj.sum().item()
        has_isolated = (adj.sum(dim=1) == 0).any().item()
        
        # Report
        num_nodes = nodes.size(1) # [B, N, C]
        feat_dim = nodes.size(2)
        
        print(f"   ✅ Nodes Shape: {nodes.shape} => {num_nodes} nodes per image")
        print(f"   ✅ Node Dim:   {feat_dim} (Should be 2048)")
        print(f"   🔗 Initial Edges (8-neighbor): {int(num_edges)}")
        if has_isolated:
             print(f"   ⚠️ WARNING: Found isolated nodes! (Nodes with 0 edges)")
        else:
             print(f"   ✅ Graph Connectivity: OK (No isolated nodes)")

        # Verify Adaptive Pooling effect
        if (grid_h, grid_w) == (12, 12):
            print(f"   ℹ️ Note: Input was 16x8, Output is 12x12. Adaptive Pool handled the resizing.")

if __name__ == "__main__":
    debug_grid_sizes()
