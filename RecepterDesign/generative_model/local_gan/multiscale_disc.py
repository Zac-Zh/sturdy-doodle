import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


class MultiscaleDiscriminator(nn.Module):
    """Multi-scale discriminator for validating generated molecular structures."""

    def __init__(self, input_dim, hidden_dim=64, num_scales=3, dropout=0.1):
        """
        Initialize multi-scale discriminator.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_scales: Number of spatial scales to consider
            dropout: Dropout probability
        """
        super(MultiscaleDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

        # Initial node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-scale GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_scales):
            # Radius increases with scale
            radius = 3.0 * (1.5 ** i)  # 3.0, 4.5, 6.75, ...
            self.gnn_layers.append(
                RadiusGraphConv(hidden_dim, hidden_dim, radius=radius)
            )

        # Pooling and final classification layers
        self.pool_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_scales)
        ])

        # Final discrimination layers
        self.geometry_scorer = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.function_scorer = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, pos, batch, edge_index=None, edge_attr=None):
        """
        Forward pass of multi-scale discriminator.

        Args:
            x: Node features [num_nodes, input_dim]
            pos: Node positions [num_nodes, 3]
            batch: Batch indices [num_nodes]
            edge_index: Optional edge indices [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]

        Returns:
            tuple: (Geometry score, Function score)
        """
        # Initial node encoding
        h = self.node_encoder(x)

        # Process at multiple scales
        pooled_features = []
        for i, gnn in enumerate(self.gnn_layers):
            # Apply GNN at this scale
            h_scale = gnn(h, pos, batch, edge_index)

            # Global pooling for this scale
            pooled = global_mean_pool(h_scale, batch)

            # Scale-specific processing
            pooled = self.pool_layers[i](pooled)

            # Save for final scoring
            pooled_features.append(pooled)

        # Concatenate multi-scale features
        multi_scale_features = torch.cat(pooled_features, dim=1)

        # Score geometry and function
        geom_score = self.geometry_scorer(multi_scale_features)
        func_score = self.function_scorer(multi_scale_features)

        return geom_score, func_score


class RadiusGraphConv(MessagePassing):
    """Graph convolution with radius-based edge connection."""

    def __init__(self, in_channels, out_channels, radius=5.0):
        """
        Initialize radius-based graph convolution.

        Args:
            in_channels: Input channel dimension
            out_channels: Output channel dimension
            radius: Radius cutoff for edge formation
        """
        super(RadiusGraphConv, self).__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius

        # Message calculation network
        self.message_nn = nn.Sequential(
            nn.Linear(in_channels * 2 + 1, out_channels),  # +1 for distance
            nn.ReLU()
        )

        # Update network
        self.update_nn = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, pos, batch, edge_index=None):
        """
        Forward pass of radius graph convolution.

        Args:
            x: Node features [num_nodes, in_channels]
            pos: Node positions [num_nodes, 3]
            batch: Batch indices [num_nodes]
            edge_index: Optional existing edge indices

        Returns:
            torch.Tensor: Updated node features
        """
        # If edge_index is not provided, compute edges based on radius
        if edge_index is None:
            edge_index = self._compute_radius_edges(pos, batch)

        # Compute edge attributes (distances)
        row, col = edge_index
        edge_attr = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)

        # Run message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """Compute messages based on source, destination features and edge attributes."""
        # Concatenate source, destination features and distance
        inputs = torch.cat([x_i, x_j, edge_attr], dim=1)
        # Apply message network
        return self.message_nn(inputs)

    def update(self, aggr_out, x):
        """Update node features based on aggregated messages."""
        # Concatenate node features with aggregated messages
        inputs = torch.cat([x, aggr_out], dim=1)
        # Apply update network
        return self.update_nn(inputs)

    def _compute_radius_edges(self, pos, batch):
        """Compute edge indices based on radius cutoff."""
        # For simplicity, we'll do a brute force approach here
        # In a real implementation, we would use spatial indexing (KDTree, etc.)
        device = pos.device
        num_nodes = pos.size(0)

        # Compute all pairwise distances
        # First, expand positions to create matrices for all pairs
        pos_i = pos.unsqueeze(1)  # [num_nodes, 1, 3]
        pos_j = pos.unsqueeze(0)  # [1, num_nodes, 3]

        # Compute squared distances
        dist_sq = torch.sum((pos_i - pos_j) ** 2, dim=2)  # [num_nodes, num_nodes]

        # Create mask for batch (nodes can only connect within same batch)
        batch_i = batch.unsqueeze(1)  # [num_nodes, 1]
        batch_j = batch.unsqueeze(0)  # [1, num_nodes]
        batch_mask = (batch_i == batch_j)  # [num_nodes, num_nodes]

        # Create mask for radius cutoff
        radius_mask = dist_sq <= (self.radius ** 2)  # [num_nodes, num_nodes]

        # Remove self-loops
        self_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device)

        # Combine masks
        combined_mask = batch_mask & radius_mask & self_mask

        # Get indices where mask is True
        edge_index = combined_mask.nonzero(as_tuple=True)

        # Convert to expected format [2, num_edges]
        return torch.stack(edge_index)
