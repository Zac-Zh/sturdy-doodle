import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_scatter import scatter_mean
import e3nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from torch_scatter import scatter_mean

class SE3EquivariantGNN(nn.Module):
    """SE(3)-equivariant graph neural network for 3D structure generation."""

    def __init__(self, num_atom_types, hidden_dim=64, num_layers=3, num_edge_types=4):
        """
        Initialize SE(3)-equivariant GNN.

        Args:
            num_atom_types: Number of atom types
            hidden_dim: Hidden dimension
            num_layers: Number of message passing layers
            num_edge_types: Number of edge types
        """
        super(SE3EquivariantGNN, self).__init__()

        # Create irreps for the model
        self.scalar_irreps = o3.Irreps("{}x0e".format(hidden_dim))
        self.vector_irreps = o3.Irreps("{}x1o".format(hidden_dim // 3))
        self.hidden_irreps = self.scalar_irreps + self.vector_irreps
        self.output_irreps = o3.Irreps("1x0e + 1x1o")  # Position and type prediction

        # Node embedding
        self.node_embedding = nn.Embedding(num_atom_types, hidden_dim)

        # Edge embedding
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_dim)

        # SE(3) equivariant layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                EquivariantConvLayer(
                    irreps_in=self.hidden_irreps,
                    irreps_out=self.hidden_irreps,
                    irreps_edge=self.scalar_irreps,
                    hidden_features=hidden_dim,
                )
            )

        # Output layers
        # For atom type prediction (scalar)
        self.type_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atom_types)
        )

        # For 3D position prediction (vector)
        self.pos_pred = nn.Sequential(
            nn.Linear(hidden_dim // 3 * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x, edge_index, edge_type, pos, batch=None):
        """
        Forward pass of equivariant GNN.

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges]
            pos: Node positions [num_nodes, 3]
            batch: Batch indices [num_nodes]

        Returns:
            tuple: (New positions, Node type logits)
        """
        # Node feature embedding
        node_attr = self.node_embedding(x)  # [num_nodes, hidden_dim]

        # Edge feature embedding
        edge_attr = self.edge_embedding(edge_type)  # [num_edges, hidden_dim]

        # Create initial features with irreps structure
        # Scalar part (invariant)
        scalar_features = node_attr
        # Vector part (equivariant) - initialize with zeros
        vector_features = torch.zeros(x.size(0), hidden_dim // 3, 3, device=x.device)

        # Combine features
        features = {
            "0e": scalar_features,
            "1o": vector_features
        }

        # Apply equivariant layers
        for layer in self.layers:
            features = layer(features, edge_index, edge_attr, pos)

        # Extract outputs
        scalar_out = features["0e"]
        vector_out = features["1o"].reshape(x.size(0), -1)  # Flatten vector features

        # Predict atom types
        type_logits = self.type_pred(scalar_out)

        # Predict position offsets
        pos_offsets = self.pos_pred(vector_out)

        # Get new positions
        new_pos = pos + pos_offsets

        return new_pos, type_logits


class EquivariantConvLayer(nn.Module):
    """SE(3)-equivariant convolutional layer."""

    def __init__(self, irreps_in, irreps_out, irreps_edge, hidden_features):
        """
        Initialize equivariant convolutional layer.

        Args:
            irreps_in: Input irreducible representations
            irreps_out: Output irreducible representations
            irreps_edge: Edge feature irreducible representations
            hidden_features: Hidden dimension
        """
        super(EquivariantConvLayer, self).__init__()

        # Create tensor product for message passing
        self.tp = o3.FullTensorProduct(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            irreps_edge=irreps_edge,
            shared_weights=False,
            internal_weights=True
        )

        # Scalar network for edge features
        self.edge_net = FullyConnectedNet(
            [hidden_features] * 3,
            [hidden_features],
            activation=torch.nn.functional.silu
        )

        # Gate for combining scalar and vector features
        self.gate = Gate(
            irreps_in,
            [torch.nn.functional.silu for _ in range(len(irreps_in))],
            irreps_out
        )

    def forward(self, features, edge_index, edge_attr, pos):
        """
        Forward pass of equivariant convolutional layer.

        Args:
            features: Node features with irreps structure
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, hidden_dim]
            pos: Node positions [num_nodes, 3]

        Returns:
            dict: Updated features with irreps structure
        """
        # Process edge features
        edge_scalar = self.edge_net(edge_attr)

        # Calculate relative positions
        rel_pos = pos[edge_index[1]] - pos[edge_index[0]]

        # Create edge features with spherical harmonics
        edge_features = {
            "0e": edge_scalar,
            "1o": rel_pos.unsqueeze(1)  # Add dimension for irreps
        }

        # Extract source and target indices
        src, dst = edge_index

        # Create source node features
        src_features = {
            "0e": features["0e"][src],
            "1o": features["1o"][src]
        }

        # Apply tensor product for message passing
        messages = self.tp(src_features, edge_features)

        # Aggregate messages
        aggr_messages = {
            "0e": scatter_mean(messages["0e"], dst, dim=0, dim_size=features["0e"].size(0)),
            "1o": scatter_mean(messages["1o"], dst, dim=0, dim_size=features["1o"].size(0))
        }

        # Apply gate
        output = self.gate(aggr_messages)

        # Residual connection
        output["0e"] = output["0e"] + features["0e"]
        output["1o"] = output["1o"] + features["1o"]

        return output
