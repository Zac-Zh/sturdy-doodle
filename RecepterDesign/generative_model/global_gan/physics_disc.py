import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphConv, GCNConv
from torch_geometric.utils import add_self_loops, degree
import numpy as np


class PhysicsDiscriminator(nn.Module):
    """Physics-based discriminator to evaluate protein structures."""

    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        """
        Initialize physics discriminator.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            dropout: Dropout probability
        """
        super(PhysicsDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv2 = GCNConv(hidden_dim * 2, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        # Energy term prediction layers
        # Bond energy
        self.bond_energy_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Angle energy
        self.angle_energy_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Dihedral energy
        self.dihedral_energy_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Non-bonded energy (electrostatics and van der Waals)
        self.nb_energy_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Transmembrane domain validator
        self.tm_validator = TransmembraneValidator(hidden_dim * 2, hidden_dim)

    def forward(self, x, edge_index, pos, batch, atom_types=None, is_transmembrane=False):
        """
        Forward pass of physics discriminator.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            pos: Node positions [num_nodes, 3]
            batch: Batch indices [num_nodes]
            atom_types: Optional atom types [num_nodes]
            is_transmembrane: Flag indicating if structure is a transmembrane protein

        Returns:
            dict: Energy components and total energy
        """
        # Node feature embedding
        h = self.node_embedding(x)

        # Apply graph convolutions
        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        # Compute energy components

        e_bond = self._compute_bond_energy(h, edge_index, pos)
        e_angle = self._compute_angle_energy(h, edge_index, pos)
        e_dihedral = self._compute_dihedral_energy(h, edge_index, pos)
        e_nb = self._compute_nonbonded_energy(h, pos, batch)

        # Aggregate energies
        e_total = e_bond + e_angle + e_dihedral + e_nb

        # Evaluate transmembrane domain if applicable
        tm_score = None
        if is_transmembrane:
            tm_score = self.tm_validator(h, pos, batch)
        # Add transmembrane penalty to total energy if score is poor
        if tm_score < 0.5:
            e_total = e_total + (1.0 - tm_score) * 10.0

        return {
            'bond_energy': e_bond,
            'angle_energy': e_angle,
            'dihedral_energy': e_dihedral,
            'nonbonded_energy': e_nb,
            'total_energy': e_total,
            'tm_score': tm_score
        }

    def _compute_bond_energy(self, h, edge_index, pos):
        """Compute bond energy based on node features and positions."""
        # Extract source and target nodes
        src, dst = edge_index

        # Compute bond lengths
        bond_vectors = pos[dst] - pos[src]  # [num_edges, 3]
        bond_lengths = torch.norm(bond_vectors, dim=1)  # [num_edges]

        # Get node embeddings for edges
        h_src = h[src]  # [num_edges, hidden_dim*2]
        h_dst = h[dst]  # [num_edges, hidden_dim*2]

        # Combine features for bond energy prediction
        edge_features = (h_src + h_dst) / 2  # [num_edges, hidden_dim*2]

        # Predict energy per bond
        bond_energies = self.bond_energy_predictor(edge_features).squeeze(-1)  # [num_edges]

        # Calculate bond energy based on deviation from ideal bond length
        # For simplicity, we use a harmonic potential: E = k(r - r0)^2
        ideal_bond_length = 1.5  # Approximate ideal bond length (could be type-specific)
        k_bond = 1000.0  # Force constant
        harmonic_energies = k_bond * (bond_lengths - ideal_bond_length).pow(2)

        # Combine learned and physical energies
        combined_energies = bond_energies + harmonic_energies

        # Return total bond energy
        return combined_energies.sum()

    def _compute_angle_energy(self, h, edge_index, pos):
        """Compute angle energy based on node features and positions."""
        # For simplicity, we'll just return a placeholder value
        # In a real implementation, we would:
        # 1. Find all angle triplets (atoms i-j-k)
        # 2. Compute angles between bonds
        # 3. Calculate energy using harmonic potential: E = k(θ - θ0)^2
        batch_size = torch.max(edge_index[0]).item() + 1
        return torch.tensor(1.0, device=pos.device, requires_grad=True)

    def _compute_dihedral_energy(self, h, edge_index, pos):
        """Compute dihedral angle energy based on node features and positions."""
        # Similar to angle energy, this is a placeholder
        # In a real implementation, we would:
        # 1. Find all dihedral quadruplets (atoms i-j-k-l)
        # 2. Compute dihedral angles
        # 3. Calculate energy using periodic potential: E = k[1 + cos(nφ - δ)]
        batch_size = torch.max(edge_index[0]).item() + 1
        return torch.tensor(1.0, device=pos.device, requires_grad=True)

    def _compute_nonbonded_energy(self, h, pos, batch):
        """Compute non-bonded energy (electrostatics and van der Waals)."""
        # For simplicity, we use a placeholder
        # In a real implementation, we would:
        # 1. Calculate pairwise distances between non-bonded atoms
        # 2. Apply Lennard-Jones potential for van der Waals: E_vdw = 4ε[(σ/r)^12 - (σ/r)^6]
        # 3. Apply Coulomb's law for electrostatics: E_elec = k_e * q_i * q_j / r
        batch_size = torch.max(batch).item() + 1
        return torch.tensor(1.0, device=pos.device, requires_grad=True)


class TransmembraneValidator(nn.Module):
    """Validator for transmembrane domain properties."""

    def __init__(self, input_dim, hidden_dim):
        """
        Initialize transmembrane validator.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
        """
        super(TransmembraneValidator, self).__init__()

        # MLP for hydrophobicity profile analysis
        self.hydrophobicity_analyzer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Helix property analyzer
        self.helix_analyzer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h, pos, batch):
        """
        Validate transmembrane domain properties.

        Args:
            h: Node features [num_nodes, input_dim]
            pos: Node positions [num_nodes, 3]
            batch: Batch indices [num_nodes]

        Returns:
            float: Transmembrane score (0-1, higher is better)
        """
        # Analyze hydrophobicity profile
        hydrophobicity_scores = self.hydrophobicity_analyzer(h)

        # Analyze helix properties
        helix_scores = self.helix_analyzer(h)

        # Combine scores (simple average for now)
        combined_scores = (hydrophobicity_scores + helix_scores) / 2

        # Average over all nodes to get final score
        final_score = combined_scores.mean()

        return final_score