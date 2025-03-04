import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class TransformerGenerator(nn.Module):
    """Transformer-based generator for receptor structure synthesis."""

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_seq_length=1000,
                 num_atom_types=20):
        """
        Initialize transformer generator.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            num_atom_types: Number of atom types
        """
        super(TransformerGenerator, self).__init__()

        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_atom_types = num_atom_types

        # Embedding layers
        self.atom_embedding = nn.Embedding(num_atom_types, d_model)
        self.position_embedding = PositionalEncoding(d_model, dropout, max_seq_length)

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projections
        self.atom_type_projection = nn.Linear(d_model, num_atom_types)
        self.position_projection = nn.Linear(d_model, 3)  # x, y, z coordinates
        self.bond_projection = nn.Linear(d_model, 3)  # Bond types (single, double, triple)

        # Conditional control projection
        self.condition_projection = nn.Linear(d_model, d_model)
        
        # Torsion angle prediction and constraint layers
        self.torsion_projection = nn.Linear(d_model, 4)  # phi, psi, omega, chi1
        self.torsion_constraint_encoder = nn.Linear(4, d_model)

    def forward(self, template_atoms, template_positions, target_function=None, torsion_angles=None, mask=None):
        """
        Forward pass of transformer generator.

        Args:
            template_atoms: Template atom types [batch_size, seq_length]
            template_positions: Template positions [batch_size, seq_length, 3]
            target_function: Optional target function encoding [batch_size, d_model]
            torsion_angles: Optional torsion angle constraints [batch_size, num_torsions, 4]
            mask: Optional mask [batch_size, seq_length]

        Returns:
            tuple: (Generated atom types, Generated positions, Generated bonds, Generated torsion angles)
        """
        batch_size, seq_length = template_atoms.shape
        device = template_atoms.device

        # Embed atom types
        atom_embeddings = self.atom_embedding(template_atoms)  # [batch_size, seq_length, d_model]

        # Add positional encoding
        src = self.position_embedding(atom_embeddings)  # [batch_size, seq_length, d_model]

        # Create positional features from 3D coordinates
        pos_features = self._get_position_features(template_positions)  # [batch_size, seq_length, d_model]
        src = src + pos_features

        # If torsion angle constraints provided, incorporate them
        if torsion_angles is not None:
            torsion_features = self._encode_torsion_constraints(torsion_angles, seq_length)
            src = src + torsion_features

        # If target function provided, use it to condition the generation
        if target_function is not None:
            # Project target function to source dimension
            func_feature = self.condition_projection(target_function).unsqueeze(1)  # [batch_size, 1, d_model]
            # Concatenate to source sequence
            src = torch.cat([func_feature, src], dim=1)  # [batch_size, seq_length+1, d_model]

            # Adjust mask if needed
            if mask is not None:
                func_mask = torch.ones(batch_size, 1, device=device)
                mask = torch.cat([func_mask, mask], dim=1)

        # Create target for autoregressive generation (shifted input)
        tgt = src.clone()

        # Generate mask for autoregressive generation
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)

        # Apply transformer
        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=None if mask is None else ~mask.bool(),
            tgt_key_padding_mask=None if mask is None else ~mask.bool()
        )

        # Project outputs to atom types, positions, and bonds
        atom_logits = self.atom_type_projection(output)
        positions = self.position_projection(output)
        bonds = self.bond_projection(output)
        
        # Project to torsion angles
        torsion_angles_pred = self.torsion_projection(output)
        
        # Apply physical constraints to torsion angles
        torsion_angles_pred = self._apply_torsion_constraints(torsion_angles_pred)

        return atom_logits, positions, bonds, torsion_angles_pred

    def _get_position_features(self, positions):
        """
        Convert 3D positions to feature vectors.

        Args:
            positions: [batch_size, seq_length, 3]

        Returns:
            torch.Tensor: Position features [batch_size, seq_length, d_model]
        """
        batch_size, seq_length, _ = positions.shape
        device = positions.device

        # Calculate pairwise distances between atoms
        pos_i = positions.unsqueeze(2)  # [batch_size, seq_length, 1, 3]
        pos_j = positions.unsqueeze(1)  # [batch_size, 1, seq_length, 3]
        dist_ij = torch.norm(pos_i - pos_j, dim=-1)  # [batch_size, seq_length, seq_length]

        # Create distance features using sinusoidal encoding
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) * -(math.log(10000.0) / self.d_model))
        dist_enc = torch.zeros(batch_size, seq_length, seq_length, self.d_model, device=device)

        # Apply sinusoidal encoding to distances
        dist_enc[:, :, :, 0::2] = torch.sin(dist_ij.unsqueeze(-1) * div_term)
        dist_enc[:, :, :, 1::2] = torch.cos(dist_ij.unsqueeze(-1) * div_term)

        # Aggregate distance features
        pos_features = torch.mean(dist_enc, dim=2)  # [batch_size, seq_length, d_model]

        return pos_features

    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence.

        Args:
            sz: Sequence length

        Returns:
            torch.Tensor: Mask tensor
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _encode_torsion_constraints(self, torsion_angles, seq_length):
        """
        Encode torsion angle constraints into features.
        
        Args:
            torsion_angles: [batch_size, num_torsions, 4] (phi, psi, omega, chi1)
            seq_length: Length of sequence
            
        Returns:
            torch.Tensor: Torsion features [batch_size, seq_length, d_model]
        """
        batch_size = torsion_angles.shape[0]
        device = torsion_angles.device
        
        # Convert torsion angles to features
        torsion_features = self.torsion_constraint_encoder(torsion_angles)  # [batch_size, num_torsions, d_model]
        
        # Create a mapping from torsion indices to sequence positions
        # Assuming torsion_angles has a corresponding position index
        # For simplicity, we'll assume torsion angles are provided for all residues
        if torsion_angles.shape[1] == seq_length:
            return torsion_features
        else:
            # If torsion angles are provided for only some residues,
            # we need to map them to the correct positions
            full_features = torch.zeros(batch_size, seq_length, self.d_model, device=device)
            # This is a simplified mapping - in practice, you would need to know which
            # torsion angle corresponds to which residue position
            num_torsions = min(torsion_angles.shape[1], seq_length)
            full_features[:, :num_torsions, :] = torsion_features[:, :num_torsions, :]
            return full_features
    
    def _apply_torsion_constraints(self, torsion_angles_pred):
        """
        Apply physical constraints to predicted torsion angles.
        
        Args:
            torsion_angles_pred: [batch_size, seq_length, 4] (phi, psi, omega, chi1)
            
        Returns:
            torch.Tensor: Constrained torsion angles [batch_size, seq_length, 4]
        """
        # Convert to radians in range [-π, π]
        torsion_angles_rad = torch.tanh(torsion_angles_pred) * math.pi
        
        # Apply Ramachandran constraints (simplified)
        # In a real implementation, you would use a more sophisticated approach
        # such as energy-based constraints or learned distributions
        
        # Example: Constrain omega angle (peptide bond) to be near 180 degrees (π radians)
        # Omega is typically close to 180 degrees (trans configuration)
        omega = torsion_angles_rad[:, :, 2:3]  # Extract omega angle
        omega_constrained = torch.sign(omega) * (0.9 * math.pi + 0.1 * torch.abs(omega))
        
        # Replace omega in the predicted angles
        torsion_angles_constrained = torsion_angles_rad.clone()
        torsion_angles_constrained[:, :, 2:3] = omega_constrained
        
        return torsion_angles_constrained


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch_size, seq_length, d_model]

        Returns:
            torch.Tensor: Output with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
