import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

    def forward(self, template_atoms, template_positions, target_function=None, torsion_angles=None, mask=None):
        """
        Forward pass of transformer generator.

        Args:
            template_atoms: Template atom types [batch_size, seq_length]
            template_positions: Template positions [batch_size, seq_length, 3]
            target_function: Optional target function encoding [batch_size, d_model]
            torsion_angles: Optional torsion angle constraints [batch_size, num_torsions, 1]
            mask: Optional mask [batch_size, seq_length]

        Returns:
            tuple: (Generated atom types, Generated positions, Generated bonds)
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
        memory = self.transformer.encoder(src, src_key_padding_mask=~mask if mask is not None else None)
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=~mask if mask is not None else None,
                                          memory_key_padding_mask=~mask if mask is not None else None)

        # Generate outputs
        atom_logits = self.atom_type_projection(output)  # [batch_size, seq_length, num_atom_types]
        positions = self.position_projection(output)  # [batch_size, seq_length, 3]
        bond_logits = self.bond_projection(output)  # [batch_size, seq_length, 3]

        # Apply torsion angle constraints if provided
        if torsion_angles is not None:
            positions = self._apply_torsion_constraints(positions, torsion_angles)

        return atom_logits, positions, bond_logits

    def _get_position_features(self, positions):
        """Convert 3D positions to feature space."""
        batch_size, seq_length, _ = positions.shape

        # Scale positions for better numerical stability
        scaled_pos = positions / 10.0

        # Create position features in various ways
        pos_feat = torch.zeros(batch_size, seq_length, self.d_model, device=positions.device)

        # Use sine/cosine functions of different frequencies
        for i in range(0, self.d_model, 6):
            if i < self.d_model - 5:
                div_term = 10000.0 ** (torch.arange(0, 3, dtype=torch.float, device=positions.device) / self.d_model)

                # Encode x, y, z with different frequency sinusoids
                pos_feat[:, :, i:i + 3] = scaled_pos * div_term
                pos_feat[:, :, i + 3:i + 6] = torch.sin(scaled_pos * div_term)

        return pos_feat

    def _generate_square_subsequent_mask(self, sz):
        """Generate mask for autoregressive generation."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _apply_torsion_constraints(self, positions, torsion_angles):
        """Apply torsion angle constraints to positions."""
        # This would involve rotating substructures according to the specified torsion angles
        # For simplicity, we'll just return the original positions
        # A full implementation would apply rotations based on torsion bonds
        return positions


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

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
