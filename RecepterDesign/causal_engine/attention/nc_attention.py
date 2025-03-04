import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedCausalAttention(nn.Module):
    """Normalized causal attention mechanism with temperature control."""

    def __init__(self, embed_dim, num_heads=8, temperature=1.0, dropout=0.1):
        """
        Initialize normalized causal attention module.

        Args:
            embed_dim: Dimension of embedding
            num_heads: Number of attention heads
            temperature: Temperature parameter for attention scaling
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = embed_dim // num_heads

        # Check if dimensions are compatible
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Causal score projection
        self.causal_score_proj = nn.Linear(embed_dim, num_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, causal_weights=None, attn_mask=None):
        """
        Forward pass of normalized causal attention.

        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            causal_weights: Optional tensor of causal edge weights [batch_size, seq_len, seq_len]
            attn_mask: Optional attention mask [batch_size, seq_len, seq_len]

        Returns:
            tuple: (Output tensor, attention weights)
        """
        batch_size, q_len, _ = query.size()
        k_len = key.size(1)
        v_len = value.size(1)

        # Project and reshape for multi-head attention
        q = self.q_proj(query).reshape(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, v_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # If causal weights provided, incorporate them
        if causal_weights is not None:
            # Generate causal scores
            causal_scores = self.causal_score_proj(query)  # [batch_size, seq_len, num_heads]
            causal_scores = causal_scores.transpose(1, 2).unsqueeze(-1)  # [batch_size, num_heads, seq_len, 1]

            # Reshape causal weights for broadcasting
            if causal_weights.dim() == 3:  # [batch_size, seq_len, seq_len]
                causal_weights = causal_weights.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            # Apply temperature scaling to causal scores
            causal_weights = causal_weights / self.temperature

            # Combine with attention scores
            scores = scores + causal_weights

        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        # Normalize attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project output
        output = output.transpose(1, 2).reshape(batch_size, q_len, self.embed_dim)
        output = self.out_proj(output)

        return output, attn_weights


class CausalGraphAttention(nn.Module):
    """Graph attention network with causal weighting."""

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, temperature=1.0):
        """
        Initialize causal graph attention module.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
            temperature: Temperature for attention softmax
        """
        super(CausalGraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.temperature = temperature

        # Feature transformation
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        # Causal score projection
        self.causal_proj = nn.Linear(in_features, 1, bias=False)

        # Activations
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, adj, causal_weights=None):
        """
        Forward pass of causal graph attention.

        Args:
            x: Node feature matrix [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            causal_weights: Optional causal edge weights [num_nodes, num_nodes]

        Returns:
            torch.Tensor: Output node features
        """
        num_nodes = x.size(0)

        # Apply feature transformation
        h = self.W(x)  # [num_nodes, out_features]

        # Prepare for attention calculation
        a_input = torch.cat([h.repeat(1, num_nodes).view(num_nodes * num_nodes, -1),
                             h.repeat(num_nodes, 1)], dim=1)
        a_input = a_input.view(num_nodes, num_nodes, 2 * self.out_features)

        # Calculate attention coefficients
        e = self.leakyrelu(self.a(a_input).squeeze(-1))

        # Add causal weighting if provided
        if causal_weights is not None:
            # Generate causal scores
            causal_scores = self.causal_proj(x).squeeze(-1)  # [num_nodes]
            # Apply to causal weights matrix
            causal_effect = causal_scores.unsqueeze(0) - causal_scores.unsqueeze(1)
            causal_effect = causal_effect * causal_weights / self.temperature
            e = e + causal_effect

        # Apply adjacency mask and normalization
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Normalize with softmax
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)

        # Apply attention to features
        h_prime = torch.matmul(attention, h)

        return h_prime
