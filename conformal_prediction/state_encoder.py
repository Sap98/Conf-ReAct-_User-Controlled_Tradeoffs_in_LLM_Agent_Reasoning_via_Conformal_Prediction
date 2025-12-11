import torch
import torch.nn as nn
from typing import Optional


class StateEncoder(nn.Module):
    """
    2-layer Transformer encoder that processes a state sequence:
      - Input: [location_embedding, action_1_emb, action_2_emb, ..., action_N_emb] (padded to max_actions)
      - Output: A single state embedding vector

    The transformer attends over the entire sequence and produces a contextualized
    representation of the state.
    """

    def __init__(
        self,
        d_model: int = 768,           # BERT hidden size
        nhead: int = 8,               # Number of attention heads
        num_layers: int = 2,          # Number of transformer layers
        dim_feedforward: int = 2048,  # FFN dimension
        dropout: float = 0.1,         # Dropout rate
        pooling_method: str = "cls"   # How to get final state embedding: "cls", "mean", "max"
    ):
        super().__init__()

        self.d_model = d_model
        self.pooling_method = pooling_method

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU is common in modern transformers
            batch_first=True    # Input shape: (batch, seq, feature)
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Optional: Add a final projection layer for the output embedding
        self.output_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        state_sequences: torch.Tensor,  # (batch_size, seq_len, d_model)
        state_masks: torch.Tensor,      # (batch_size, seq_len) - 1 for real, 0 for padding
    ) -> torch.Tensor:
        """
        Args:
            state_sequences: (batch_size, seq_len, d_model) where seq_len = 1 + max_actions
                            First position is location embedding, rest are action embeddings
            state_masks: (batch_size, seq_len) - 1 for valid positions, 0 for padding

        Returns:
            state_embeddings: (batch_size, d_model) - Final state representation
        """
        batch_size, seq_len, _ = state_sequences.shape

        # Create attention mask for transformer
        # PyTorch transformer expects: True for positions to IGNORE, False for positions to ATTEND
        # So we need to invert our mask (we have 1=valid, 0=pad)
        attn_mask = (state_masks == 0)  # (batch_size, seq_len) - True for padding

        # Pass through transformer encoder
        # The transformer will use the attention mask to ignore padded positions
        transformer_output = self.transformer_encoder(
            state_sequences,
            src_key_padding_mask=attn_mask  # (batch, seq_len)
        )  # (batch_size, seq_len, d_model)

        # Pool to get final state embedding
        if self.pooling_method == "cls":
            # Use the first token (location token) as the state representation
            state_emb = transformer_output[:, 0, :]  # (batch_size, d_model)

        elif self.pooling_method == "mean":
            # Mean pooling over non-masked positions
            # Expand mask for broadcasting: (batch, seq_len, 1)
            mask_expanded = state_masks.unsqueeze(-1).float()

            # Sum over valid positions
            summed = torch.sum(transformer_output * mask_expanded, dim=1)  # (batch, d_model)

            # Divide by number of valid positions
            counts = torch.sum(mask_expanded, dim=1).clamp(min=1)  # (batch, 1)
            state_emb = summed / counts  # (batch, d_model)

        elif self.pooling_method == "max":
            # Max pooling over non-masked positions
            # Set padded positions to very negative value before max
            mask_expanded = state_masks.unsqueeze(-1).float()
            masked_output = transformer_output.clone()
            masked_output[state_masks == 0] = float('-inf')

            state_emb, _ = torch.max(masked_output, dim=1)  # (batch, d_model)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # Optional: Apply final projection and layer norm
        state_emb = self.output_projection(state_emb)
        state_emb = self.layer_norm(state_emb)

        return state_emb  # (batch_size, d_model)


class StateEncoderWithPositionalEncoding(nn.Module):
    """
    Enhanced version with learnable positional encodings.
    Since we have a structured sequence (location first, then actions in temporal order),
    positional information might be helpful.
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 31,  # 1 location + 30 actions
        pooling_method: str = "cls"
    ):
        super().__init__()

        self.d_model = d_model
        self.pooling_method = pooling_method

        # Learnable positional encodings
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        state_sequences: torch.Tensor,
        state_masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = state_sequences.shape

        # Add positional encodings
        state_sequences = state_sequences + self.pos_encoding[:, :seq_len, :]
        state_sequences = self.dropout(state_sequences)

        # Create attention mask
        attn_mask = (state_masks == 0)

        # Transformer encoding
        transformer_output = self.transformer_encoder(
            state_sequences,
            src_key_padding_mask=attn_mask
        )

        # Pool to get state embedding (same as base class)
        if self.pooling_method == "cls":
            state_emb = transformer_output[:, 0, :]
        elif self.pooling_method == "mean":
            mask_expanded = state_masks.unsqueeze(-1).float()
            summed = torch.sum(transformer_output * mask_expanded, dim=1)
            counts = torch.sum(mask_expanded, dim=1).clamp(min=1)
            state_emb = summed / counts
        elif self.pooling_method == "max":
            mask_expanded = state_masks.unsqueeze(-1).float()
            masked_output = transformer_output.clone()
            masked_output[state_masks == 0] = float('-inf')
            state_emb, _ = torch.max(masked_output, dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # Final projection
        state_emb = self.output_projection(state_emb)
        state_emb = self.layer_norm(state_emb)

        return state_emb


# Utility function to create model with default settings
def create_state_encoder(
    use_positional_encoding: bool = False,
    d_model: int = 768,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a state encoder.

    Args:
        use_positional_encoding: Whether to use learnable positional encodings
        d_model: Hidden dimension (should match BERT's output)
        **kwargs: Additional arguments passed to the model

    Returns:
        StateEncoder model instance
    """
    if use_positional_encoding:
        return StateEncoderWithPositionalEncoding(d_model=d_model, **kwargs)
    else:
        return StateEncoder(d_model=d_model, **kwargs)
