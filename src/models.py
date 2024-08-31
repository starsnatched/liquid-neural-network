import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import List, Tuple

class FlashAttention2(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        output = self.out_proj(context_layer)
        return output, attention_probs.mean(dim=1)

class LiquidLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, sequence_length: int, activation: callable = torch.tanh):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sequence_length = sequence_length
        self.weights = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.tau = nn.Parameter(torch.ones(out_features))
        self.register_buffer('prev_state', torch.zeros(sequence_length, out_features))
        self.activation = activation

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        # logging.debug(f"LiquidLayer input shape: {x.shape}, weights shape: {self.weights.shape}")
        if x.dim() == 3:  # (batch_size, sequence_length, in_features)
            batch_size, seq_len, _ = x.size()
            x_flat = x.view(-1, self.in_features)
            output_flat = F.linear(x_flat, self.weights, self.bias)
            output = output_flat.view(batch_size, seq_len, self.out_features)
        else:
            output = F.linear(x, self.weights, self.bias)
        
        dx = (-self.prev_state + self.activation(output)) / self.tau
        new_state = self.prev_state + dx * dt
        self.prev_state = new_state.detach()
        return new_state

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_sizes: List[int], sequence_length: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_sizes = hidden_sizes
        self.sequence_length = sequence_length

        logging.info(f"Initializing LiquidNeuralNetwork with vocab_size={vocab_size}, embed_size={embed_size}, hidden_sizes={hidden_sizes}, sequence_length={sequence_length}")

        self.embedding = nn.Embedding(vocab_size, embed_size)

        liquid_layers = []
        input_size = embed_size
        for i, size in enumerate(hidden_sizes):
            liquid_layers.append(LiquidLayer(input_size, size, sequence_length, torch.tanh))
            input_size = size
        self.liquid_layers = nn.ModuleList(liquid_layers)

        self.attention = FlashAttention2(hidden_sizes[-1], num_heads, dropout)
        self.output_layer = nn.Linear(hidden_sizes[-1], vocab_size)

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        # logging.debug(f"Input shape: {x.shape}")
        x = self.embedding(x)
        # logging.debug(f"After embedding shape: {x.shape}")

        hidden_states = []
        for i, layer in enumerate(self.liquid_layers):
            x = layer(x, dt)
            # logging.debug(f"After liquid layer {i} shape: {x.shape}")
            hidden_states.append(x)

        context_vector, attention_weights = self.attention(hidden_states[-1])
        # logging.debug(f"After attention shape: {context_vector.shape}")
        output = self.output_layer(context_vector)
        # logging.debug(f"Output shape: {output.shape}")
        return output, hidden_states, attention_weights

    def generate(self, data_generator, start_text: str, gen_length: int, temperature: float = 0.5) -> str:
        self.eval()
        generated_text = start_text

        current_chars = torch.tensor([data_generator.char_to_idx.get(ch, 0) for ch in start_text], dtype=torch.long).unsqueeze(0)
        current_chars = self._pad_sequence(current_chars)

        with torch.no_grad():
            for _ in range(gen_length):
                output, _, _ = self(current_chars)
                output = output[0, -1, :] / temperature
                probs = F.softmax(output, dim=-1)

                next_char_idx = torch.multinomial(probs, 1).item()
                next_char = data_generator.idx_to_char.get(next_char_idx, '')

                generated_text += next_char
                current_chars = torch.cat([current_chars, torch.tensor([[next_char_idx]], dtype=torch.long)], dim=1)
                current_chars = self._pad_sequence(current_chars)

        return generated_text

    def _pad_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.size(1) < self.sequence_length:
            padding = torch.zeros(1, self.sequence_length - sequence.size(1), dtype=torch.long)
            return torch.cat([padding, sequence], dim=1)
        return sequence[:, -self.sequence_length:]

    def expand_embedding(self, new_vocab_size: int) -> None:
        old_vocab_size, embed_size = self.embedding.weight.shape
        if new_vocab_size > old_vocab_size:
            new_embedding = nn.Embedding(new_vocab_size, embed_size)
            new_embedding.weight.data[:old_vocab_size] = self.embedding.weight.data
            self.embedding = new_embedding
            self.output_layer = nn.Linear(self.hidden_sizes[-1], new_vocab_size)
            self.vocab_size = new_vocab_size
            logging.info(f"Expanded embedding layer from {old_vocab_size} to {new_vocab_size}")