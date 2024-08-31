import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class LiquidLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, sequence_length: int, dt: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sequence_length = sequence_length
        self.dt = dt
        
        self.W = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.b = nn.Parameter(torch.zeros(out_features))
        
        self.tau = nn.Parameter(torch.ones(out_features))
        
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        x_flat = x.view(-1, self.in_features)
        
        I = F.linear(x_flat, self.W, self.b)
        
        I = I.view(batch_size, seq_len, self.out_features)
        
        u = torch.zeros(batch_size, seq_len + 1, self.out_features, device=x.device)
        s = torch.zeros(batch_size, seq_len + 1, self.out_features, device=x.device)
        
        for t in range(seq_len):
            du = (-u[:, t] + I[:, t]) / self.tau
            ds = (-s[:, t] + u[:, t]) / self.tau
            
            u[:, t+1] = u[:, t] + du * self.dt
            s[:, t+1] = s[:, t] + ds * self.dt
        
        output = torch.tanh(s[:, 1:])
        
        output = self.layer_norm(output)
        
        return output

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_sizes: List[int], sequence_length: int, num_heads: int = 8, dropout: float = 0.1, dt: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_sizes = hidden_sizes
        self.sequence_length = sequence_length
        self.dt = dt

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len=sequence_length)
        
        liquid_layers = []
        input_size = embed_size
        for i, size in enumerate(hidden_sizes):
            liquid_layers.append(LiquidLayer(input_size, size, sequence_length, dt))
            input_size = size
        self.liquid_layers = nn.ModuleList(liquid_layers)

        self.attention = nn.MultiheadAttention(hidden_sizes[-1], num_heads, dropout=dropout)
        self.output_layer = nn.Linear(hidden_sizes[-1], vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.log_dt = nn.Parameter(torch.log(torch.tensor(dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        
        x = self.embedding(x)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        self.dt = torch.exp(self.log_dt)

        for layer in self.liquid_layers:
            layer.dt = self.dt
            x = layer(x)
            x = self.dropout(x)
        
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        x, _ = self.attention(x, x, x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_size)

        output = self.output_layer(x)

        return output

    def generate(self, data_generator, start_text: str, gen_length: int, temperature: float = 0.5) -> str:
        self.eval()
        generated_text = start_text

        current_chars = torch.tensor([data_generator.char_to_idx.get(ch, 0) for ch in start_text], dtype=torch.long).unsqueeze(0)
        current_chars = self._pad_sequence(current_chars)
        current_chars = current_chars.to(self.embedding.weight.device)

        with torch.no_grad():
            for _ in range(gen_length):
                output = self(current_chars)
                output = output[0, -1, :] / temperature
                probs = F.softmax(output, dim=-1)

                next_char_idx = torch.multinomial(probs, 1).item()
                next_char = data_generator.idx_to_char.get(next_char_idx, '')

                generated_text += next_char
                next_char_tensor = torch.tensor([[next_char_idx]], dtype=torch.long, device=self.embedding.weight.device)
                current_chars = torch.cat([current_chars, next_char_tensor], dim=1)
                current_chars = self._pad_sequence(current_chars)

        return generated_text

    def _pad_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.size(1) < self.sequence_length:
            padding = torch.zeros(1, self.sequence_length - sequence.size(1), dtype=torch.long, device=sequence.device)
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