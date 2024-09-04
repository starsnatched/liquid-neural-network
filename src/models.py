import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import List

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
        
        self.liquid_layers = nn.ModuleList([
            LiquidLayer(
                in_features=embed_size if i == 0 else hidden_sizes[i-1],
                out_features=size,
                sequence_length=sequence_length,
                dt=dt
            ) for i, size in enumerate(hidden_sizes)
        ])

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