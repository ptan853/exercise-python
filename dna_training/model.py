"""Small, original-style bidirectional Transformer (Post-LN, ReLU)."""
from dataclasses import dataclass
import math
import torch
from torch import nn

@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_length: int = 512

    def validate(self):
        if self.d_model <= 0 or self.d_model % 2 or self.nhead <= 0 or self.d_model % self.nhead:
            raise ValueError('d_model must be positive, even and divisible by nhead')
        if self.num_layers < 1 or self.dim_feedforward < 1 or self.max_length < 2:
            raise ValueError('Invalid model dimensions')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_length):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError('Input exceeds configured position length')
        return x + self.pe[:, :x.shape[1]].to(dtype=x.dtype)

class DNATransformer(nn.Module):
    def __init__(self, vocab_size, num_classes, pad_id, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        c = self.config
        self.d_model = c.d_model
        self.embedding = nn.Embedding(vocab_size, c.d_model, padding_idx=pad_id)
        self.position_encoding = SinusoidalPositionEncoding(c.d_model, c.max_length)
        self.input_dropout = nn.Dropout(c.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=c.d_model, nhead=c.nhead, dim_feedforward=c.dim_feedforward,
            dropout=c.dropout, activation='relu', batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, c.num_layers, enable_nested_tensor=False)
        self.classification_head = nn.Linear(c.d_model, num_classes)
        self.mlm_head = nn.Linear(c.d_model, vocab_size)
        for parameter in self.encoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.input_dropout(self.position_encoding(x))
        hidden = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        return {'class_logits': self.classification_head(hidden[:, 0]),
                'mlm_logits': self.mlm_head(hidden)}
