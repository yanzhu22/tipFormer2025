import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln = nn.LayerNorm(embed_dim)
        self.ff_self = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, context):
        attn_output, _ = self.mha(query=x, key=context, value=context)
        x = self.ln(x + attn_output)
        x = self.ln(x + self.ff_self(x))
        return x

class TipFormer(nn.Module):
    def __init__(self, chem_embed_dim=384, protein_embed_dim=1024, hidden_dim=32, num_heads=8):
        super().__init__()
        self.chem_encoder = nn.Sequential(
            nn.Conv1d(chem_embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.protein_encoder = nn.Sequential(
            nn.Conv1d(protein_embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.cross_attn = CrossAttention(hidden_dim, num_heads)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, chem_emb, protein_emb):
        chem_emb = self.chem_encoder(chem_emb.permute(0, 2, 1)).permute(0, 2, 1)
        protein_emb = self.protein_encoder(protein_emb.permute(0, 2, 1)).permute(0, 2, 1)
        interaction = self.cross_attn(chem_emb, protein_emb)
        pooled = torch.mean(interaction, dim=1)
        return self.predictor(pooled)
      
