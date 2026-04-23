import torch
from torch import nn


class Detector(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int, det_queries: int = 16) -> None:
        super().__init__()
        self.det_queries = det_queries
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, emb_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.query_embed = nn.Parameter(torch.randn(det_queries, emb_dim) * 0.02)
        self.query_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True)
        self.query_norm = nn.LayerNorm(emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.score_head = nn.Linear(emb_dim, 1)
        self.box_head = nn.Linear(emb_dim, 4)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        memory = features.flatten(2).transpose(1, 2)
        query = self.query_embed.unsqueeze(0).expand(images.shape[0], -1, -1)
        query_out, _ = self.query_attn(query=query, key=memory, value=memory)
        tokens = self.query_norm(self.proj(query + query_out))
        scores = torch.sigmoid(self.score_head(tokens)).squeeze(-1)
        boxes = torch.sigmoid(self.box_head(tokens))
        return {"tokens": tokens, "scores": scores, "boxes": boxes}
