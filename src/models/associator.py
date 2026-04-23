import torch
from torch import nn


class Associator(nn.Module):
    def __init__(self, emb_dim: int, temperature: float = 0.07) -> None:
        super().__init__()
        self.det_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.track_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self._temperature = temperature

    def forward(self, det_tokens: torch.Tensor, tracks: torch.Tensor, track_conf: torch.Tensor) -> torch.Tensor:
        det_q = self.det_proj(det_tokens)
        track_k = self.track_proj(tracks)
        det_q = torch.nn.functional.normalize(det_q, dim=-1)
        track_k = torch.nn.functional.normalize(track_k, dim=-1)
        logits = torch.matmul(det_q, track_k.unsqueeze(1).transpose(-1, -2)) / self._temperature
        logits = logits + track_conf.unsqueeze(1).unsqueeze(1).clamp_min(1e-6).log()
        return torch.softmax(logits, dim=-1)
