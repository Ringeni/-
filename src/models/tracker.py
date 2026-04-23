import torch
from torch import nn


class Tracker(nn.Module):
    def __init__(self, emb_dim: int, track_slots: int) -> None:
        super().__init__()
        self.track_slots = track_slots
        self.init_tracks = nn.Parameter(torch.randn(track_slots, emb_dim) * 0.02)
        self.update = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.conf_head = nn.Linear(emb_dim, 1)

    def initialize(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        embeddings = self.init_tracks.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        conf = torch.full((batch_size, self.track_slots), 0.5, device=device)
        return {"embeddings": embeddings, "conf": conf}

    def forward(
        self,
        state: dict[str, torch.Tensor],
        det_tokens: torch.Tensor,
        det_scores: torch.Tensor,
        cam_mask: torch.Tensor,
        momentum: float = 0.85,
    ) -> dict[str, torch.Tensor]:
        tracks = state["embeddings"]
        conf = state["conf"]
        batch_size, cam_count, det_count, emb_dim = det_tokens.shape
        det_flat = det_tokens.reshape(batch_size, cam_count * det_count, emb_dim)
        score_flat = det_scores.reshape(batch_size, cam_count * det_count)
        valid_flat = cam_mask.unsqueeze(-1).expand(-1, -1, det_count).reshape(batch_size, cam_count * det_count)
        score_flat = torch.where(valid_flat, score_flat, torch.zeros_like(score_flat))
        det_flat = det_flat * score_flat.unsqueeze(-1)
        update_out, _ = self.update(query=tracks, key=det_flat, value=det_flat)
        updated = self.norm1(tracks + update_out)
        updated = self.norm2(updated + self.ffn(updated))
        merged = momentum * tracks + (1.0 - momentum) * updated
        track_conf = torch.sigmoid(self.conf_head(merged)).squeeze(-1)
        track_conf = 0.5 * conf + 0.5 * track_conf
        return {"embeddings": merged, "conf": track_conf}
