import torch
from torch import nn

from .associator import Associator
from .detector import Detector
from .tracker import Tracker


class MctrSystem(nn.Module):
    def __init__(self, in_channels: int = 3, emb_dim: int = 128, track_slots: int = 64, det_queries: int = 16) -> None:
        super().__init__()
        self.detector = Detector(in_channels=in_channels, emb_dim=emb_dim, det_queries=det_queries)
        self.tracker = Tracker(emb_dim=emb_dim, track_slots=track_slots)
        self.associator = Associator(emb_dim=emb_dim)
        self.det_queries = det_queries

    def forward(self, images: torch.Tensor, cam_mask: torch.Tensor, prev_state: dict[str, torch.Tensor] | None = None) -> dict:
        batch_size, cam_count, channels, height, width = images.shape
        flat_images = images.reshape(batch_size * cam_count, channels, height, width)
        det_out = self.detector(flat_images)
        det_tokens = det_out["tokens"].reshape(batch_size, cam_count, self.det_queries, -1)
        det_scores = det_out["scores"].reshape(batch_size, cam_count, self.det_queries)
        det_boxes = det_out["boxes"].reshape(batch_size, cam_count, self.det_queries, 4)
        det_scores = det_scores * cam_mask.unsqueeze(-1).to(det_scores.dtype)
        det_tokens = det_tokens * det_scores.unsqueeze(-1)
        state = prev_state
        if state is None:
            state = self.tracker.initialize(batch_size=batch_size, device=images.device)
        updated_state = self.tracker(state=state, det_tokens=det_tokens, det_scores=det_scores, cam_mask=cam_mask)
        assoc = self.associator(det_tokens=det_tokens, tracks=updated_state["embeddings"], track_conf=updated_state["conf"])
        return {
            "det_tokens": det_tokens,
            "det_scores": det_scores,
            "det_boxes": det_boxes,
            "tracks": updated_state["embeddings"],
            "track_conf": updated_state["conf"],
            "assoc": assoc,
            "state": updated_state,
        }
