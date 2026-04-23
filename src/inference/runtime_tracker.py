from dataclasses import dataclass

import torch

from .decode import decode_targets_from_assoc


@dataclass(frozen=True)
class TrackerStepOutput:
    target_id: torch.Tensor
    pred_slot: torch.Tensor
    max_prob: torch.Tensor
    det_conf: torch.Tensor
    best_query: torch.Tensor


class RuntimeTracker:
    def __init__(self, prob_threshold: float = 0.5) -> None:
        if not 0.0 <= prob_threshold <= 1.0:
            raise ValueError("prob_threshold 必须在 [0,1]")
        self._prob_threshold = prob_threshold

    def step(self, assoc: torch.Tensor, cam_mask: torch.Tensor, det_scores: torch.Tensor | None = None) -> TrackerStepOutput:
        decoded = decode_targets_from_assoc(assoc=assoc, cam_mask=cam_mask, det_scores=det_scores, prob_threshold=self._prob_threshold)
        return TrackerStepOutput(
            target_id=decoded["target_id"],
            pred_slot=decoded["pred_slot"],
            max_prob=decoded["max_prob"],
            det_conf=decoded["det_conf"],
            best_query=decoded["best_query"],
        )
