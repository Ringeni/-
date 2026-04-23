import torch
from torch import nn
import torch.nn.functional as F


class StageCCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, outputs: dict, batch: dict) -> dict:
        det_tokens = outputs["det_tokens"]
        det_scores = outputs["det_scores"]
        det_boxes = outputs["det_boxes"]
        assoc = outputs["assoc"]
        cam_mask = batch["cam_mask"].to(det_tokens.device)
        target_mask = batch["target_mask"].to(det_tokens.device)
        ids = batch["ids"].to(det_tokens.device)
        boxes = batch["boxes"].to(det_tokens.device)
        det_norm = det_tokens.pow(2).mean()
        det_supervision = _build_detector_loss(det_scores=det_scores, det_boxes=det_boxes, boxes=boxes, target_mask=target_mask, cam_mask=cam_mask, image_hw=batch["images"].shape[-2:])
        assoc_entropy = -(assoc.clamp_min(1e-8).log() * assoc).sum(dim=-1)
        valid_assoc = cam_mask.unsqueeze(-1).expand_as(assoc_entropy)
        assoc_entropy_loss = assoc_entropy[valid_assoc].mean() if bool(valid_assoc.any()) else torch.zeros((), device=det_tokens.device)
        assoc_pair_loss = _build_assoc_supervised_loss(assoc=assoc, ids=ids, target_mask=target_mask, cam_mask=cam_mask)
        assoc_consistency = _build_cross_view_consistency_loss(assoc=assoc, ids=ids, target_mask=target_mask, cam_mask=cam_mask)
        total_loss = det_norm + det_supervision + assoc_entropy_loss + assoc_pair_loss + assoc_consistency
        return {
            "loss": total_loss,
            "det_norm_loss": det_norm.detach(),
            "det_supervision_loss": det_supervision.detach(),
            "assoc_entropy_loss": assoc_entropy_loss.detach(),
            "assoc_pair_loss": assoc_pair_loss.detach(),
            "assoc_consistency_loss": assoc_consistency.detach(),
        }


def _build_assoc_supervised_loss(assoc: torch.Tensor, ids: torch.Tensor, target_mask: torch.Tensor, cam_mask: torch.Tensor) -> torch.Tensor:
    target_ids, valid = _query_ids_from_targets(ids=ids, target_mask=target_mask, cam_mask=cam_mask, max_queries=assoc.shape[2])
    if not bool(valid.any()):
        return torch.zeros((), device=assoc.device)
    track_slots = assoc.shape[-1]
    target_slots = torch.remainder(target_ids.clamp_min(0), track_slots)
    log_prob = assoc.clamp_min(1e-8).log().reshape(-1, track_slots)
    target_flat = target_slots.reshape(-1).long()
    valid_flat = valid.reshape(-1)
    loss_all = F.nll_loss(log_prob, target_flat, reduction="none")
    return loss_all[valid_flat].mean()


def _build_cross_view_consistency_loss(assoc: torch.Tensor, ids: torch.Tensor, target_mask: torch.Tensor, cam_mask: torch.Tensor) -> torch.Tensor:
    target_ids, valid = _query_ids_from_targets(ids=ids, target_mask=target_mask, cam_mask=cam_mask, max_queries=assoc.shape[2])
    if not bool(valid.any()):
        return torch.zeros((), device=assoc.device)
    probs = assoc
    bsz, cam_count, query_count, _ = probs.shape
    loss_terms: list[torch.Tensor] = []
    for b in range(bsz):
        for c1 in range(cam_count):
            for c2 in range(c1 + 1, cam_count):
                if not bool(cam_mask[b, c1]) or not bool(cam_mask[b, c2]):
                    continue
                ids1 = target_ids[b, c1]
                ids2 = target_ids[b, c2]
                valid1 = valid[b, c1]
                valid2 = valid[b, c2]
                for q1 in range(query_count):
                    if not bool(valid1[q1]):
                        continue
                    same = (ids2 == ids1[q1]) & valid2
                    if not bool(same.any()):
                        continue
                    ref = probs[b, c1, q1]
                    peer = probs[b, c2, same].mean(dim=0)
                    loss_terms.append(F.mse_loss(ref, peer))
    if not loss_terms:
        return torch.zeros((), device=assoc.device)
    return torch.stack(loss_terms).mean()


def _query_ids_from_targets(ids: torch.Tensor, target_mask: torch.Tensor, cam_mask: torch.Tensor, max_queries: int) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, cam_count, max_targets = ids.shape
    device = ids.device
    query_ids = torch.full((bsz, cam_count, max_queries), fill_value=-1, dtype=torch.long, device=device)
    query_valid = torch.zeros((bsz, cam_count, max_queries), dtype=torch.bool, device=device)
    for b in range(bsz):
        for c in range(cam_count):
            if not bool(cam_mask[b, c]):
                continue
            valid_target = target_mask[b, c]
            if not bool(valid_target.any()):
                continue
            candidate = ids[b, c][valid_target]
            take = min(max_queries, int(candidate.shape[0]))
            query_ids[b, c, :take] = candidate[:take]
            query_valid[b, c, :take] = True
    return query_ids, query_valid


def _build_detector_loss(
    det_scores: torch.Tensor,
    det_boxes: torch.Tensor,
    boxes: torch.Tensor,
    target_mask: torch.Tensor,
    cam_mask: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    bsz, cam_count, query_count = det_scores.shape
    obj_target = torch.zeros_like(det_scores)
    box_target = torch.zeros_like(det_boxes)
    box_weight = torch.zeros_like(det_scores)
    img_h, img_w = image_hw
    norm = torch.tensor([img_w, img_h, img_w, img_h], dtype=det_boxes.dtype, device=det_boxes.device).view(1, 1, 1, 4)
    box_gt = boxes / norm
    for b in range(bsz):
        for c in range(cam_count):
            if not bool(cam_mask[b, c]):
                continue
            valid_target = target_mask[b, c]
            if not bool(valid_target.any()):
                continue
            candidate = box_gt[b, c][valid_target]
            take = min(query_count, int(candidate.shape[0]))
            obj_target[b, c, :take] = 1.0
            box_target[b, c, :take] = candidate[:take].clamp(0.0, 1.0)
            box_weight[b, c, :take] = 1.0
    valid_cam = cam_mask.unsqueeze(-1).expand_as(det_scores)
    score_loss = F.binary_cross_entropy(det_scores.clamp(1e-6, 1 - 1e-6)[valid_cam], obj_target[valid_cam]) if bool(valid_cam.any()) else torch.zeros((), device=det_scores.device)
    valid_box = box_weight.unsqueeze(-1).expand_as(det_boxes) > 0
    box_loss = F.l1_loss(det_boxes[valid_box], box_target[valid_box]) if bool(valid_box.any()) else torch.zeros((), device=det_scores.device)
    return score_loss + box_loss
