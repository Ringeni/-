import torch


def decode_targets_from_assoc(assoc: torch.Tensor, cam_mask: torch.Tensor, det_scores: torch.Tensor | None = None, prob_threshold: float = 0.5) -> dict:
    if assoc.ndim == 4:
        max_prob_q, pred_slot_q = assoc.max(dim=-1)
        if det_scores is None:
            det_scores = torch.ones_like(max_prob_q)
        quality = max_prob_q * det_scores
        best_q = quality.argmax(dim=-1)
        gather_q = best_q.unsqueeze(-1)
        max_prob = torch.gather(max_prob_q, dim=-1, index=gather_q).squeeze(-1)
        pred_slot = torch.gather(pred_slot_q, dim=-1, index=gather_q).squeeze(-1)
        det_conf = torch.gather(det_scores, dim=-1, index=gather_q).squeeze(-1)
    else:
        max_prob, pred_slot = assoc.max(dim=-1)
        det_conf = torch.ones_like(max_prob)
        best_q = torch.zeros_like(max_prob, dtype=torch.long)
    valid = cam_mask & (max_prob >= prob_threshold)
    target_id = torch.where(valid, pred_slot.to(torch.long), torch.full_like(pred_slot.to(torch.long), fill_value=-1))
    return {
        "pred_slot": pred_slot,
        "max_prob": max_prob,
        "det_conf": det_conf,
        "best_query": best_q,
        "valid": valid,
        "target_id": target_id,
    }
