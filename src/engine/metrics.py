import torch


def association_top1_accuracy(assoc: torch.Tensor, ids: torch.Tensor, target_mask: torch.Tensor, cam_mask: torch.Tensor) -> float:
    if assoc.ndim == 4:
        assoc = assoc[:, :, 0, :]
    track_slots = assoc.shape[-1]
    valid = (target_mask.any(dim=-1)) & cam_mask
    if not bool(valid.any()):
        return 0.0
    first_pos = target_mask.float().argmax(dim=-1)
    gather_index = first_pos.unsqueeze(-1)
    selected_ids = torch.gather(ids, dim=-1, index=gather_index).squeeze(-1)
    gt_slots = torch.remainder(selected_ids, track_slots)
    pred_slots = assoc.argmax(dim=-1)
    correct = (pred_slots == gt_slots) & valid
    return float(correct.sum().item() / valid.sum().item())


def association_confidence_mean(assoc: torch.Tensor, cam_mask: torch.Tensor) -> float:
    if assoc.ndim == 4:
        assoc = assoc[:, :, 0, :]
    max_prob = assoc.max(dim=-1).values
    valid = cam_mask
    if not bool(valid.any()):
        return 0.0
    return float(max_prob[valid].mean().item())
