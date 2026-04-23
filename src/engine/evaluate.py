from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import MmpMultiViewDataset, TemporalStepSampler, collate_multiview_batch
from src.losses import StageCCriterion
from src.models import MctrSystem
from .metrics import association_confidence_mean, association_top1_accuracy


@dataclass(frozen=True)
class EvalSummary:
    steps: int
    mean_loss: float
    assoc_top1_acc: float
    assoc_conf_mean: float
    checkpoint_path: Path


def evaluate_checkpoint(config) -> EvalSummary:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MmpMultiViewDataset(
        data_root=Path(config.data_root),
        subset=config.subset,
        split=config.split,
        min_views=config.min_views,
        split_part=config.eval_split_part,
        split_ratio=config.split_ratio,
    )
    if len(dataset) == 0:
        raise RuntimeError("阶段C评估数据集为空，请检查配置")
    sampler = TemporalStepSampler(
        data_source=dataset,
        stride=config.temporal_stride,
        max_steps=config.eval_max_steps,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_multiview_batch,
    )
    checkpoint_path = _resolve_checkpoint_path(config)
    model = MctrSystem(
        in_channels=3,
        emb_dim=config.model_emb_dim,
        track_slots=config.model_track_slots,
        det_queries=config.model_det_queries,
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    criterion = StageCCriterion()
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    conf_sum = 0.0
    steps = 0
    state: dict[str, torch.Tensor] | None = None
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            cam_mask = batch["cam_mask"].to(device)
            outputs = model(images=images, cam_mask=cam_mask, prev_state=state)
            state = {k: v.detach() for k, v in outputs["state"].items()}
            loss_items = criterion(outputs=outputs, batch=batch)
            loss_sum += float(loss_items["loss"].detach().cpu().item())
            acc_sum += association_top1_accuracy(
                assoc=outputs["assoc"],
                ids=batch["ids"].to(device),
                target_mask=batch["target_mask"].to(device),
                cam_mask=cam_mask,
            )
            conf_sum += association_confidence_mean(assoc=outputs["assoc"], cam_mask=cam_mask)
            steps += 1
    if steps == 0:
        raise RuntimeError("阶段C评估没有有效步，请检查 eval_max_steps")
    return EvalSummary(
        steps=steps,
        mean_loss=loss_sum / steps,
        assoc_top1_acc=acc_sum / steps,
        assoc_conf_mean=conf_sum / steps,
        checkpoint_path=checkpoint_path,
    )


def _resolve_checkpoint_path(config) -> Path:
    if config.eval_checkpoint_path.strip():
        candidate = Path(config.eval_checkpoint_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"评估 checkpoint 不存在: {candidate}")
        return candidate
    ckpt_dir = Path(config.checkpoint_dir).expanduser().resolve()
    candidates = sorted(ckpt_dir.glob("stagec_epoch*.pt"))
    if not candidates:
        raise FileNotFoundError(f"未找到可评估 checkpoint: {ckpt_dir}")
    return candidates[-1]
