from dataclasses import dataclass
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from src.data import MmpMultiViewDataset, TemporalStepSampler, collate_multiview_batch
from src.losses import StageCCriterion
from src.models import MctrSystem


@dataclass(frozen=True)
class TrainSummary:
    epoch: int
    steps: int
    mean_loss: float
    det_norm_loss: float
    det_supervision_loss: float
    assoc_entropy_loss: float
    assoc_pair_loss: float
    assoc_consistency_loss: float
    elapsed_sec: float
    checkpoint_path: Path


def train_one_epoch(config) -> TrainSummary:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MmpMultiViewDataset(
        data_root=Path(config.data_root),
        subset=config.subset,
        split=config.split,
        min_views=config.min_views,
        split_part=config.split_part,
        split_ratio=config.split_ratio,
    )
    if len(dataset) == 0:
        raise RuntimeError("阶段C数据集为空，请检查配置")
    sampler = TemporalStepSampler(data_source=dataset, stride=config.temporal_stride, max_steps=config.max_steps)
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_multiview_batch,
    )
    model = MctrSystem(
        in_channels=3,
        emb_dim=config.model_emb_dim,
        track_slots=config.model_track_slots,
        det_queries=config.model_det_queries,
    ).to(device)
    criterion = StageCCriterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    model.train()
    start = time.time()
    loss_sum = 0.0
    det_norm_sum = 0.0
    det_supervision_sum = 0.0
    assoc_entropy_sum = 0.0
    assoc_pair_sum = 0.0
    assoc_consistency_sum = 0.0
    steps = 0
    state: dict[str, torch.Tensor] | None = None
    for batch in loader:
        images = batch["images"].to(device)
        cam_mask = batch["cam_mask"].to(device)
        outputs = model(images=images, cam_mask=cam_mask, prev_state=state)
        loss_items = criterion(outputs=outputs, batch=batch)
        optimizer.zero_grad(set_to_none=True)
        loss_items["loss"].backward()
        optimizer.step()
        state = {k: v.detach() for k, v in outputs["state"].items()}
        steps += 1
        loss_sum += float(loss_items["loss"].detach().cpu().item())
        det_norm_sum += float(loss_items["det_norm_loss"].cpu().item())
        det_supervision_sum += float(loss_items["det_supervision_loss"].cpu().item())
        assoc_entropy_sum += float(loss_items["assoc_entropy_loss"].cpu().item())
        assoc_pair_sum += float(loss_items["assoc_pair_loss"].cpu().item())
        assoc_consistency_sum += float(loss_items["assoc_consistency_loss"].cpu().item())
    if steps == 0:
        raise RuntimeError("阶段C没有训练步，请检查 max_steps 与数据配置")
    ckpt_dir = Path(config.checkpoint_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"stagec_epoch1_steps{steps}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "model_emb_dim": config.model_emb_dim,
                "model_track_slots": config.model_track_slots,
            },
            "steps": steps,
        },
        ckpt_path,
    )
    elapsed = time.time() - start
    return TrainSummary(
        epoch=1,
        steps=steps,
        mean_loss=loss_sum / steps,
        det_norm_loss=det_norm_sum / steps,
        det_supervision_loss=det_supervision_sum / steps,
        assoc_entropy_loss=assoc_entropy_sum / steps,
        assoc_pair_loss=assoc_pair_sum / steps,
        assoc_consistency_loss=assoc_consistency_sum / steps,
        elapsed_sec=elapsed,
        checkpoint_path=ckpt_path,
    )
