from collections import Counter
from pathlib import Path
import argparse

from torch.utils.data import DataLoader
import torch

from src.data import MmpMultiViewDataset, TemporalStepSampler, build_data_health_report, collate_multiview_batch
from src.engine import evaluate_checkpoint, train_one_epoch
from src.service import create_app
from src.storage import MySQLStore, RedisStore, RealtimeTrack, TrackHistory
from src.models import MctrSystem
from src.system import MmpCafeDataset, load_config
from src.inference import RuntimeTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="index",
        choices=["index", "stage-a", "healthcheck", "stage-b", "stage-c", "stage-c-eval", "stage-d", "stage-e"],
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "stage-a":
        run_stage_a(config)
        return
    if args.mode == "healthcheck":
        run_healthcheck(config)
        return
    if args.mode == "stage-b":
        run_stage_b(config)
        return
    if args.mode == "stage-c":
        run_stage_c(config)
        return
    if args.mode == "stage-c-eval":
        run_stage_c_eval(config)
        return
    if args.mode == "stage-d":
        run_stage_d(config)
        return
    if args.mode == "stage-e":
        run_stage_e(config)
        return
    run_index(config)


def run_index(config) -> None:
    dataset = MmpCafeDataset(data_root=Path(config.data_root), subset=config.subset, split=config.split)
    total = len(dataset)
    if total == 0:
        raise RuntimeError("未读取到任何样本，请检查 data_root/subset/split 是否正确")
    group_counter = Counter(record.camera_group for record in dataset.records())
    scene_counter = Counter(record.scene for record in dataset.records())
    print(f"data_root={config.data_root}")
    print(f"subset={config.subset} split={config.split}")
    print(f"total_samples={total}")
    print(f"camera_groups={dict(group_counter)}")
    print(f"scenes={dict(scene_counter)}")
    show_count = min(config.max_samples, total)
    print(f"preview_samples={show_count}")
    for idx in range(show_count):
        sample = dataset.get(idx)
        label_keys = list(sample.label.keys()) if isinstance(sample.label, dict) else []
        print(
            f"[{idx}] group={sample.record.camera_group} scene={sample.record.scene} "
            f"frame={sample.record.frame_id} cam={sample.record.camera_id} "
            f"img={sample.record.image_path.name} label_keys={label_keys}"
        )


def run_stage_a(config) -> None:
    dataset = MmpMultiViewDataset(
        data_root=Path(config.data_root),
        subset=config.subset,
        split=config.split,
        min_views=config.min_views,
        split_part=config.split_part,
        split_ratio=config.split_ratio,
    )
    if len(dataset) == 0:
        raise RuntimeError("阶段A数据集为空，请检查配置与数据目录")
    sampler = TemporalStepSampler(
        data_source=dataset,
        stride=config.temporal_stride,
        max_steps=config.max_steps,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_multiview_batch,
    )
    first_batch = next(iter(loader))
    print(f"stage_a_samples={len(dataset)}")
    print(f"camera_ids={first_batch['camera_ids'].tolist()}")
    print(f"images_shape={tuple(first_batch['images'].shape)}")
    print(f"cam_mask_shape={tuple(first_batch['cam_mask'].shape)} valid={int(first_batch['cam_mask'].sum())}")
    print(f"boxes_shape={tuple(first_batch['boxes'].shape)}")
    print(f"ids_shape={tuple(first_batch['ids'].shape)}")
    print(f"target_mask_shape={tuple(first_batch['target_mask'].shape)} valid={int(first_batch['target_mask'].sum())}")
    print(f"frame_ids={first_batch['frame_ids'].tolist()}")
    print(f"scenes={first_batch['scenes']}")
    print(f"camera_groups={first_batch['camera_groups']}")
    print(f"split_part={config.split_part} split_ratio={config.split_ratio}")


def run_healthcheck(config) -> None:
    report = build_data_health_report(
        data_root=Path(config.data_root),
        subset=config.subset,
        split=config.split,
        max_records=config.healthcheck_max_records,
    )
    print(f"total_pairs={report['total_pairs']}")
    print(f"checked_pairs={report['checked_pairs']}")
    print(f"missing_label_count={report['missing_label_count']}")
    print(f"invalid_box_count={report['invalid_box_count']}")
    print(f"empty_label_count={report['empty_label_count']}")
    print(f"empty_label_ratio={report['empty_label_ratio']:.6f}")
    print(f"image_size_hist={report['image_size_hist']}")


def run_stage_b(config) -> None:
    dataset = MmpMultiViewDataset(
        data_root=Path(config.data_root),
        subset=config.subset,
        split=config.split,
        min_views=config.min_views,
        split_part=config.split_part,
        split_ratio=config.split_ratio,
    )
    if len(dataset) == 0:
        raise RuntimeError("阶段B数据集为空，请检查配置与数据目录")
    sampler = TemporalStepSampler(
        data_source=dataset,
        stride=config.temporal_stride,
        max_steps=config.max_steps,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_multiview_batch,
    )
    batch = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MctrSystem(
        in_channels=3,
        emb_dim=config.model_emb_dim,
        track_slots=config.model_track_slots,
        det_queries=config.model_det_queries,
    ).to(device)
    images = batch["images"].to(device)
    cam_mask = batch["cam_mask"].to(device)
    with torch.no_grad():
        outputs = model(images=images, cam_mask=cam_mask)
    print(f"stage_b_samples={len(dataset)}")
    print(f"device={device.type}")
    print(f"model_emb_dim={config.model_emb_dim} model_track_slots={config.model_track_slots}")
    print(f"input_images_shape={tuple(images.shape)}")
    print(f"det_tokens_shape={tuple(outputs['det_tokens'].shape)}")
    print(f"det_scores_shape={tuple(outputs['det_scores'].shape)}")
    print(f"tracks_shape={tuple(outputs['tracks'].shape)}")
    print(f"assoc_shape={tuple(outputs['assoc'].shape)}")
    assoc_sum = outputs["assoc"].sum(dim=-1).mean().item()
    print(f"assoc_row_softmax_mean_sum={assoc_sum:.6f}")


def run_stage_c(config) -> None:
    summary = train_one_epoch(config)
    print(f"stage_c_epoch={summary.epoch}")
    print(f"stage_c_steps={summary.steps}")
    print(f"stage_c_mean_loss={summary.mean_loss:.6f}")
    print(f"stage_c_det_norm_loss={summary.det_norm_loss:.6f}")
    print(f"stage_c_det_supervision_loss={summary.det_supervision_loss:.6f}")
    print(f"stage_c_assoc_entropy_loss={summary.assoc_entropy_loss:.6f}")
    print(f"stage_c_assoc_pair_loss={summary.assoc_pair_loss:.6f}")
    print(f"stage_c_assoc_consistency_loss={summary.assoc_consistency_loss:.6f}")
    print(f"stage_c_elapsed_sec={summary.elapsed_sec:.3f}")
    print(f"stage_c_checkpoint={summary.checkpoint_path}")


def run_stage_c_eval(config) -> None:
    summary = evaluate_checkpoint(config)
    print(f"stage_c_eval_steps={summary.steps}")
    print(f"stage_c_eval_mean_loss={summary.mean_loss:.6f}")
    print(f"stage_c_eval_assoc_top1_acc={summary.assoc_top1_acc:.6f}")
    print(f"stage_c_eval_assoc_conf_mean={summary.assoc_conf_mean:.6f}")
    print(f"stage_c_eval_checkpoint={summary.checkpoint_path}")


def run_stage_d(config) -> None:
    sequence, camera_group, scene = _collect_sequence(config)
    device, model = _load_stagec_model(config)
    tracker = RuntimeTracker(prob_threshold=config.infer_prob_threshold)
    state = None
    for step_idx, item in enumerate(sequence):
        batch = collate_multiview_batch([item])
        images = batch["images"].to(device)
        cam_mask = batch["cam_mask"].to(device)
        camera_ids = batch["camera_ids"].tolist()
        frame_id = int(item["frame_id"])
        with torch.no_grad():
            outputs = model(images=images, cam_mask=cam_mask, prev_state=state)
        state = {k: v.detach() for k, v in outputs["state"].items()}
        decoded = tracker.step(assoc=outputs["assoc"], cam_mask=cam_mask, det_scores=outputs["det_scores"])
        target_id = decoded.target_id.squeeze(0).tolist()
        pred_slot = decoded.pred_slot.squeeze(0).tolist()
        max_prob = decoded.max_prob.squeeze(0).tolist()
        print(f"[stage-d] seq={step_idx} frame_id={frame_id} scene={scene} camera_group={camera_group}")
        for cam_id, t_id, slot, prob in zip(camera_ids, target_id, pred_slot, max_prob):
            print(f"  cam={cam_id} pred_slot={slot} target_id={t_id} max_prob={prob:.6f}")


def run_stage_e(config) -> None:
    sequence, camera_group, scene = _collect_sequence(config)
    device, model = _load_stagec_model(config)
    tracker = RuntimeTracker(prob_threshold=config.infer_prob_threshold)
    redis_store = RedisStore()
    mysql_store = MySQLStore(db_path=Path(config.sqlite_path).expanduser().resolve())
    state = None
    for item in sequence:
        batch = collate_multiview_batch([item])
        images = batch["images"].to(device)
        cam_mask = batch["cam_mask"].to(device)
        camera_ids = batch["camera_ids"].tolist()
        frame_id = int(item["frame_id"])
        with torch.no_grad():
            outputs = model(images=images, cam_mask=cam_mask, prev_state=state)
        state = {k: v.detach() for k, v in outputs["state"].items()}
        decoded = tracker.step(assoc=outputs["assoc"], cam_mask=cam_mask, det_scores=outputs["det_scores"])
        realtime_rows: list[RealtimeTrack] = []
        history_rows: list[TrackHistory] = []
        target_id = decoded.target_id.squeeze(0).tolist()
        pred_slot = decoded.pred_slot.squeeze(0).tolist()
        max_prob = decoded.max_prob.squeeze(0).tolist()
        for cam_id, t_id, slot, prob in zip(camera_ids, target_id, pred_slot, max_prob):
            realtime_rows.append(
                RealtimeTrack(
                    camera_group=camera_group,
                    scene=scene,
                    frame_id=frame_id,
                    camera_id=int(cam_id),
                    target_id=int(t_id),
                    pred_slot=int(slot),
                    max_prob=float(prob),
                )
            )
            history_rows.append(
                TrackHistory(
                    camera_group=camera_group,
                    scene=scene,
                    frame_id=frame_id,
                    camera_id=int(cam_id),
                    target_id=int(t_id),
                    pred_slot=int(slot),
                    max_prob=float(prob),
                )
            )
        redis_store.upsert_realtime_tracks(realtime_rows)
        mysql_store.insert_track_history(history_rows)
    app = create_app(redis_store=redis_store, mysql_store=mysql_store)
    realtime_total = len(redis_store.list_realtime_tracks(camera_group=camera_group))
    history_preview = mysql_store.query_track_history(target_id=1, limit=5)
    print(f"stage_e_scene={scene} camera_group={camera_group} frames={len(sequence)}")
    print(f"stage_e_realtime_tracks={realtime_total}")
    print(f"stage_e_history_preview={len(history_preview)}")
    print(f"stage_e_api_routes={len(app.routes)}")
    print("stage_e_status=service_storage_ready")
    print("stage_e_start_api=uvicorn src.service.app:app --host 0.0.0.0 --port 8000")


def _resolve_stagec_checkpoint(config) -> Path:
    if config.eval_checkpoint_path.strip():
        candidate = Path(config.eval_checkpoint_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"推理 checkpoint 不存在: {candidate}")
        return candidate
    ckpt_dir = Path(config.checkpoint_dir).expanduser().resolve()
    candidates = sorted(ckpt_dir.glob("stagec_epoch*.pt"))
    if not candidates:
        raise FileNotFoundError(f"未找到 stagec checkpoint: {ckpt_dir}")
    return candidates[-1]


def _collect_sequence(config) -> tuple[list[dict], str, str]:
    dataset = MmpMultiViewDataset(
        data_root=Path(config.data_root),
        subset=config.subset,
        split=config.split,
        min_views=config.min_views,
        split_part=config.split_part,
        split_ratio=config.split_ratio,
    )
    if len(dataset) == 0:
        raise RuntimeError("推理数据集为空，请检查配置与数据目录")
    first = dataset[0]
    camera_group = first["camera_group"]
    scene = first["scene"]
    sequence: list[dict] = []
    for i in range(len(dataset)):
        item = dataset[i]
        if item["camera_group"] != camera_group or item["scene"] != scene:
            break
        sequence.append(item)
        if len(sequence) >= config.infer_max_frames:
            break
    if not sequence:
        raise RuntimeError("未找到可推理的序列")
    return sequence, camera_group, scene


def _load_stagec_model(config) -> tuple[torch.device, MctrSystem]:
    checkpoint_path = _resolve_stagec_checkpoint(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MctrSystem(
        in_channels=3,
        emb_dim=config.model_emb_dim,
        track_slots=config.model_track_slots,
        det_queries=config.model_det_queries,
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return device, model


if __name__ == "__main__":
    main()
