from dataclasses import dataclass
from pathlib import Path
import os
import yaml


@dataclass(frozen=True)
class AppConfig:
    data_root: Path
    split: str
    subset: str
    max_samples: int
    batch_size: int
    num_workers: int
    min_views: int
    temporal_stride: int
    max_steps: int
    split_part: str
    split_ratio: float
    healthcheck_max_records: int
    model_emb_dim: int
    model_track_slots: int
    model_det_queries: int
    lr: float
    weight_decay: float
    checkpoint_dir: str
    eval_split_part: str
    eval_max_steps: int
    eval_checkpoint_path: str
    infer_max_frames: int
    infer_prob_threshold: float
    sqlite_path: str


def _resolve_data_root(value: str | None) -> Path:
    if value and value.strip():
        return Path(value).expanduser().resolve()
    env_value = os.getenv("MCT_DATA_ROOT", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path("data").resolve()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    config_dict: dict[str, object] = {}
    if config_path is not None:
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            config_dict = loaded
    data_root = _resolve_data_root(config_dict.get("data_root") if isinstance(config_dict.get("data_root"), str) else None)
    split = str(config_dict.get("split", "train"))
    subset = str(config_dict.get("subset", "mmp_cafe"))
    max_samples = int(config_dict.get("max_samples", 10))
    batch_size = int(config_dict.get("batch_size", 2))
    num_workers = int(config_dict.get("num_workers", 0))
    min_views = int(config_dict.get("min_views", 2))
    temporal_stride = int(config_dict.get("temporal_stride", 1))
    max_steps = int(config_dict.get("max_steps", 8))
    split_part = str(config_dict.get("split_part", "all"))
    split_ratio = float(config_dict.get("split_ratio", 0.9))
    healthcheck_max_records = int(config_dict.get("healthcheck_max_records", 5000))
    model_emb_dim = int(config_dict.get("model_emb_dim", 128))
    model_track_slots = int(config_dict.get("model_track_slots", 64))
    model_det_queries = int(config_dict.get("model_det_queries", 16))
    lr = float(config_dict.get("lr", 1e-4))
    weight_decay = float(config_dict.get("weight_decay", 1e-4))
    checkpoint_dir = str(config_dict.get("checkpoint_dir", "checkpoints"))
    eval_split_part = str(config_dict.get("eval_split_part", "val"))
    eval_max_steps = int(config_dict.get("eval_max_steps", 8))
    eval_checkpoint_path = str(config_dict.get("eval_checkpoint_path", ""))
    infer_max_frames = int(config_dict.get("infer_max_frames", 4))
    infer_prob_threshold = float(config_dict.get("infer_prob_threshold", 0.5))
    sqlite_path = str(config_dict.get("sqlite_path", "runtime/track_history.db"))
    return AppConfig(
        data_root=data_root,
        split=split,
        subset=subset,
        max_samples=max_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        min_views=min_views,
        temporal_stride=temporal_stride,
        max_steps=max_steps,
        split_part=split_part,
        split_ratio=split_ratio,
        healthcheck_max_records=healthcheck_max_records,
        model_emb_dim=model_emb_dim,
        model_track_slots=model_track_slots,
        model_det_queries=model_det_queries,
        lr=lr,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir,
        eval_split_part=eval_split_part,
        eval_max_steps=eval_max_steps,
        eval_checkpoint_path=eval_checkpoint_path,
        infer_max_frames=infer_max_frames,
        infer_prob_threshold=infer_prob_threshold,
        sqlite_path=sqlite_path,
    )
