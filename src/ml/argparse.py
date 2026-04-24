import argparse
from pathlib import Path
import json
import yaml


class ConfigAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        merged = {}
        for value in values:
            path = Path(value).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() in {".yaml", ".yml"}:
                loaded = yaml.safe_load(text) or {}
            elif path.suffix.lower() == ".json":
                loaded = json.loads(text)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
            if not isinstance(loaded, dict):
                raise ValueError(f"Config root must be dict: {path}")
            merged = _deep_merge(merged, loaded)
        setattr(namespace, self.dest, merged)


def _deep_merge(base: dict, incoming: dict) -> dict:
    result = dict(base)
    for key, value in incoming.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
