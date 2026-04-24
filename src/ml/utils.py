class Config:
    def __init__(self, source):
        if isinstance(source, Config):
            source = source.to_dict()
        if hasattr(source, "__dict__"):
            data = dict(vars(source))
        elif isinstance(source, dict):
            data = dict(source)
        else:
            raise TypeError("Config source must be argparse namespace or dict")
        merged = {}
        cfg_payload = data.pop("CFG", None)
        object.__setattr__(self, "CFG", cfg_payload)
        if isinstance(cfg_payload, dict):
            merged = _deep_merge(merged, cfg_payload)
        merged = _deep_merge(merged, data)
        for key, value in merged.items():
            object.__setattr__(self, key, _to_config(value))

    def __setattr__(self, key, value):
        object.__setattr__(self, key, _to_config(value))

    def __delattr__(self, key):
        object.__delattr__(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __iter__(self):
        return iter(self.__dict__)

    def to_dict(self):
        output = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                output[key] = value.to_dict()
            else:
                output[key] = value
        return output

    def __repr__(self):
        return f"Config({self.to_dict()})"


def _to_config(value):
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_to_config(v) for v in value]
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if any(ch in lowered for ch in [".", "e"]):
                return float(lowered)
            return int(lowered)
        except ValueError:
            return value
    return value


def _deep_merge(base: dict, incoming: dict) -> dict:
    result = dict(base)
    for key, value in incoming.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
