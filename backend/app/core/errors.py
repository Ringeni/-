from dataclasses import dataclass


class ErrorCode:
    SUCCESS = 0
    PARAM_INVALID = 4001
    FORBIDDEN = 4003
    NOT_FOUND = 4004
    STATUS_CONFLICT = 4009
    INFERENCE_UNAVAILABLE = 5001
    CACHE_UNAVAILABLE = 5002
    DB_UNAVAILABLE = 5003
    PLAYBACK_UNAVAILABLE = 5004
    INTERNAL_ERROR = 5999


@dataclass
class AppException(Exception):
    code: int
    message: str
    http_status: int = 400
