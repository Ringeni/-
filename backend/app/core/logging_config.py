import logging

from app.core.config import get_settings
from app.core.request_context import get_request_id


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [requestId=%(request_id)s] %(name)s - %(message)s",
    )
    request_id_filter = RequestIdFilter()
    logging.getLogger().addFilter(request_id_filter)
