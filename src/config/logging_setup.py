import logging
import sys

import structlog

from src.config.settings import LogConfig


def setup_logging(config: LogConfig) -> None:
    """
    Configure structlog once at application startup.

    Two renderers:
      - "text"  -> human-readable coloured output for local dev
      - "json"  -> machine-readable JSON for production log aggregators
                  (ELK, Datadog, CloudWatch, Splunk, etc.)

    Also configures stdlib logging at the same level so third-party
    libraries (langchain, drain3, etc.) emit through the same pipeline.
    """
    log_level = getattr(logging, config.level.upper(), logging.INFO)

    # Stdlib logging — captures output from third-party libraries.
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stdout,
    )

    # Silence third-party loggers — their HTTP traces, model downloads, and
    # tqdm progress bars are implementation details, not application events.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Processors run on every log call in order.
    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,           # adds "level" field
        structlog.stdlib.add_logger_name,          # adds "logger" field (module name)
        structlog.processors.TimeStamper(fmt="iso"),  # adds "timestamp" in ISO 8601
        structlog.processors.StackInfoRenderer(),  # renders stack_info if present
        structlog.processors.ExceptionRenderer(),  # renders exc_info into the event
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if config.format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
