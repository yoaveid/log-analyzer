from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEntry(BaseModel):
    timestamp: datetime
    level: LogLevel
    service: str
    message: str
    raw_line: str


class EnrichedLogEntry(LogEntry):
    root_cause: Optional[str] = None
    mitigation: Optional[str] = None
    cache_hit: bool = False
