from pathlib import Path

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    model: str
    base_url: str
    max_retries: int


class EmbedderConfig(BaseModel):
    model: str


class CacheConfig(BaseModel):
    high_threshold: float
    low_threshold: float
    recency_window_seconds: int
    staleness_days: int


class SpikeConfig(BaseModel):
    bucket_seconds: int
    z_threshold: float
    min_history: int
    min_spike_count: int


class BurstConfig(BaseModel):
    bucket_seconds: int
    threshold_per_bucket: int
    consecutive_buckets: int


class AnomalyConfig(BaseModel):
    novelty_threshold: float
    min_store_size: int
    spike: SpikeConfig
    burst: BurstConfig


class LogConfig(BaseModel):
    level: str    # DEBUG | INFO | WARNING | ERROR
    format: str   # "text" | "json"


class AppConfig(BaseModel):
    llm: LLMConfig
    embedder: EmbedderConfig
    cache: CacheConfig
    anomaly: AnomalyConfig
    logging: LogConfig


_DEFAULT_CONFIG_PATH = Path("config/settings.yaml")


def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> AppConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return AppConfig.model_validate(data or {})
