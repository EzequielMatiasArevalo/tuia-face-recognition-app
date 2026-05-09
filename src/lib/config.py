from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), env_file_encoding="utf-8", extra="ignore")

    cors_origins: str = Field(default="*")
    app_name: str = "Facial Recognition TP1"
    model_name: str | None = None
    similarity_metric: str = "cosine"
    similarity_threshold: float = 0.55
    embeddings_path: Path = Path("data/embeddings.json")
    data_path: Path = Path("data")
    output_path: Path = Path("output")
    model_path: Path = Path("lib/models")
    max_workers: int = 2
    face_size: int = 112
    embedding_dim: int = 512
    use_pgvector: bool = True
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "vector_db"
    postgres_user: str = "user"
    postgres_password: str = "password"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
