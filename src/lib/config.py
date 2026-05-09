from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
import logging
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    cors_origins: str = Field(
        default="*",
        description="Comma-separated allowed origins, or * for any.",
        validation_alias="CORS_ORIGINS",
    )
    app_name: str = Field(default="Facial Recognition TP1", validation_alias="APP_NAME")
    model_name: str | None = Field(default=None, validation_alias="MODEL_NAME")
    similarity_metric: str = Field(default="cosine", validation_alias="SIMILARITY_METRIC")
    similarity_threshold: float = Field(default=0.55, validation_alias="SIMILARITY_THRESHOLD")
    embeddings_path: Path = Field(default=Path("data/embeddings.json"), validation_alias="EMBEDDINGS_PATH")
    data_path: Path = Field(default=Path("data"), validation_alias="DATA_PATH")
    output_path: Path = Field(default=Path("output"), validation_alias="OUTPUT_PATH")
    model_path: Path = Field(default=Path("lib/models"), validation_alias="MODEL_PATH")
    max_workers: int = Field(default=2, validation_alias="MAX_WORKERS")
    face_size: int = Field(default=112, validation_alias="FACE_SIZE")
    embedding_dim: int = Field(default=512, validation_alias="EMBEDDING_DIM")
    use_pgvector: bool = Field(default=True, validation_alias="USE_PGVECTOR")
    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="vector_db", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="user", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(default="password", validation_alias="POSTGRES_PASSWORD")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
