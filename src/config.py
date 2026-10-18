from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Runtime configuration.

    Values are read from the process environment first and from
    ``PROJECT_ROOT/.env`` second. That precedence matters: the previous
    implementation loaded ``.env`` with ``override=True``, so a stale local file
    silently won over an explicitly exported variable -- including in CI.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    default_model: str = "claude-sonnet-4-20250514"


settings = Settings()
