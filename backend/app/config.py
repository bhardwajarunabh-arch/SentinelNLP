from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "SentinelNLP"
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    artifact_dir: Path = Path("artifacts")
    allowed_origins: str = "http://localhost:5173"
    random_seed: int = 42
    baseline_model_path: Path = Path("models/baseline")
    transformer_model_path: Path = Path("models/xlmr")
    max_input_length: int = 10000
    nvd_page_size: int = 2000
    nvd_request_delay_seconds: float = 0.7

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
