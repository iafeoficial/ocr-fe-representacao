from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    ocr_shared_secret: str = ""
    # Comma-separated browser origins, or "*" (default) for internal OCR + X-OCR-Secret
    cors_origins: str = "*"
    host: str = "0.0.0.0"
    port: int = 8000

    def parsed_cors_origins(self) -> list[str]:
        raw = (self.cors_origins or "*").strip()
        if raw == "*":
            return ["*"]
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        return origins or ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
