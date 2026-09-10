from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    ocr_shared_secret: str = ""
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
