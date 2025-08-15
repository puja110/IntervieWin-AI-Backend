from pydantic_settings import BaseSettings, SettingsConfigDict
import sys

class Settings(BaseSettings):
    GROQ_API_KEY: str
    ENV: str = "development"  # "development" or "production"
    LOG_LEVEL: str = "info"

    model_config = SettingsConfigDict(
        env_file=".env",            # load from .env if available
        env_file_encoding="utf-8",
        case_sensitive=True
    )

settings = Settings()

# Safety check
if not settings.GROQ_API_KEY:
    sys.exit("ERROR: GROQ_API_KEY is missing! Set it in .env or as an environment variable.")

if settings.ENV.lower() == "production" and settings.GROQ_API_KEY.lower().startswith("sample"):
    sys.exit("ERROR: GROQ_API_KEY is invalid. Set your real key in production.")
