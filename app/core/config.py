from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Swastha Sevak"
    environment: str = "development"
    database_url: str = Field(
        "postgresql+asyncpg://postgres:yourpassword@127.0.0.1:5432/swasthya_sahayak",
        env="DATABASE_URL",
    )
    whatsapp_api_url: str = Field("", env="WHATSAPP_API_URL")
    whatsapp_token: str = Field("", env="WHATSAPP_TOKEN")
    meta_phone_number_id: str = Field("", env="META_PHONE_NUMBER_ID")
    meta_access_token: str = Field("", env="META_ACCESS_TOKEN")
    meta_verify_token: str = Field("", env="META_VERIFY_TOKEN")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Prevent pydantic-settings crashing from extra .env vars
    }


settings = Settings()
