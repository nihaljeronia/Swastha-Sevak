from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    app_name: str = "Swastha Sevak"
    environment: str = "development"
    database_url: str = Field(
        "postgresql+asyncpg://sahayak:yourpassword@localhost:5432/swasthya_sahayak",
        env="DATABASE_URL",
    )
    whatsapp_api_url: str = Field("", env="WHATSAPP_API_URL")
    whatsapp_token: str = Field("", env="WHATSAPP_TOKEN")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
