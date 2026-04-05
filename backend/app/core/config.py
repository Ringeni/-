from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = Field(default="multi-camera-tracking-backend", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_port: int = Field(default=8000, alias="APP_PORT")

    mysql_host: str = Field(alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str = Field(alias="MYSQL_USER")
    mysql_password: str = Field(alias="MYSQL_PASSWORD")
    mysql_db: str = Field(alias="MYSQL_DB")

    redis_host: str = Field(alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")

    jwt_secret_key: str = Field(alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_expire_hours: int = Field(default=2, alias="JWT_ACCESS_EXPIRE_HOURS")
    jwt_refresh_expire_days: int = Field(default=7, alias="JWT_REFRESH_EXPIRE_DAYS")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def async_database_url(self) -> str:
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )

    @property
    def redis_url(self) -> str:
        auth_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth_part}{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
