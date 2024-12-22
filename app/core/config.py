from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="my_prefix_")

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
