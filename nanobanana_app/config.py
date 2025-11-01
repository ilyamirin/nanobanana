import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


# Load environment variables from a local .env file if present
load_dotenv()


@dataclass(frozen=True)
class Settings:
    replicate_api_token: str

    @staticmethod
    def load() -> "Settings":
        token = os.getenv("REPLICATE_API_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "REPLICATE_API_TOKEN is not set. Create a .env file (see .env.example) "
                "or export the variable in your shell."
            )
        return Settings(replicate_api_token=token)


def get_settings() -> Settings:
    return Settings.load()
