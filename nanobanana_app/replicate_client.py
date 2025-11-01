from __future__ import annotations

from typing import Optional

import replicate

from .config import get_settings


def get_client(api_token: Optional[str] = None) -> replicate.Client:
    """Create and return a Replicate client using the configured API token.

    Args:
        api_token: Optional explicit token. If not provided, loads from settings.
    """
    token = api_token or get_settings().replicate_api_token
    return replicate.Client(api_token=token)


def get_model(client: Optional[replicate.Client] = None, *, name: str = "google/nano-banana") -> replicate.Model:
    """Get the specified Replicate model (defaults to google/nano-banana)."""
    client = client or get_client()
    return client.models.get(name)
