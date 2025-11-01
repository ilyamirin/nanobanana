#!/usr/bin/env python3
"""
Quick connectivity check for Replicate and the google/nano-banana model.

Usage:
  1) Create a `.env` file in the project root with:
       REPLICATE_API_TOKEN=your-token-here
  2) Install deps: `pip install -r requirements.txt`
  3) Run: `python scripts/check_replicate_connection.py`

This script will:
  - Initialize the Replicate client
  - Fetch the `google/nano-banana` model
  - Print the latest version id and model info summary

No generation is performed here to avoid costs; it's only a metadata fetch.
"""
import sys

from nanobanana_app.replicate_client import get_client, get_model


def main() -> int:
    try:
        client = get_client()
        model = get_model(client)
        # Some models (like google/nano-banana) don't expose a versions list endpoint.
        # Avoid calling model.versions.list(); instead, try to read latest_version if present.
        latest_version = getattr(model, "latest_version", None)
        latest_id = None
        try:
            # latest_version may be an object with id or a plain string; handle both.
            if latest_version is not None:
                latest_id = getattr(latest_version, "id", None) or (
                    latest_version if isinstance(latest_version, str) else None
                )
        except Exception:
            latest_id = None

        print("Connection OK ✅")
        print(f"Model: {model.owner}/{model.name}")
        if latest_id:
            print(f"Latest version: {latest_id}")
        else:
            print("Latest version: (not exposed by this model)")
        return 0
    except Exception as e:
        print("Connection FAILED ❌")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
