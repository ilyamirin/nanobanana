#!/usr/bin/env python3
"""
Streamlit UI for google/nano-banana via Replicate.

Goals:
- Cover ALL available model settings by dynamically reading the model's input schema
  from Replicate's API (when available), and rendering appropriate widgets.
- Fallback to a JSON editor for arbitrary inputs in case the schema isn't exposed.
- Generate images and save outputs to `outputs/` while also previewing them inline.

Run:
  streamlit run streamlit_app.py

Requirements:
  - REPLICATE_API_TOKEN must be set (via .env or env var), as configured in nanobanana_app/config.py
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from nanobanana_app.config import get_settings
from nanobanana_app.replicate_client import get_client

REPLICATE_API_BASE = "https://api.replicate.com/v1"
MODEL_NAME = "google/nano-banana"
OUTPUTS_DIR = Path("outputs")


# ----------------------------
# Helpers: Replicate Schema
# ----------------------------

def _http_get(url: str, token: str) -> Dict[str, Any]:
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Token {token}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:  # nosec B310 - trusted URL format to Replicate API
        data = resp.read().decode("utf-8")
        return json.loads(data)


def fetch_latest_version_id(token: str) -> Optional[str]:
    try:
        data = _http_get(f"{REPLICATE_API_BASE}/models/{MODEL_NAME}", token)
        latest = data.get("latest_version")
        if isinstance(latest, dict):
            return latest.get("id")
        if isinstance(latest, str):
            return latest
    except Exception:
        return None
    return None


def fetch_openapi_schema_for_version(version_id: str, token: str) -> Optional[Dict[str, Any]]:
    try:
        data = _http_get(
            f"{REPLICATE_API_BASE}/models/{MODEL_NAME}/versions/{version_id}", token
        )
        return data.get("openapi_schema")
    except Exception:
        return None


# ----------------------------
# Helpers: Dynamic UI from Schema
# ----------------------------

def _sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    return name or "result"


def _extract_inputs_schema(openapi_schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Replicate OpenAPI schema typically nests input schema at components/schemas/Input.
    Return (properties, required_keys).
    """
    if not openapi_schema:
        return {}, []

    # Common locations in Replicate schemas
    # 1) components.schemas.Input
    comp = openapi_schema.get("components", {}).get("schemas", {})
    input_schema = comp.get("Input") or comp.get("input")
    if isinstance(input_schema, dict):
        props = input_schema.get("properties", {}) or {}
        required = input_schema.get("required", []) or []
        return props, required

    # 2) Or top-level `input` in the schema.
    top_props = openapi_schema.get("input", {}).get("properties", {})
    top_req = openapi_schema.get("input", {}).get("required", [])
    if top_props:
        return top_props, top_req or []

    return {}, []


def _widget_for_property(name: str, schema: Dict[str, Any], required: bool) -> Any:
    """Render a Streamlit widget based on a JSON Schema property.
    Return the chosen value.
    """
    title = schema.get("title") or name
    desc = schema.get("description", "")
    default = schema.get("default")

    type_ = schema.get("type")
    enum = schema.get("enum")

    # Number/Integer with bounds
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    multiple_of = schema.get("multipleOf")

    # Heuristics: treat big text fields as textarea
    is_prompt_like = name.lower() in {"prompt", "text", "caption"}

    help_text = desc + (" (required)" if required else "")

    if enum:
        # Enum select
        return st.selectbox(title, options=enum, index=(enum.index(default) if default in enum else 0), help=help_text, key=f"inp_{name}")

    if type_ in ("string", None):
        if is_prompt_like or (schema.get("format") == "textarea"):
            return st.text_area(title, value=default or "", help=help_text, key=f"inp_{name}")
        return st.text_input(title, value=default or "", help=help_text, key=f"inp_{name}")

    if type_ == "boolean":
        return st.checkbox(title, value=bool(default) if default is not None else False, help=help_text, key=f"inp_{name}")

    if type_ in ("number", "integer"):
        step = multiple_of or (1 if type_ == "integer" else 0.1)
        if minimum is not None and maximum is not None:
            # Use slider if bounds provided
            if type_ == "integer":
                return st.slider(title, int(minimum), int(maximum), int(default or minimum or 0), step=int(step), help=help_text, key=f"inp_{name}")
            return st.slider(title, float(minimum), float(maximum), float(default or minimum or 0.0), step=float(step), help=help_text, key=f"inp_{name}")
        # Fall back to number_input
        if type_ == "integer":
            return st.number_input(title, value=int(default or 0), step=int(step), help=help_text, key=f"inp_{name}")
        return st.number_input(title, value=float(default or 0.0), step=float(step), help=help_text, key=f"inp_{name}")

    if type_ == "array":
        items = schema.get("items", {})
        if items.get("type") == "string":
            val = st.text_area(title, value="\n".join(default or []) if isinstance(default, list) else "", help=(help_text + "\n(enter one item per line)"), key=f"inp_{name}")
            # Split by lines, ignore empties
            return [line.strip() for line in val.splitlines() if line.strip()]
        # Fallback JSON
        val = st.text_area(title + " (JSON)", value=json.dumps(default or [], ensure_ascii=False, indent=2), help=help_text, key=f"inp_{name}")
        try:
            return json.loads(val)
        except Exception:
            st.warning(f"Поле {name}: некорректный JSON. Будет проигнорировано.")
            return default

    if type_ == "object":
        val = st.text_area(title + " (JSON)", value=json.dumps(default or {}, ensure_ascii=False, indent=2), help=help_text, key=f"inp_{name}")
        try:
            return json.loads(val)
        except Exception:
            st.warning(f"Поле {name}: некорректный JSON. Будет проигнорировано.")
            return default or {}

    # Unknown type: raw JSON editor
    val = st.text_area(title + " (JSON)", value=json.dumps(default, ensure_ascii=False, indent=2) if default is not None else "", help=help_text, key=f"inp_{name}")
    try:
        return json.loads(val)
    except Exception:
        return val


# ----------------------------
# Helpers: Prediction and saving
# ----------------------------

def ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as r:  # nosec B310 - URL is provided by Replicate
        with open(dest, "wb") as f:
            f.write(r.read())


def save_images(urls: List[str], base_name: str) -> List[Path]:
    ensure_outputs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _sanitize_filename(base_name)[:50]
    paths: List[Path] = []
    for i, url in enumerate(urls):
        ext = os.path.splitext(urllib.request.urlparse(url).path)[1] or ".png"
        filename = f"{ts}_{base}_{i+1}{ext}"
        path = OUTPUTS_DIR / filename
        try:
            download_file(url, path)
            paths.append(path)
        except Exception as e:
            st.error(f"Ошибка сохранения файла: {e}")
    return paths


def extract_image_urls(result: Any) -> List[str]:
    """Extract image URLs from various possible Replicate outputs.

    Handles:
    - direct string URL
    - list of URLs or list of dicts with url/image/src
    - dicts with keys: images/image/url/output/result (possibly nested)
    - objects with attribute `.output`
    - arbitrary nesting: recursively scans for http(s) URLs in strings
    """
    url_regex = re.compile(r'https?://[^\s\]\}\'">]+', re.IGNORECASE)

    def is_image_like(url: str) -> bool:
        # Accept common image extensions or replicate.delivery links
        lower = url.lower()
        if any(lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
            return True
        if "replicate.delivery" in lower:
            return True
        return lower.startswith("http://") or lower.startswith("https://")

    def collect(obj: Any, acc: List[str]) -> None:
        if obj is None:
            return
        # If Prediction-like object with .output, unwrap it
        if hasattr(obj, "output") and not isinstance(obj, (str, bytes, dict, list, tuple)):
            try:
                collect(getattr(obj, "output"), acc)
            except Exception:
                pass
        if isinstance(obj, str):
            # The whole string could be a URL or contain one
            candidates = [obj] if obj.startswith("http") else url_regex.findall(obj)
            for u in candidates:
                if is_image_like(u):
                    acc.append(u)
            return
        if isinstance(obj, (list, tuple)):
            for it in obj:
                collect(it, acc)
            return
        if isinstance(obj, dict):
            # Prefer known keys first
            for k in ("images", "output", "result", "image", "url", "src"):
                if k in obj:
                    collect(obj[k], acc)
            # Then scan all remaining values
            for v in obj.values():
                collect(v, acc)
            return
        # Fallback: try string representation for URLs
        try:
            s = str(obj)
            for u in url_regex.findall(s):
                if is_image_like(u):
                    acc.append(u)
        except Exception:
            pass

    urls: List[str] = []
    collect(result, urls)

    # De-duplicate while preserving order
    seen = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    return unique_urls


# ----------------------------
# Streamlit App
# ----------------------------

def main() -> None:
    st.set_page_config(page_title="Nano Banana — Image Generator", page_icon="🍌", layout="wide")
    st.title("🍌 Google Nano Banana — генерация изображений")

    # Load settings and client
    try:
        settings = get_settings()
    except Exception as e:
        st.error("Не найден токен REPLICATE_API_TOKEN. Создайте .env или экспортируйте переменную окружения.")
        st.stop()

    client = get_client(settings.replicate_api_token)

    # Sidebar: schema load and controls
    with st.sidebar:
        st.header("Настройки модели")
        st.caption("Загрузка схемы входных параметров модели для максимального покрытия настроек.")
        with st.spinner("Загрузка схемы модели..."):
            version_id = fetch_latest_version_id(settings.replicate_api_token)
            openapi_schema = fetch_openapi_schema_for_version(version_id, settings.replicate_api_token) if version_id else None
        if version_id:
            st.success(f"Версия модели: {version_id[:8]}…")
        else:
            st.warning("Версия модели не раскрывается API. Доступен только ручной ввод параметров.")

        show_raw_json = st.toggle("Показывать редактор JSON", value=False, help="Позволяет задать любые параметры руками, если чего-то нет в форме.")

    # Center: dynamic form from schema
    input_values: Dict[str, Any] = {}
    props, required = _extract_inputs_schema(openapi_schema) if openapi_schema else ({}, [])

    has_schema = bool(props)
    if has_schema:
        st.subheader("Параметры ввода")
        # Render prompt-like fields first for UX
        ordered_items = sorted(props.items(), key=lambda kv: (0 if kv[0].lower() in {"prompt", "text", "caption"} else 1, kv[0]))
        for name, prop in ordered_items:
            try:
                val = _widget_for_property(name, prop, required=(name in required))
                if val is not None and (val != "" or name in required):
                    input_values[name] = val
            except Exception as e:
                st.warning(f"Не удалось отрендерить поле {name}: {e}")
    else:
        st.info("Схема входных параметров недоступна. Используйте JSON-параметры ниже.")

    # Raw JSON editor for completeness
    with st.expander("JSON-параметры (произвольные)", expanded=not has_schema or show_raw_json):
        default_json = st.session_state.get("raw_json", "{}")
        raw = st.text_area("Входные параметры (JSON)", value=default_json, height=200, key="raw_json")
        extra_inputs: Dict[str, Any] = {}
        try:
            extra_inputs = json.loads(raw) if raw.strip() else {}
            if not isinstance(extra_inputs, dict):
                st.warning("JSON верхнего уровня должен быть объектом { ... }.")
                extra_inputs = {}
        except Exception as e:
            st.error(f"Ошибка разбора JSON: {e}")
            extra_inputs = {}

    # Merge inputs, JSON overrides named widgets
    final_inputs = {**input_values, **extra_inputs}

    # Prompt preview and run
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("\n")
        generate = st.button("🚀 Сгенерировать", type="primary")
    with col2:
        st.caption("Все параметры будут отправлены в Replicate как есть. Убедитесь в корректности значений.")

    if generate:
        if not final_inputs:
            st.warning("Нет входных параметров. Добавьте хотя бы prompt или укажите параметры в JSON.")
            st.stop()

        with st.spinner("Генерация... это может занять время"):
            try:
                # Prefer fully qualified model name (owner/model), Replicate will resolve latest version.
                result = client.run(MODEL_NAME, input=final_inputs)  # type: ignore[arg-type]
            except Exception as e:
                st.error("Ошибка генерации. Проверьте параметры или повторите попытку позже.")
                with st.expander("Подробности ошибки"):
                    st.exception(e)
                st.stop()

        # Handle results
        image_urls = extract_image_urls(result)
        if not image_urls:
            # Show raw result for debugging
            st.warning("Не удалось извлечь URL изображений из ответа. См. сырой ответ ниже.")
            with st.expander("Сырой ответ Replicate"):
                st.write(result)
            st.stop()

        # Save images
        primary_name = str(final_inputs.get("prompt") or final_inputs.get("text") or "image")
        saved_paths = save_images(image_urls, base_name=primary_name)

        # Show gallery
        st.success(f"Готово! Сохранено {len(saved_paths)} файл(ов) в папку {OUTPUTS_DIR}/")
        cols = st.columns(min(3, len(saved_paths)) or 1)
        for i, p in enumerate(saved_paths):
            with cols[i % len(cols)]:
                st.image(str(p), caption=p.name, use_column_width=True)
                st.code(str(p), language="text")

        with st.expander("URLs источников"):
            for url in image_urls:
                st.write(url)


if __name__ == "__main__":
    main()
