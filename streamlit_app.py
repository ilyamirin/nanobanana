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
    # Replace whitespace with underscore for readability
    name = re.sub(r"\s+", "_", name)
    # Keep Unicode letters/digits and safe punctuation (._-); drop other illegal characters
    name = re.sub(r"[^\w._-]", "", name)
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
    lower_name = name.lower()
    is_prompt_like = lower_name in {"prompt", "text", "caption"}

    # Heuristics for image-like inputs (URLs or file upload)
    fmt = (schema.get("format") or "").lower()
    items_fmt = (schema.get("items", {}).get("format") or "").lower() if isinstance(schema.get("items"), dict) else ""
    is_image_name = ("image" in lower_name) or (lower_name.endswith(":image"))
    is_image_format = fmt in {"uri", "url", "image", "binary"} or items_fmt in {"uri", "url", "image", "binary"}
    is_array = (type_ == "array")
    is_image_array = is_array and (is_image_name or is_image_format)
    is_single_image = (type_ in ("string", None)) and (is_image_name or is_image_format)

    help_text = desc + (" (required)" if required else "")

    if enum:
        # Enum select
        return st.selectbox(title, options=enum, index=(enum.index(default) if default in enum else 0), help=help_text, key=f"inp_{name}")

    # Special handling: image inputs (0..10 files)
    if is_image_array:
        uploaded = st.file_uploader(
            label=(title + " (0–10 изображений)"),
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=f"inp_{name}"
        )
        files = list(uploaded) if uploaded else []
        if len(files) > 10:
            st.warning(f"Поле {name}: загружено больше 10 файлов, лишние будут проигнорированы.")
            files = files[:10]
        # Allow empty list (0 images)
        return files

    if is_single_image:
        uploaded = st.file_uploader(
            label=(title + " (изображение, опционально)"),
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            key=f"inp_{name}"
        )
        # Allow None (0 images) or a single file
        return uploaded

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

# ----------------------------
# Helpers: Batch import (Markdown / CSV)
# ----------------------------

def _parse_markdown_table(md_text: str) -> List[Dict[str, Any]]:
    """Parse simple GitHub-style Markdown tables.
    Returns list of dict rows. Header row determines keys.
    """
    lines = [ln.strip() for ln in md_text.splitlines()]
    # Extract contiguous block(s) of table lines (starting with | or containing multiple pipes)
    tables: List[List[str]] = []
    cur: List[str] = []
    for ln in lines:
        if (ln.startswith("|") and "|" in ln.strip("|")) or (ln.count("|") >= 2):
            cur.append(ln)
        else:
            if cur:
                tables.append(cur)
                cur = []
    if cur:
        tables.append(cur)
    results: List[Dict[str, Any]] = []
    for tbl in tables:
        if len(tbl) < 2:
            continue
        header = tbl[0]
        body = tbl[1:]
        # remove alignment row if present
        if body and set(body[0].replace(" ", "").replace(":", "-").strip("|")) <= {"-", "|"}:
            body = body[1:]
        headers = [h.strip() for h in header.strip().strip("|").split("|")]
        norm_headers = [h.lower() for h in headers]
        for row in body:
            cells = [c.strip().strip('"') for c in row.strip().strip("|").split("|")]
            if len(cells) != len(headers):
                if len(cells) < len(headers):
                    cells += [""] * (len(headers) - len(cells))
                else:
                    cells = cells[: len(headers)]
            item = {norm_headers[i]: cells[i] for i in range(len(headers))}
            if any(v for v in item.values()):
                results.append(item)
    return results


def _parse_markdown_json_blocks(md_text: str) -> List[Dict[str, Any]]:
    """Parse ```json ...``` blocks; each must be a JSON object containing at least `prompt`.
    Returns list of dicts.
    """
    results: List[Dict[str, Any]] = []
    import re as _re
    pattern = _re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", _re.IGNORECASE)
    for m in pattern.finditer(md_text):
        block = m.group(1)
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and ("prompt" in obj or "text" in obj or "caption" in obj):
                results.append(obj)
        except Exception:
            continue
    return results


def import_from_markdown(md_text: str) -> List[Dict[str, Any]]:
    """Try both table and json code blocks. Normalize keys.
    Normalized keys: title, prompt (+ any extras kept).
    Also supports Russian headers: 'название', 'промпт', 'описание'.
    """
    items: List[Dict[str, Any]] = []
    table_rows = _parse_markdown_table(md_text)
    json_rows = _parse_markdown_json_blocks(md_text)
    rows: List[Dict[str, Any]] = []
    if table_rows:
        rows.extend(table_rows)
    if json_rows:
        rows.extend(json_rows)
    def norm_key(k: str) -> str:
        k_low = k.lower().strip()
        mapping = {
            "название": "title",
            "название карты": "title",
            "заголовок": "title",
            "имя": "title",
            "title": "title",
            "name": "title",
            # size-related
            "размер": "size",
            "size": "size",
            "aspect ratio": "size",
            "aspect_ratio": "size",
            "aspect": "size",
            "ratio": "size",
            "соотношение": "size",
            "соотношение сторон": "size",
            "формат": "size",
            # prompt-related
            "prompt": "prompt",
            "text": "prompt",
            "caption": "prompt",
            "промпт": "prompt",
            "промпт для генерации": "prompt",
            "промпт для генерации (на русском)": "prompt",
            # fallbacks / extra
            "описание": "description",
            "ключевой образ / жест": "description",
            "№": "index",
        }
        return mapping.get(k_low, k_low)
    for row in rows:
        norm = {norm_key(k): v for k, v in row.items()}
        title = str(norm.get("title") or norm.get("name") or norm.get("index") or "").strip()
        prompt_val = norm.get("prompt") or norm.get("text") or norm.get("caption") or ""
        if not prompt_val:
            prompt_val = norm.get("description", "")
        if not prompt_val:
            continue
        item = {k: v for k, v in norm.items()}
        item["title"] = title
        item["prompt"] = prompt_val
        # keep normalized size if present
        if item.get("size") is None:
            item.pop("size", None)
        items.append(item)
    return items


def import_from_csv_bytes(data: bytes, encoding: str = "utf-8") -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse CSV bytes. Returns (rows, headers). Tries delimiter sniffing. Keeps raw headers.
    """
    import io, csv
    text = data.decode(encoding, errors="replace")
    sniffer = csv.Sniffer()
    try:
        sample = "\n".join(text.splitlines()[:3])
        dialect = sniffer.sniff(sample) if sample else csv.excel
    except Exception:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    rows: List[List[str]] = [r for r in reader]
    if not rows:
        return [], []
    headers = [h.strip() for h in rows[0]]
    out: List[Dict[str, Any]] = []
    for r in rows[1:]:
        if not any((cell or "").strip() for cell in r):
            continue
        rec = {headers[i] if i < len(headers) else f"col{i}": (r[i] if i < len(r) else "") for i in range(max(len(headers), len(r)))}
        out.append(rec)
    return out, headers


def _detect_prompt_key_from_schema(props: Dict[str, Any]) -> str:
    for key in ("prompt", "text", "caption"):
        if key in props:
            return key
    return "prompt"


def _detect_size_application(props: Dict[str, Any]) -> Dict[str, Any]:
    """Detect how to apply per-item size/aspect overrides based on schema properties.
    Returns a dict with keys:
      - mode: 'aspect' | 'wh' | 'single' | 'none'
      - key(s): depending on mode: 'aspect_key' or ('w_key','h_key') or 'single_key'
      - enums: optional list of allowed enum values for aspect/single.
    """
    props = props or {}
    # Prefer explicit aspect ratio
    if "aspect_ratio" in props:
        enum = props["aspect_ratio"].get("enum") if isinstance(props["aspect_ratio"], dict) else None
        return {"mode": "aspect", "aspect_key": "aspect_ratio", "enums": enum}
    # Sometimes named 'aspect' or similar
    for k in props.keys():
        if k.lower() in {"aspect", "aspectratio", "ratio"}:
            enum = props[k].get("enum") if isinstance(props[k], dict) else None
            return {"mode": "aspect", "aspect_key": k, "enums": enum}
    # Width/Height pair
    if "width" in props and "height" in props:
        return {"mode": "wh", "w_key": "width", "h_key": "height"}
    # Single size-like keys
    for k in ("size", "image_size", "image_dimensions", "dimensions", "output_size"):
        if k in props:
            enum = props[k].get("enum") if isinstance(props[k], dict) else None
            return {"mode": "single", "single_key": k, "enums": enum}
    return {"mode": "none"}


def _normalize_aspect_string(s: str) -> Optional[str]:
    """Return canonical aspect string like '9:16' or '16:9' or '1:1' if possible."""
    if not s:
        return None
    raw = str(s).strip().lower()
    # Named presets
    named = {
        "square": "1:1", "квадрат": "1:1", "1:1": "1:1",
        "portrait": "9:16", "портрет": "9:16", "вертик": "9:16",
        "landscape": "16:9", "альбом": "16:9", "горизонт": "16:9",
    }
    for k, v in named.items():
        if k in raw:
            return v
    import re as _re
    m = _re.match(r"^(\d+)\s*[:x×/*-]\s*(\d+)$", raw)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > 0 and b > 0:
            # reduce by gcd
            from math import gcd
            g = gcd(a, b)
            return f"{a//g}:{b//g}"
    return None


def _parse_size_to_inputs(size_val: str, props: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Parse free-form size string and map to model inputs according to schema props.
    Returns (overrides_dict, warning_message). If cannot parse, returns ({}, msg).
    Accepted forms: '9:16', '16:9', '1:1', '1024x576', '1024×576', '800*1200',
    named presets 'portrait'/'landscape'/'square' and RU equivalents.
    """
    mode_info = _detect_size_application(props)
    mode = mode_info.get("mode")
    if not size_val or mode == "none":
        return {}, None
    s = str(size_val).strip()
    # Try absolute WxH first
    import re as _re
    abs_m = _re.match(r"^(\d{2,5})\s*[x×*]\s*(\d{2,5})$", s.lower())
    if abs_m:
        w, h = int(abs_m.group(1)), int(abs_m.group(2))
        if w <= 0 or h <= 0:
            return {}, f"Некорректный размер: {size_val}"
        if mode == "wh":
            return {mode_info["w_key"]: w, mode_info["h_key"]: h}, None
        if mode == "aspect":
            # reduce to ratio
            from math import gcd
            g = gcd(w, h)
            aspect = f"{w//g}:{h//g}"
            enums = mode_info.get("enums") or []
            if enums and aspect not in enums:
                # Try to find compatible value (some enums may list 'portrait', etc.)
                # Fallback to first enum
                return {mode_info["aspect_key"]: enums[0]}, None
            return {mode_info["aspect_key"]: aspect}, None
        if mode == "single":
            # If enums exist, prefer nearest matching textual representation
            enums = mode_info.get("enums") or []
            if enums:
                # Try exact 'WxH'
                val = f"{w}x{h}"
                if val in enums:
                    return {mode_info["single_key"]: val}, None
                # Try aspect fallback
                from math import gcd
                g = gcd(w, h)
                aspect = f"{w//g}:{h//g}"
                if aspect in enums:
                    return {mode_info["single_key"]: aspect}, None
                return {mode_info["single_key"]: enums[0]}, None
            return {mode_info["single_key"]: f"{w}x{h}"}, None
    # Try aspect forms
    aspect = _normalize_aspect_string(s)
    if not aspect:
        return {}, f"Не удалось распознать размер: {size_val}"
    if mode == "aspect":
        enums = mode_info.get("enums") or []
        if enums and aspect not in enums and enums:
            return {mode_info["aspect_key"]: enums[0]}, None
        return {mode_info["aspect_key"]: aspect}, None
    if mode == "wh":
        # Without absolute pixels we cannot set width/height reliably
        return {}, f"Для этой модели нужен формат WxH в пикселях; пропуск значения '{size_val}'"
    if mode == "single":
        enums = mode_info.get("enums") or []
        if enums and aspect not in enums:
            return {mode_info["single_key"]: enums[0]}, None
        return {mode_info["single_key"]: aspect}, None
    return {}, None


def _detect_multi_output_key(props: Dict[str, Any]) -> Optional[str]:
    """Try to find a key in the schema that controls number of outputs per single run.
    Common names: num_outputs, num_samples, num_images, images, n, samples.
    Returns original key name from schema (preserve exact casing) or None.
    """
    if not props:
        return None
    candidates = {"num_outputs", "num_samples", "num_images", "images", "n", "samples"}
    for k in props.keys():
        lk = k.lower()
        if lk in candidates:
            return k
    return None

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
        # Кэширование схемы и версии модели в пределах сессии Streamlit
        version_id = st.session_state.get("version_id")
        openapi_schema = st.session_state.get("openapi_schema")

        if not version_id or not openapi_schema:
            with st.spinner("Загрузка схемы модели..."):
                _version_id = fetch_latest_version_id(settings.replicate_api_token)
                _openapi_schema = (
                    fetch_openapi_schema_for_version(_version_id, settings.replicate_api_token)
                    if _version_id
                    else None
                )
            # Сохраняем в session_state только если успешно получили и версию, и схему
            if _version_id and _openapi_schema:
                st.session_state["version_id"] = _version_id
                st.session_state["openapi_schema"] = _openapi_schema
                version_id, openapi_schema = _version_id, _openapi_schema
            else:
                # Покажем то, что смогли получить (возможно None)
                version_id, openapi_schema = _version_id, _openapi_schema

        # Кнопка для ручного обновления (очистка кэша)
        if st.button(
            "Обновить схему",
            help="Принудительно перезагрузить схему модели из API (очистить кэш на время текущей сессии)",
        ):
            st.session_state.pop("version_id", None)
            st.session_state.pop("openapi_schema", None)
            st.experimental_rerun()

        if version_id:
            st.success(f"Версия модели: {version_id[:8]}…")
        else:
            st.warning("Версия модели не раскрывается API. Доступен только ручной ввод параметров.")

        show_raw_json = st.toggle("Показывать редактор JSON", value=False, help="Позволяет задать любые параметры руками, если чего-то нет в форме.")

        st.divider()
        st.subheader("Пакетная загрузка")
        st.caption("Загрузите Markdown (.md) с таблицей/JSON‑блоками или CSV с колонками.")
        batch_file = st.file_uploader("Файл с промптами (Markdown или CSV)", type=["md", "markdown", "csv"], key="batch_file")
        parsed_items: List[Dict[str, Any]] = []
        csv_headers: List[str] = []
        csv_rows: List[Dict[str, Any]] = []
        if batch_file is not None:
            try:
                if batch_file.type in ("text/markdown",) or batch_file.name.lower().endswith((".md", ".markdown")):
                    md_bytes = batch_file.read()
                    md_text = md_bytes.decode("utf-8", errors="replace")
                    parsed_items = import_from_markdown(md_text)
                else:
                    data = batch_file.read()
                    csv_rows, csv_headers = import_from_csv_bytes(data)
                    # Try to map CSV to normalized items
                    def guess_key(keys: List[str], *candidates: str) -> Optional[str]:
                        low = [k.lower() for k in keys]
                        for cand in candidates:
                            if cand in low:
                                return keys[low.index(cand)]
                        return None
                    title_key = guess_key(csv_headers, "имя", "название", "название карты", "title", "name", "№")
                    prompt_key = guess_key(csv_headers, "промпт для генерации (на русском)", "промпт для генерации", "промпт", "prompt", "text", "caption", "описание", "ключевой образ / жест")
                    size_key = guess_key(csv_headers, "размер", "size", "aspect ratio", "aspect_ratio", "aspect", "ratio", "соотношение", "соотношение сторон", "формат")
                    # Allow user override
                    with st.expander("Настройка колонок CSV", expanded=True):
                        title_key = st.selectbox("Колонка с названием (опционально)", options=["<нет>"] + csv_headers, index=(0 if not title_key else (["<нет>"] + csv_headers).index(title_key)))
                        prompt_key = st.selectbox("Колонка с промптом/описанием", options=csv_headers, index=(csv_headers.index(prompt_key) if prompt_key in csv_headers else 0))
                        size_key = st.selectbox("Колонка с размером (опционально)", options=["<нет>"] + csv_headers, index=(0 if not size_key else (["<нет>"] + csv_headers).index(size_key)))
                    for r in csv_rows:
                        item_title = ("" if title_key in (None, "<нет>") else str(r.get(title_key, "")).strip())
                        item_prompt = str(r.get(prompt_key, "")).strip()
                        if not item_prompt:
                            continue
                        item_size = ("" if size_key in (None, "<нет>") else str(r.get(size_key, "")).strip())
                        rec = {"title": item_title, "prompt": item_prompt, **r}
                        if item_size:
                            rec["size"] = item_size
                        parsed_items.append(rec)
            except Exception as e:
                st.error(f"Ошибка разбора файла: {e}")
                parsed_items = []
        if parsed_items:
            st.session_state["batch_items"] = parsed_items
            st.success(f"Загружено записей: {len(parsed_items)}")
        else:
            if batch_file is not None:
                st.warning("Не удалось извлечь промпты из файла.")

        # Let user clear
        if st.button("Очистить пакет", use_container_width=True):
            st.session_state.pop("batch_items", None)

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

    # How many variants to generate per prompt
    num_variants: int = st.number_input(
        "Количество экспериментов (вариантов на 1 промпт)",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Сгенерировать несколько вариантов изображений для одного и того же промпта"
    )

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

        # Determine if the model supports multi-output in a single run
        multi_key = _detect_multi_output_key(props)
        inputs_for_run = {**final_inputs}
        max_allowed = None
        if multi_key and isinstance(props.get(multi_key), dict):
            max_allowed = props[multi_key].get("maximum")
        planned = num_variants
        if max_allowed is not None:
            try:
                planned = min(int(max_allowed), planned)
            except Exception:
                pass
        aggregated_urls: List[str] = []

        if multi_key:
            inputs_for_run[multi_key] = planned
            with st.spinner("Генерация... это может занять время"):
                try:
                    result = client.run(MODEL_NAME, input=inputs_for_run)  # type: ignore[arg-type]
                    aggregated_urls.extend(extract_image_urls(result))
                except Exception as e:
                    st.error("Ошибка генерации. Проверьте параметры или повторите попытку позже.")
                    with st.expander("Подробности ошибки"):
                        st.exception(e)
                    st.stop()
        else:
            progress = st.progress(0.0, text="Генерация вариантов...")
            errors: List[str] = []
            for i in range(planned):
                progress.progress((i)/planned, text=f"Вариант {i+1}/{planned}")
                try:
                    result = client.run(MODEL_NAME, input=inputs_for_run)  # type: ignore[arg-type]
                    urls = extract_image_urls(result)
                    aggregated_urls.extend(urls)
                except Exception as e:
                    errors.append(str(e))
            progress.progress(1.0, text="Готово")
            if errors:
                with st.expander("Ошибки некоторых запусков"):
                    for msg in errors:
                        st.write(msg)

        # Handle results
        if not aggregated_urls:
            st.warning("Не удалось извлечь URL изображений из ответа(ов). См. подробности выше.")
            st.stop()

        # Save images
        primary_name = str(final_inputs.get("prompt") or final_inputs.get("text") or "image")
        saved_paths = save_images(aggregated_urls, base_name=primary_name)

        # Show gallery
        st.success(f"Готово! Сохранено {len(saved_paths)} файл(ов) в папку {OUTPUTS_DIR}/")
        cols = st.columns(min(3, len(saved_paths)) or 1)
        for i, p in enumerate(saved_paths):
            with cols[i % len(cols)]:
                st.image(str(p), caption=p.name, use_container_width=True)
                # st.code(str(p), language="text")  # TODO: fix streamlit bug

        with st.expander("URLs источников"):
            for url in aggregated_urls:
                st.write(url)

    # Batch section (if any items uploaded)
    batch_items: List[Dict[str, Any]] = st.session_state.get("batch_items", [])
    if batch_items:
        st.subheader("Пакетная генерация")
        st.caption("Выберите записи для запуска. Будут использованы текущие параметры формы как общие, а текст из файла подставится в поле промпта.")
        # Detect prompt key expected by model
        prompt_field = _detect_prompt_key_from_schema(props)
        st.write(f"Поле для промпта по схеме: `{prompt_field}`")
        # Preview table
        preview_rows = [{"title": it.get("title", ""), "prompt": it.get("prompt", "")[:120]} for it in batch_items]
        st.dataframe(preview_rows, use_container_width=True)
        # Selection
        all_labels = [f"{(it.get('title') or '')} — {it.get('prompt','')[:40]}" for it in batch_items]
        selected = st.multiselect("Какие записи запускать?", options=list(range(len(batch_items))), format_func=lambda i: all_labels[i], default=list(range(len(batch_items))))
        run_batch = st.button("🚀 Запустить пакет", type="primary")
        if run_batch:
            if not selected:
                st.warning("Не выбраны записи для генерации.")
            else:
                progress = st.progress(0.0, text="Подготовка...")
                status_area = st.empty()
                successes = 0
                failures = 0
                for idx, i in enumerate(selected, start=1):
                    item = batch_items[i]
                    # Prefer provided title; if absent, fall back to index from source (e.g., '№') or sequence number
                    item_title = (item.get("title") or str(item.get("index") or f"{i+1}"))
                    item_prompt = item.get("prompt") or ""
                    if not item_prompt:
                        failures += 1
                        status_area.warning(f"[{idx}/{len(selected)}] Пропуск: пустой промпт для '{item_title}'.")
                        progress.progress(idx/len(selected))
                        continue
                    # Compose inputs: current UI values + prompt field overridden per item
                    inputs = {**final_inputs}
                    inputs[prompt_field] = item_prompt
                    # If per-item size provided, try to parse and override corresponding inputs
                    size_val = (item.get("size") or "").strip()
                    if size_val:
                        overrides, warn_msg = _parse_size_to_inputs(size_val, props)
                        if overrides:
                            inputs.update(overrides)
                        if warn_msg:
                            status_area.warning(f"[{idx}/{len(selected)}] Размер '{size_val}': {warn_msg}")
                    try:
                        status_area.info(f"[{idx}/{len(selected)}] Генерация: {item_title}")
                        result = client.run(MODEL_NAME, input=inputs)  # type: ignore[arg-type]
                        urls = extract_image_urls(result)
                        if not urls:
                            failures += 1
                            status_area.error(f"[{idx}/{len(selected)}] Нет изображений: {item_title}")
                        else:
                            save_images(urls, base_name=item_title or inputs.get(prompt_field, "image"))
                            successes += 1
                            status_area.success(f"[{idx}/{len(selected)}] Готово: {item_title} ({len(urls)} URL)")
                    except Exception as e:
                        failures += 1
                        with st.expander(f"Ошибка для '{item_title}'"):
                            st.exception(e)
                    progress.progress(idx/len(selected))
                st.info(f"Пакет завершён: успехов {successes}, ошибок {failures}.")


if __name__ == "__main__":
    main()
