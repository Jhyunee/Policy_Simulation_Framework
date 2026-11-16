import concurrent.futures
import json
import os
import re
import threading
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Shared thread pool for wrapping vertex calls with a hard timeout. The google-genai
# SDK has no per-call timeout option, so blocking generate_content can hang
# silently when Google's load balancer drops a connection without sending FIN/RST.
# We submit the call here and enforce timeout via future.result(timeout=...).
_VERTEX_TIMEOUT_EXEC = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="vertex-call"
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class UsageTracker:
    """Process-wide accumulator for LLM token usage and call counts.

    A single instance (USAGE_TRACKER) is updated by every chat_completion call. CLI
    commands reset() it at the start of a run and snapshot() it at the end to write
    a per-run-dir log. Thread-safe via a single Lock so concurrent stage2 ToC
    forward/backward + scenario threads don't lose token counts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.calls_with_usage = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0

    def record(self, usage: dict | None) -> None:
        with self._lock:
            self.calls += 1
            if not usage:
                return
            self.calls_with_usage += 1
            self.prompt_tokens += int(usage.get("prompt_tokens") or usage.get("promptTokenCount") or 0)
            self.completion_tokens += int(
                usage.get("completion_tokens") or usage.get("candidatesTokenCount") or 0
            )
            self.total_tokens += int(usage.get("total_tokens") or usage.get("totalTokenCount") or 0)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                # Our canonical keys
                "llm_calls": self.calls,
                "llm_calls_with_usage": self.calls_with_usage,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                # Aliases for ported `stage2_toc/` code that expects Mirofish_v1's shape
                "call_count": self.calls,
                "elapsed_sec": 0.0,
            }

    # --- Compatibility shim for ported `stage2_toc/` code ---
    # Mirofish_v1's stage2_toc imports `token_tracker` and calls enable()/disable()/
    # is_enabled() and a 3-arg record(prompt_tokens, completion_tokens, elapsed)
    # variant. We make those no-ops / map them to our usage shape so we don't have
    # to edit every call site in the ported pipeline.

    def enable(self) -> None:
        return

    def disable(self) -> None:
        return

    def is_enabled(self) -> bool:
        return True


USAGE_TRACKER = UsageTracker()
# Alias for ported `stage2_toc/` code that imports `token_tracker`.
token_tracker = USAGE_TRACKER


_JSON_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n?", re.MULTILINE)


def _extract_json(text: str) -> str:
    """Recover JSON content from an LLM response that may include code fences or surrounding prose.

    Strategies, tried in order:
      1. Strip ```json ... ``` or ``` ... ``` fences.
      2. If the result still does not parse, slice the largest balanced {...} or [...] substring
         starting at the first { or [.
    """
    text = (text or "").strip()
    if not text:
        return text

    # Strip JSON-invalid control characters. Raw \n and \t inside string values
    # also break json.loads even though they're harmless in JSON structure —
    # we drop ALL control chars except space; structural whitespace in the JSON
    # tree is preserved by the cleanup that runs after this on _slice_balanced_json.
    # Strip the truly forbidden control chars (\x00-\x08, \x0b, \x0c, \x0e-\x1f).
    # We keep \x09 (tab), \x0a (newline), \x0d (carriage return) and rely on
    # json.loads(strict=False) below to tolerate them inside string values.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    if text.startswith("```"):
        without_open = _JSON_FENCE_RE.sub("", text, count=1)
        if without_open.rstrip().endswith("```"):
            without_open = without_open.rstrip()[: -3]
        text = without_open.strip()

    try:
        # strict=False allows raw control chars (notably \n, \t) inside string
        # values — LLMs frequently emit unescaped newlines in `narrative` fields.
        json.loads(text, strict=False)
        return text
    except json.JSONDecodeError:
        pass

    sliced = _slice_balanced_json(text)
    return sliced if sliced is not None else text


def _slice_balanced_json(text: str) -> str | None:
    """Return the substring from the first opening brace/bracket to its matching close, or None."""
    open_chars = {"{": "}", "[": "]"}
    start = -1
    opener = ""
    for i, ch in enumerate(text):
        if ch in open_chars:
            start = i
            opener = ch
            break
    if start < 0:
        return None
    closer = open_chars[opener]
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalize_usage(usage: dict | None) -> dict:
    """Return canonical {prompt_tokens, completion_tokens, total_tokens} from any provider shape."""
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("promptTokenCount") or 0),
        "completion_tokens": int(
            usage.get("completion_tokens") or usage.get("candidatesTokenCount") or 0
        ),
        "total_tokens": int(usage.get("total_tokens") or usage.get("totalTokenCount") or 0),
    }


def chat_completion_with_usage(
    messages: list[dict],
    config: dict,
    max_retries: int = 3,
    initial_backoff: float = 2.0,
) -> tuple[str, dict]:
    """Like chat_completion but also returns the normalized usage dict from the successful attempt."""
    llm_config = config.get("llm", {})
    max_retries = int(llm_config.get("max_retries", max_retries))
    initial_backoff = float(llm_config.get("initial_backoff", initial_backoff))
    last_exc: Exception | None = None
    last_raw: str = ""
    for attempt in range(max_retries):
        try:
            raw, usage = _chat_completion_once(messages, config)
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            last_exc = exc
            last_raw = ""
            if attempt < max_retries - 1:
                wait = initial_backoff * (2 ** attempt)
                print(f"[llm] HTTP/transport error on attempt {attempt + 1} ({type(exc).__name__}): {str(exc)[:160]}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise

        last_raw = raw
        cleaned = _extract_json(raw)
        if not cleaned:
            last_exc = ValueError("LLM returned empty response")
            if attempt < max_retries - 1:
                wait = initial_backoff * (2 ** attempt)
                print(f"[llm] empty response on attempt {attempt + 1}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise last_exc

        try:
            json.loads(cleaned, strict=False)
            return cleaned, _normalize_usage(usage)
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = initial_backoff * (2 ** attempt)
                print(f"[llm] JSONDecodeError on attempt {attempt + 1}: {str(exc)[:160]}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            preview = (last_raw[:240] + "...") if len(last_raw) > 240 else last_raw
            raise json.JSONDecodeError(
                f"{exc.msg} (raw response prefix: {preview!r})",
                exc.doc,
                exc.pos,
            )

    if last_exc:
        raise last_exc
    return "", _normalize_usage(None)


def chat_completion(
    messages: list[dict],
    config: dict,
    max_retries: int = 3,
    initial_backoff: float = 2.0,
) -> str:
    """Call the configured LLM and return a JSON-parseable response string.

    Retries on transient HTTP failures, empty responses, and unparseable JSON. Each retry waits
    initial_backoff * 2**attempt seconds. Raises on final failure with a short prefix of the last
    raw response in the error message to make debugging easier.
    """
    content, _ = chat_completion_with_usage(messages, config, max_retries, initial_backoff)
    return content


def _chat_completion_once(messages: list[dict], config: dict) -> tuple[str, dict | None]:
    """Call the configured provider and return (full_text, raw_usage_or_None).
    Supports OpenAI-format (streaming SSE) and Google Gemini-format (JSON) based on config api_type."""
    api_type = config["llm"].get("api_type", "openai")

    # Vertex AI uses Application Default Credentials (gcloud auth), not an API key.
    if api_type == "google_vertex":
        return _gemini_vertex_completion(messages, config)

    api_key = os.environ.get(config["llm"]["api_key_env"])
    if not api_key:
        raise ValueError(
            f"API key not found. Set the '{config['llm']['api_key_env']}' environment variable."
        )

    if api_type == "google":
        return _gemini_completion(messages, config, api_key)
    elif api_type == "google_direct":
        return _gemini_direct_completion(messages, config, api_key)
    else:
        return _openai_completion(messages, config, api_key)


def _gemini_vertex_completion(messages: list[dict], config: dict) -> tuple[str, dict | None]:
    """Google Vertex AI call via google-genai SDK.

    Uses Application Default Credentials (run ``gcloud auth application-default login``
    or set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON path).
    Reads ``project`` and ``location`` from config (with env-var override).
    """
    from google import genai
    from google.genai import types

    project = config["llm"].get("project") or os.environ.get("GCP_PROJECT")
    location = config["llm"].get("location") or os.environ.get("GCP_LOCATION", "us-central1")
    model = config["llm"].get("model", "gemini-2.5-flash")

    if not project:
        raise ValueError("Vertex AI requires 'project' in config or GCP_PROJECT env var.")

    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(f"[System]\n{m['content']}")
        elif m["role"] == "user":
            parts.append(f"[User]\n{m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"[Assistant]\n{m['content']}")
    combined = "\n\n".join(parts)

    client = genai.Client(vertexai=True, project=project, location=location)
    gen_cfg = types.GenerateContentConfig(
        temperature=config["llm"].get("temperature", 0.7),
        max_output_tokens=config["llm"].get("max_tokens", 8192),
        response_mime_type="application/json",
    )
    # Hard timeout for the generate_content call. google-genai SDK has no native
    # per-call timeout, so we wrap in a future and time out via future.result().
    # On timeout the underlying thread is left to finish (or never finish) — we just
    # stop waiting. Configurable via config["llm"]["request_timeout"] (default 90s).
    timeout_sec = float(config["llm"].get("request_timeout", 90.0))
    future = _VERTEX_TIMEOUT_EXEC.submit(
        client.models.generate_content,
        model=model,
        contents=combined,
        config=gen_cfg,
    )
    try:
        response = future.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise ValueError(
            f"vertex_genai_error: TimeoutError: no response after {timeout_sec}s"
        ) from None
    except Exception as exc:
        # Re-raise as ValueError so the outer retry loop (which catches ValueError,
        # RequestException, KeyError, IndexError) picks it up and backs off. Covers
        # google.genai.errors.ClientError (429 quota), ServerError (5xx), etc.
        raise ValueError(f"vertex_genai_error: {type(exc).__name__}: {str(exc)[:240]}") from exc
    full_text = response.text or ""
    usage = None
    if getattr(response, "usage_metadata", None):
        um = response.usage_metadata
        usage = {
            "promptTokenCount": getattr(um, "prompt_token_count", None),
            "candidatesTokenCount": getattr(um, "candidates_token_count", None),
            "totalTokenCount": getattr(um, "total_token_count", None),
        }
    USAGE_TRACKER.record(usage)
    return full_text, usage


def _gemini_direct_completion(messages: list[dict], config: dict, api_key: str) -> tuple[str, dict | None]:
    """Google AI Studio direct API call.

    Differs from Luxia-bridged ``_gemini_completion``:
      - Header auth: ``x-goog-api-key`` instead of ``apikey``.
      - Body shape: ``{"contents":[{"parts":[{"text": ...}]}]}`` (array of role parts).
      - Response: streamGenerateContent returns a JSON ARRAY of chunks (not SSE).
    """
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(f"[System]\n{m['content']}")
        elif m["role"] == "user":
            parts.append(f"[User]\n{m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"[Assistant]\n{m['content']}")
    combined = "\n\n".join(parts)

    payload = {
        "contents": [{"parts": [{"text": combined}]}],
        # Force Gemini to emit valid JSON (escaped quotes, no markdown fences)
        # so json.loads downstream doesn't trip on raw quotes / newlines in
        # narrative strings.
        "generationConfig": {
            "response_mime_type": "application/json",
            "maxOutputTokens": config["llm"].get("max_tokens", 8192),
            "temperature": config["llm"].get("temperature", 0.7),
        },
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    timeout = config["llm"].get("request_timeout", 180)
    response = requests.post(config["llm"]["url"], headers=headers, json=payload, timeout=(10, timeout))
    if not response.ok:
        print(f"[gemini_direct] {response.status_code} error: {response.text[:500]}")
    response.raise_for_status()

    chunks = response.json()
    if isinstance(chunks, dict):
        chunks = [chunks]
    full_text = ""
    usage: dict | None = None
    for ch in chunks:
        for cand in ch.get("candidates") or []:
            content = cand.get("content") or {}
            for p in content.get("parts") or []:
                if p.get("text"):
                    full_text += p["text"]
        if ch.get("usageMetadata"):
            usage = ch["usageMetadata"]
    USAGE_TRACKER.record(usage)
    return full_text, usage


def _openai_completion(messages: list[dict], config: dict, api_key: str) -> tuple[str, dict | None]:
    """OpenAI-compatible streaming SSE call.

    Supports two auth styles via config["llm"]["auth_header"]:
      - "apikey" (default, used by Luxia bridge)
      - "bearer" (used by OpenAI, Deepseek, Together, etc.)
    """
    auth_style = config["llm"].get("auth_header", "apikey")
    if auth_style == "bearer":
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    else:
        headers = {"apikey": api_key, "Content-Type": "application/json"}
    # GPT-5 / o1 / o3 family use `max_completion_tokens` instead of `max_tokens`,
    # and reject any temperature other than the default (1).
    model_name = config["llm"]["model"]
    is_reasoning_family = any(model_name.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
    tokens_field = config["llm"].get(
        "tokens_field",
        "max_completion_tokens" if is_reasoning_family else "max_tokens",
    )
    payload = {
        "model": model_name,
        "messages": messages,
        tokens_field: config["llm"].get("max_tokens", 2048),
        "stream": True,
    }
    if not is_reasoning_family:
        payload["temperature"] = config["llm"].get("temperature", 0.7)
    if "thinking" in config["llm"]:
        payload["thinking"] = config["llm"]["thinking"]
    # Ask OpenAI-compatible providers (DeepSeek, OpenAI, Together, ...) to emit
    # a final SSE chunk with token usage. Bridges that ignore the field still work.
    payload["stream_options"] = {"include_usage": True}

    timeout = config["llm"].get("request_timeout", 180)
    response = requests.post(
        config["llm"]["url"],
        headers=headers,
        json=payload,
        stream=True,
        timeout=(10, timeout),
    )
    response.raise_for_status()

    full_text = ""
    usage: dict | None = None
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded == "data: [DONE]":
            break
        if decoded.startswith("data: "):
            chunk = json.loads(decoded[6:])
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    full_text += delta["content"]
            # Final chunk (or any chunk) with usage info — keep the latest.
            if chunk.get("usage"):
                usage = chunk["usage"]

    USAGE_TRACKER.record(usage)
    return full_text, usage


def _gemini_completion(messages: list[dict], config: dict, api_key: str) -> tuple[str, dict | None]:
    """Google Gemini-format call via Luxia bridge.

    Supports two endpoint behaviors:
      - Legacy `flash20/content` endpoint: non-streaming JSON response.
      - Dynamic `dynamic/content` endpoint: requires `stream: true`, returns SSE.
    Auto-detected by URL substring. Dynamic endpoint streams data: {...} chunks
    that we concatenate.
    """
    headers = {
        "apikey": api_key,
        "Content-Type": "application/json",
    }
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(f"[System]\n{m['content']}")
        elif m["role"] == "user":
            parts.append(f"[User]\n{m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"[Assistant]\n{m['content']}")
    combined = "\n\n".join(parts)

    url = config["llm"]["url"]
    is_dynamic = "/dynamic/" in url
    payload = {
        "model": config["llm"]["model"],
        "contents": combined,
        "generationConfig": {
            "response_mime_type": "application/json",
            "maxOutputTokens": config["llm"].get("max_tokens", 8192),
            "temperature": config["llm"].get("temperature", 0.7),
        },
    }
    if is_dynamic:
        payload["stream"] = True

    timeout = config["llm"].get("request_timeout", 180)
    if is_dynamic:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=(10, timeout))
        if not response.ok:
            print(f"[gemini] {response.status_code} error: {response.text[:500]}")
        response.raise_for_status()
        full_text = ""
        usage: dict | None = None
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded == "data: [DONE]":
                break
            if decoded.startswith("data: "):
                try:
                    chunk = json.loads(decoded[6:])
                except json.JSONDecodeError:
                    continue
                candidates = chunk.get("candidates") or []
                if candidates:
                    content = candidates[0].get("content") or {}
                    for p in content.get("parts") or []:
                        if p.get("text"):
                            full_text += p["text"]
                if chunk.get("usageMetadata"):
                    usage = chunk["usageMetadata"]
        USAGE_TRACKER.record(usage)
        return full_text, usage

    response = requests.post(url, headers=headers, json=payload, timeout=(10, timeout))
    if not response.ok:
        print(f"[gemini] {response.status_code} error: {response.text[:500]}")
    response.raise_for_status()
    data = response.json()
    usage = data.get("usageMetadata") or data.get("usage")
    USAGE_TRACKER.record(usage)
    return data["candidates"][0]["content"]["parts"][0]["text"], usage
