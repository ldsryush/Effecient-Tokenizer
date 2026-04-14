"""
LLM Dispatcher
---------------
Handles the outbound request to the target LLM API.

Responsibilities:
  - Route to the correct endpoint (OpenAI vs Anthropic)
  - Inject prompt-cache hints when the cache router signals a static-prefix hit
  - Retry with exponential backoff on transient errors (429, 5xx)
  - Fallback to an alternative model if the primary is rate-limited or unavailable
  - Record the final post-compression token count for the observability layer

The dispatcher is intentionally decoupled from the pipeline — it receives a
fully-optimised payload and returns a standard response dict.

Environment variables:
  OPENAI_API_KEY        — forwarded as Bearer token to OpenAI
  ANTHROPIC_API_KEY     — forwarded as x-api-key to Anthropic
  LLM_FALLBACK_MODEL    — model to try if primary fails (default: gpt-4o-mini)
  LLM_MAX_RETRIES       — max retry attempts (default: 3)
  LLM_TIMEOUT_S         — per-request timeout in seconds (default: 60)
  DISPATCH_DRY_RUN      — if "true", skip actual API call (for testing)
"""
from __future__ import annotations
import asyncio
import os
import time
import json
import math
from typing import Optional, Any

_DRY_RUN: bool = os.environ.get("DISPATCH_DRY_RUN", "false").lower() == "true"
_MAX_RETRIES: int = int(os.environ.get("LLM_MAX_RETRIES", "3"))
_TIMEOUT_S: float = float(os.environ.get("LLM_TIMEOUT_S", "60"))
_FALLBACK_MODEL: str = os.environ.get("LLM_FALLBACK_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Endpoint routing
# ---------------------------------------------------------------------------

def _is_anthropic(model: str) -> bool:
    return model.lower().startswith("claude")


def _openai_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def _anthropic_url() -> str:
    return os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

def _build_openai_payload(
    messages: list[dict],
    model: str,
    static_cache_hit: bool,
    **kwargs: Any,
) -> dict:
    payload: dict = {
        "model":    model,
        "messages": messages,
        **kwargs,
    }
    # OpenAI supports prompt caching automatically for qualifying prompts;
    # nothing extra needed — but we annotate the payload for observability.
    if static_cache_hit:
        payload["_cache_hint"] = "static_prefix_cached"
    return payload


def _build_anthropic_payload(
    messages: list[dict],
    system_prompt: str,
    model: str,
    static_cache_hit: bool,
    **kwargs: Any,
) -> dict:
    """
    Build an Anthropic messages payload.
    Injects cache_control on the system prompt when a static-prefix hit occurred.
    """
    system: Any = system_prompt
    if static_cache_hit and system_prompt:
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    payload: dict = {
        "model":    model,
        "system":   system,
        "messages": messages,
        "max_tokens": kwargs.pop("max_tokens", 4096),
        **kwargs,
    }
    return payload


# ---------------------------------------------------------------------------
# Async HTTP call — non-blocking, uses httpx.AsyncClient
# Falls back to a thread-pool executor for the urllib path so the event loop
# is never blocked regardless of which HTTP backend is available.
# ---------------------------------------------------------------------------

async def _http_post_async(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    """Async HTTP POST — never blocks the event loop."""
    try:
        import httpx  # type: ignore
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
    except ImportError:
        pass

    # urllib fallback — run in a thread so we don't block the event loop
    import urllib.request, urllib.error

    def _sync_post() -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={**headers, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_post)


# ---------------------------------------------------------------------------
# Core dispatch with retry + fallback  (async)
# ---------------------------------------------------------------------------

async def dispatch(
    *,
    system_prompt: str,
    history: list[dict],
    user_message: str,
    model: str,
    static_cache_hit: bool = False,
    summary: str = "",
    extra_params: Optional[dict] = None,
    forwarded_api_key: Optional[str] = None,   # key forwarded from the calling app
) -> dict:
    """
    Send the optimised payload to the LLM and return a normalised response dict.

    Returns:
        {
          "id":            <str>,
          "model":         <str>,
          "content":       <str — assistant reply>,
          "usage":         {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
          "latency_ms":    <float>,
          "cache_hit":     <bool>,
          "fallback_used": <bool>,
          "dry_run":       <bool>,
        }
    """
    if _DRY_RUN:
        return _dry_run_response(model, system_prompt, history, user_message)

    # Resolve API key: prefer forwarded key from calling app, fall back to env
    _key_env = "ANTHROPIC_API_KEY" if _is_anthropic(model) else "OPENAI_API_KEY"
    resolved_key = forwarded_api_key or os.environ.get(_key_env, "").strip()
    if not resolved_key:
        return _dry_run_response(
            model, system_prompt, history, user_message,
            note="[No API key provided — pass Authorization: Bearer sk-... in your request, or set "
                 + _key_env + " env var]",
        )

    params = extra_params or {}
    fallback_used = False

    for attempt in range(_MAX_RETRIES + 1):
        current_model = model if attempt < _MAX_RETRIES else _FALLBACK_MODEL
        if current_model != model:
            fallback_used = True

        try:
            return await _single_dispatch_async(
                system_prompt=system_prompt,
                history=history,
                user_message=user_message,
                model=current_model,
                static_cache_hit=static_cache_hit,
                summary=summary,
                params=params,
                fallback_used=fallback_used,
                resolved_key=resolved_key,
            )
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower()
            is_server_err = any(c in err_str for c in ("500", "502", "503", "504"))

            if attempt < _MAX_RETRIES and (is_rate_limit or is_server_err):
                backoff = min(2 ** attempt * 0.5, 8.0)
                await asyncio.sleep(backoff)   # non-blocking sleep
                continue

            # Last attempt failed — return error response
            return {
                "id":            "err",
                "model":         current_model,
                "content":       f"[dispatcher error] {exc}",
                "usage":         {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_ms":    0.0,
                "cache_hit":     False,
                "fallback_used": fallback_used,
                "dry_run":       False,
                "error":         str(exc),
            }

    # Should not reach here
    return _dry_run_response(model, system_prompt, history, user_message)


async def _single_dispatch_async(
    *,
    system_prompt: str,
    history: list[dict],
    user_message: str,
    model: str,
    static_cache_hit: bool,
    summary: str,
    params: dict,
    fallback_used: bool,
    resolved_key: str = "",
) -> dict:
    t0 = time.perf_counter()

    # Build messages list for the API
    messages: list[dict] = []
    if summary:
        messages.append({"role": "user", "content": f"[Context summary]\n{summary}"})
        messages.append({"role": "assistant", "content": "Understood."})
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    if _is_anthropic(model):
        api_key = resolved_key or os.environ.get("ANTHROPIC_API_KEY", "")
        headers = {
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
            "anthropic-beta":    "prompt-caching-2024-07-31",
        }
        payload = _build_anthropic_payload(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            static_cache_hit=static_cache_hit,
            **params,
        )
        url = f"{_anthropic_url()}/messages"
        raw = await _http_post_async(url, headers, payload, _TIMEOUT_S)

        content = ""
        if isinstance(raw.get("content"), list):
            content = " ".join(b.get("text", "") for b in raw["content"] if b.get("type") == "text")
        usage_raw = raw.get("usage", {})
        usage = {
            "prompt_tokens":     usage_raw.get("input_tokens", 0),
            "completion_tokens": usage_raw.get("output_tokens", 0),
            "total_tokens":      usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
        }
    else:
        api_key = resolved_key or os.environ.get("OPENAI_API_KEY", "")
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        payload = _build_openai_payload(
            messages=messages,
            model=model,
            static_cache_hit=static_cache_hit,
            **params,
        )
        # Remove internal hint before sending
        payload.pop("_cache_hint", None)
        url = f"{_openai_url()}/chat/completions"
        raw = await _http_post_async(url, headers, payload, _TIMEOUT_S)

        choices = raw.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""
        usage_raw = raw.get("usage", {})
        usage = {
            "prompt_tokens":     usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "total_tokens":      usage_raw.get("total_tokens", 0),
        }

    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "id":            raw.get("id", ""),
        "model":         model,
        "content":       content,
        "usage":         usage,
        "latency_ms":    round(latency_ms, 2),
        "cache_hit":     static_cache_hit,
        "fallback_used": fallback_used,
        "dry_run":       False,
    }


def _dry_run_response(
    model: str, system_prompt: str, history: list[dict], user_message: str,
    note: str = "[DRY RUN] No LLM call made. Set OPENAI_API_KEY or pass Authorization: Bearer sk-... in your request.",
) -> dict:
    """Return a placeholder response for dry-run / testing mode."""
    approx_in = len(system_prompt.split()) + sum(len(m.get("content","").split()) for m in history) + len(user_message.split())
    return {
        "id":            "dry-run-0",
        "model":         model,
        "content":       note,
        "usage":         {"prompt_tokens": approx_in, "completion_tokens": 0, "total_tokens": approx_in},
        "latency_ms":    0.0,
        "cache_hit":     False,
        "fallback_used": False,
        "dry_run":       True,
    }
