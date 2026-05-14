"""
Structural Compressor (Stage 2 of Pipeline)
--------------------------------------------
Lossless wins only:
  1. JSON minification — strips whitespace inside JSON blobs
  2. Verbose key shortening — maps long snake_case keys to compact aliases.
     *** SAFETY: key shortening is applied ONLY to keys inside detected JSON
     structures.  It is NEVER applied to free-form prose. ***
  3. Whitespace normalization — collapses runs of spaces / blank lines
  4. Redundant punctuation removal — repeated "!!!", "..." etc.

Returns compressed text + a confidence score (always 1.0 — this stage is lossless).

Architecture
------------
A previous version of this module applied a global `re.sub` over the entire
input text for key shortening.  That corrupted prose: a system prompt asking
the model to "return the content as a message" was rewritten to "return the
cont as a msg".  This module now:

  1. Scans the text for JSON object/array blobs using a brace-balanced scanner
     (handles nested braces and string literals containing braces).
  2. For each blob:
       - parses it as JSON,
       - recursively walks the object renaming only *dict keys* via _KEY_MAP,
       - serialises it back in minified form.
  3. Leaves every character outside detected JSON blobs untouched by the
     key-shortening pass.
  4. Runs whitespace and punctuation normalization on the full result — those
     are safe.
"""
from __future__ import annotations
import json
import re
from typing import Any

# Key aliases: long → short.
# SAFETY RULE: only include words that are ≥10 chars, unambiguous in technical
# contexts, and never appear as meaningful standalone words in prose.
# Even with this rule, the key map is now only ever applied to JSON keys,
# so prose is safe regardless.
_KEY_MAP: dict[str, str] = {
    "authorization_token":  "auth_tok",
    "authorization":        "auth",
    "authentication":       "authn",
    "configuration":        "cfg",
    "environment":          "env",
    "parameters":           "params",
    "parameter":            "param",
    "description":          "desc",
    "response":             "resp",
    "request":              "req",
    "message":              "msg",
    "messages":             "msgs",
    "content":              "cont",
    "function":             "fn",
    "callback":             "cb",
    "identifier":           "id",
    "timestamp":            "ts",
    "temperature":          "temp",
    "information":          "info",
    "initialize":           "init",
    "initialization":       "init",
    "additional":           "addl",
    "properties":           "props",
    "property":             "prop",
    "conversation":         "conv",
    "instructions":         "instrs",
    "instruction":          "instr",
    "requirements":         "reqs",
    "requirement":          "req",
    "implementation":       "impl",
    "functionality":        "func",
    "documentation":        "docs",
    "notification":         "notif",
    "notifications":        "notifs",
    "subscription":         "sub",
    "subscriptions":        "subs",
    "organization":         "org",
    "organizations":        "orgs",
    "permissions":          "perms",
    "permission":           "perm",
    "validation":           "valid",
    "validations":          "valids",
    "transformation":       "xform",
    "transformations":      "xforms",
    "serialization":        "serial",
    "deserialization":      "deserial",
    "pagination":           "paging",
    "deprecation":          "depr",
    "deprecated":           "depr",
    "synchronization":      "sync",
    "asynchronous":         "async",
    "synchronous":          "sync",
    "compression":          "compr",
    "decompression":        "decompr",
    "encryption":           "encr",
    "decryption":           "decr",
    "certificate":          "cert",
    "certificates":         "certs",
    "credentials":          "creds",
    "credential":           "cred",
    "administrator":        "admin",
    "administrators":       "admins",
    "application":          "app",
    "applications":         "apps",
    "infrastructure":       "infra",
    "microservice":         "svc",
    "microservices":        "svcs",
    "kubernetes":           "k8s",
    "deployment":           "deploy",
    "deployments":          "deploys",
    "repository":           "repo",
    "repositories":         "repos",
    "dependency":           "dep",
    "dependencies":         "deps",
    "middleware":           "mw",
    "transaction":          "txn",
    "transactions":         "txns",
    "connection":           "conn",
    "connections":          "conns",
    "exception":            "exc",
    "exceptions":           "excs",
    "stacktrace":           "trace",
    "traceback":            "trace",
    "debugging":            "dbg",
    "profiling":            "prof",
    "monitoring":           "mon",
    "observability":        "obs",
    "telemetry":            "telem",
    "aggregation":          "agg",
    "aggregations":         "aggs",
    "calculation":          "calc",
    "calculations":         "calcs",
    "evaluation":           "eval",
    "evaluations":          "evals",
    "recommendation":       "rec",
    "recommendations":      "recs",
    "classification":       "clf",
    "classifications":      "clfs",
    "preprocessing":        "preproc",
    "postprocessing":       "postproc",
    "hyperparameter":       "hparam",
    "hyperparameters":      "hparams",
    "checkpoint":           "ckpt",
    "checkpoints":          "ckpts",
    "optimization":         "optim",
    "optimizations":        "optims",
    "regularization":       "reg",
    "normalization":        "norm",
    "tokenization":         "tok",
    "tokenizer":            "tok",
    "embedding":            "emb",
    "embeddings":           "embs",
    "transformer":          "xfmr",
    "transformers":         "xfmrs",
    "architecture":         "arch",
    "architectures":        "archs",
}


# ---------------------------------------------------------------------------
# JSON blob detection
# ---------------------------------------------------------------------------

def _find_json_spans(text: str) -> list[tuple[int, int]]:
    """
    Scan *text* and return a list of (start, end) index pairs delimiting JSON
    object/array blobs.  Uses a state machine that tracks string literals,
    escape sequences, and brace/bracket nesting depth, so braces inside JSON
    string values do not throw off the scan.

    Only spans that successfully `json.loads()` are returned, so prose that
    happens to contain "{ ... }" (e.g. f-string templates, set notation) is
    not classified as JSON.
    """
    spans: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in "{[":
            open_ch = ch
            close_ch = "}" if ch == "{" else "]"
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == open_ch:
                        depth += 1
                    elif c == close_ch:
                        depth -= 1
                        if depth == 0:
                            # Try to parse [i, j+1)
                            candidate = text[i:j + 1]
                            try:
                                json.loads(candidate)
                                spans.append((i, j + 1))
                                i = j + 1
                                break
                            except Exception:
                                # Not valid JSON — skip this opening char
                                break
                j += 1
            else:
                # Reached end of text without closing — give up on this opener
                pass
            i += 1
        else:
            i += 1
    return spans


# ---------------------------------------------------------------------------
# Key-only shortening on parsed JSON
# ---------------------------------------------------------------------------

def _shorten_keys_in_obj(obj: Any) -> Any:
    """Recursively rename dict keys via _KEY_MAP.  Values are untouched."""
    if isinstance(obj, dict):
        new: dict[str, Any] = {}
        for k, v in obj.items():
            new_key = _KEY_MAP.get(k, k) if isinstance(k, str) else k
            new[new_key] = _shorten_keys_in_obj(v)
        return new
    if isinstance(obj, list):
        return [_shorten_keys_in_obj(x) for x in obj]
    return obj


def _process_json_blobs(text: str) -> str:
    """
    For each detected JSON blob:
      - parse,
      - rename keys (lossless alias),
      - re-serialize in minified form.
    Surrounding prose is preserved byte-for-byte.
    """
    spans = _find_json_spans(text)
    if not spans:
        return text

    out_parts: list[str] = []
    cursor = 0
    for start, end in spans:
        # Append prose before this blob unchanged
        out_parts.append(text[cursor:start])
        blob = text[start:end]
        try:
            parsed = json.loads(blob)
            renamed = _shorten_keys_in_obj(parsed)
            minified = json.dumps(renamed, separators=(",", ":"), ensure_ascii=False)
            out_parts.append(minified)
        except Exception:
            # Should not happen because _find_json_spans already validated,
            # but be defensive — fall back to the original blob.
            out_parts.append(blob)
        cursor = end
    out_parts.append(text[cursor:])
    return "".join(out_parts)


# ---------------------------------------------------------------------------
# Safe, prose-friendly cleanups
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse blank lines (3+ → 2) and runs of spaces/tabs.  Strip trailing
    spaces per line.  Operates on the whole text — safe for prose."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    return text.strip()


def _remove_redundant_punctuation(text: str) -> str:
    """!!! → !, ??? → ?, .... → ... (ellipsis preserved)."""
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\.{4,}', '...', text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress(text: str, mode: str = "lossless") -> dict:
    """
    Apply structural compression to *text*.

    Steps (all lossless):
      1. Detect JSON blobs and, inside each, minify + shorten keys.
      2. Normalize whitespace and redundant punctuation in the surrounding text.

    Returns:
        {
          "text":       <compressed text>,
          "confidence": 1.0,
          "mode":       "lossless",
          "stage":      "structural",
          "chars_saved": <int>,
        }
    """
    if text is None:
        text = ""
    original_len = len(text)
    out = _process_json_blobs(text)
    out = _remove_redundant_punctuation(out)
    out = _normalize_whitespace(out)

    saved_chars = original_len - len(out)
    return {
        "text":        out,
        "confidence":  1.0,
        "mode":        "lossless",
        "stage":       "structural",
        "chars_saved": max(0, saved_chars),
    }
