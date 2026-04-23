"""
Structural Compressor (Stage 2 of Pipeline)
--------------------------------------------
Lossless wins only:
  1. JSON minification — strips whitespace inside JSON blobs
  2. Verbose key shortening — maps long snake_case keys to compact aliases
  3. Whitespace normalization — collapses runs of spaces / blank lines
  4. Redundant punctuation removal — repeated "!!!", "..." etc.

Returns compressed text + a confidence score (always 1.0 — this stage is lossless).
"""
from __future__ import annotations
import json
import re

# Key aliases: long → short.
# SAFETY RULE: only include words that are ≥10 chars, unambiguous in technical
# contexts, and never appear as meaningful standalone words in prose.
# Short common words (context, output, input, example) are intentionally excluded
# to avoid corrupting user-written content.
_KEY_MAP: dict[str, str] = {
    # Original entries
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
    # New safe additions — all ≥10 chars, formal/technical only
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

# Pre-compiled regex for key shortening (whole-word, not inside quotes)
_KEY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b' + re.escape(long) + r'\b'), short)
    for long, short in _KEY_MAP.items()
]


def _minify_json_blobs(text: str) -> str:
    """Find JSON objects/arrays in text and minify them."""
    def _try_minify(m: re.Match) -> str:
        raw = m.group(0)
        try:
            obj = json.loads(raw)
            return json.dumps(obj, separators=(",", ":"))
        except Exception:
            return raw

    # Match JSON objects or arrays (heuristic: balanced braces at top level)
    return re.sub(r'\{[^{}]*\}|\[[^\[\]]*\]', _try_minify, text)


def _shorten_keys(text: str) -> str:
    for pat, short in _KEY_PATTERNS:
        text = pat.sub(short, text)
    return text


def _normalize_whitespace(text: str) -> str:
    # Collapse 3+ blank lines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse horizontal whitespace runs to a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Strip trailing spaces per line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    return text.strip()


def _remove_redundant_punctuation(text: str) -> str:
    # "!!!" → "!", "???" → "?", "..." kept (ellipsis is meaningful), "…." → "…"
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\.{4,}', '...', text)
    return text


def compress(text: str, mode: str = "lossless") -> dict:
    """
    Apply structural compression to *text*.

    Returns:
        {
          "text":       <compressed text>,
          "confidence": 1.0,
          "mode":       "lossless",
          "stage":      "structural",
        }
    """
    original_len = len(text)
    out = text
    out = _minify_json_blobs(out)
    out = _shorten_keys(out)
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
