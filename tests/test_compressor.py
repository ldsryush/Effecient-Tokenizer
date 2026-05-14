"""
Unit tests for app.compressor — verifies that key shortening is applied ONLY
inside JSON blobs and never to free-form prose.

Run with:
    pytest tests/test_compressor.py -v
or:
    python -m unittest tests.test_compressor
"""
from __future__ import annotations
import json
import os
import sys
import unittest

# Make project root importable when run directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.compressor import (  # noqa: E402
    compress,
    _find_json_spans,
    _shorten_keys_in_obj,
    _process_json_blobs,
)


class TestPureJSON(unittest.TestCase):
    """JSON-only input: keys must be shortened, structure preserved."""

    def test_object_keys_shortened(self) -> None:
        src = json.dumps({
            "configuration": {"environment": "prod", "temperature": 0.7},
            "messages": [{"content": "hi", "description": "greeting"}],
        }, indent=2)
        out = compress(src)["text"]
        parsed = json.loads(out)
        # Keys renamed
        self.assertIn("cfg", parsed)
        self.assertIn("env", parsed["cfg"])
        self.assertIn("temp", parsed["cfg"])
        self.assertIn("msgs", parsed)
        self.assertIn("cont", parsed["msgs"][0])
        self.assertIn("desc", parsed["msgs"][0])
        # Values are completely unchanged
        self.assertEqual(parsed["cfg"]["env"], "prod")
        self.assertEqual(parsed["msgs"][0]["cont"], "hi")
        self.assertEqual(parsed["msgs"][0]["desc"], "greeting")

    def test_array_input(self) -> None:
        src = json.dumps([{"message": "hi"}, {"message": "bye"}])
        out = compress(src)["text"]
        parsed = json.loads(out)
        self.assertEqual(parsed, [{"msg": "hi"}, {"msg": "bye"}])

    def test_minification_removes_whitespace(self) -> None:
        src = '{\n  "configuration": {\n    "environment": "prod"\n  }\n}'
        out = compress(src)["text"]
        # JSON is now minified
        self.assertEqual(out, '{"cfg":{"env":"prod"}}')

    def test_unknown_keys_unchanged(self) -> None:
        src = json.dumps({"foo": "bar", "configuration": {"baz": 1}})
        out = compress(src)["text"]
        parsed = json.loads(out)
        self.assertIn("foo", parsed)        # unknown key kept
        self.assertIn("cfg", parsed)        # known key renamed
        self.assertEqual(parsed["foo"], "bar")
        self.assertEqual(parsed["cfg"]["baz"], 1)

    def test_nested_objects(self) -> None:
        src = json.dumps({
            "request": {
                "parameters": {
                    "authorization": "Bearer xyz",
                    "configuration": {"environment": "staging"},
                }
            }
        })
        out = compress(src)["text"]
        parsed = json.loads(out)
        self.assertEqual(
            parsed,
            {"req": {"params": {"auth": "Bearer xyz", "cfg": {"env": "staging"}}}},
        )


class TestPureProse(unittest.TestCase):
    """Prose-only input: NO key shortening should ever fire."""

    def test_words_in_prose_are_preserved(self) -> None:
        src = (
            "Please return the content as a message. "
            "Include a description of the function and its parameters. "
            "Use the configuration from the environment."
        )
        out = compress(src)["text"]
        # All of these prose words must survive verbatim
        for word in [
            "content", "message", "description", "function",
            "parameters", "configuration", "environment",
        ]:
            self.assertIn(word, out, f"Prose word '{word}' was corrupted: {out!r}")

    def test_no_json_spans_detected(self) -> None:
        src = "The function f(x) computes the message content for each request."
        spans = _find_json_spans(src)
        self.assertEqual(spans, [])

    def test_brace_like_prose_unchanged(self) -> None:
        # Python set notation or template literal — looks like braces but isn't JSON
        src = "Use the set {1, 2, 3} and the template `Hello, {name}!`."
        out = compress(src)["text"]
        self.assertIn("{1, 2, 3}", out)
        self.assertIn("{name}", out)
        self.assertIn("message" if "message" in src else "Hello", out)

    def test_punctuation_normalized(self) -> None:
        src = "Wait!!! Really??? Hmm...."
        out = compress(src)["text"]
        self.assertEqual(out, "Wait! Really? Hmm...")

    def test_whitespace_normalized(self) -> None:
        src = "line 1\n\n\n\nline 2    with    spaces"
        out = compress(src)["text"]
        self.assertEqual(out, "line 1\n\nline 2 with spaces")

    def test_system_prompt_not_corrupted(self) -> None:
        """The exact scenario the bug report describes."""
        src = (
            "You are a helpful assistant. When the user asks a question, "
            "return the content as a message with a clear description "
            "of the function being called and its parameters."
        )
        out = compress(src)["text"]
        # None of the alias substitutions may have leaked
        for short in ["cont", "msg", "desc", "fn", "params"]:
            self.assertNotIn(
                f" {short} ", f" {out} ",
                f"Prose was corrupted with alias '{short}': {out!r}",
            )
        # And the original long words survive
        for long in ["content", "message", "description", "function", "parameters"]:
            self.assertIn(long, out)


class TestMixedInput(unittest.TestCase):
    """Mixed prose + JSON: only the JSON portion is rewritten."""

    def test_prose_around_json(self) -> None:
        src = (
            "Here is the configuration we will use for the message format. "
            'The payload looks like this: {"configuration": {"environment": "prod"}, '
            '"messages": [{"content": "hello", "description": "greet"}]} '
            "Please ensure the description field is filled in."
        )
        out = compress(src)["text"]

        # Prose words BEFORE and AFTER the JSON must survive
        self.assertIn("Here is the configuration we will use", out)
        self.assertIn("for the message format", out)
        self.assertIn("Please ensure the description field is filled in", out)

        # The JSON blob must be minified and have shortened keys
        self.assertIn('{"cfg":{"env":"prod"},"msgs":[{"cont":"hello","desc":"greet"}]}', out)

    def test_multiple_json_blobs(self) -> None:
        src = (
            'First request: {"message": "a"} then we '
            'send a second one: {"message": "b"}. End of description.'
        )
        out = compress(src)["text"]
        self.assertIn('{"msg":"a"}', out)
        self.assertIn('{"msg":"b"}', out)
        # Prose intact
        self.assertIn("First request:", out)
        self.assertIn("then we send a second one:", out)
        self.assertIn("End of description.", out)

    def test_json_value_strings_untouched(self) -> None:
        # Values inside JSON that happen to contain trigger words must NOT
        # be shortened — only keys are renamed.
        src = '{"message": "please return the content as a message"}'
        out = compress(src)["text"]
        parsed = json.loads(out)
        self.assertEqual(parsed["msg"], "please return the content as a message")

    def test_invalid_json_left_alone(self) -> None:
        # Looks like JSON but isn't valid — must be left as-is, including
        # the word "message" inside the prose.
        src = "Not JSON: {message: hi, content: bye} — the message stays."
        out = compress(src)["text"]
        # Original "message" preserved in prose
        self.assertIn("message", out)
        self.assertIn("content", out)

    def test_compress_returns_metadata(self) -> None:
        src = '{"configuration": {"environment": "prod"}}'
        result = compress(src)
        self.assertEqual(result["confidence"], 1.0)
        self.assertEqual(result["mode"], "lossless")
        self.assertEqual(result["stage"], "structural")
        self.assertGreaterEqual(result["chars_saved"], 0)


class TestHelpers(unittest.TestCase):
    """Direct tests of the internal helpers."""

    def test_shorten_keys_in_obj_recursive(self) -> None:
        obj = {"configuration": {"messages": [{"content": "x"}]}}
        out = _shorten_keys_in_obj(obj)
        self.assertEqual(out, {"cfg": {"msgs": [{"cont": "x"}]}})

    def test_find_spans_handles_nested(self) -> None:
        src = 'before {"a": {"b": [1, 2, {"c": 3}]}} after'
        spans = _find_json_spans(src)
        self.assertEqual(len(spans), 1)
        s, e = spans[0]
        # The full nested object should be one span
        self.assertEqual(src[s:e], '{"a": {"b": [1, 2, {"c": 3}]}}')

    def test_find_spans_handles_braces_in_strings(self) -> None:
        src = 'prefix {"text": "value with {curly} braces"} suffix'
        spans = _find_json_spans(src)
        self.assertEqual(len(spans), 1)
        s, e = spans[0]
        self.assertEqual(src[s:e], '{"text": "value with {curly} braces"}')

    def test_process_json_blobs_no_change_when_no_json(self) -> None:
        src = "Just a sentence about messages and content."
        self.assertEqual(_process_json_blobs(src), src)


if __name__ == "__main__":
    unittest.main()
