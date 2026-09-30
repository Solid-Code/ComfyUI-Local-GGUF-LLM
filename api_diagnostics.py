"""Pure diagnostics for OpenAI-compatible request/output correlation.

This module intentionally has no ComfyUI/llama.cpp imports so repeated-output
and request-signature behavior can be unit tested in isolation.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import threading
from typing import Any


def stable_text_hash(text: Any, *, length: int = 16) -> str:
    data = str(text or "").encode("utf-8", errors="replace")
    return hashlib.sha256(data).hexdigest()[: max(8, int(length))]


def openai_request_signature(body: dict | None) -> str:
    """Hash an OpenAI request while ignoring fields that should not change output.

    Seed is deliberately excluded so calls with the same prompt/settings but
    different random seeds can be compared. Streaming changes transport only, so
    it is excluded as well. The hash contains no prompt text and is log-safe.
    """
    source = dict(body or {}) if isinstance(body, dict) else {}
    source.pop("seed", None)
    source.pop("stream", None)
    payload = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


class OpenAIRepeatTracker:
    """Bounded process-local detector for identical visible responses.

    It does not treat a repeated response as an error: stochastic decoding can
    legitimately converge. It only surfaces the evidence needed to distinguish
    seed reuse from same-output-under-different-seeds.
    """

    def __init__(self, limit: int = 256) -> None:
        self.limit = max(8, int(limit))
        self._items: deque[dict] = deque()
        self._lock = threading.Lock()

    def observe(
        self,
        *,
        request_signature: str,
        request_id: int,
        sampler_seed: int | None,
        response: str,
        thinking: str = "",
    ) -> dict:
        visible_hash = stable_text_hash(response)
        thinking_hash = stable_text_hash(thinking) if thinking else ""
        record = {
            "request_signature": str(request_signature or ""),
            "request_id": int(request_id),
            "sampler_seed": None if sampler_seed is None else int(sampler_seed),
            "visible_hash": visible_hash,
            "thinking_hash": thinking_hash,
            "visible_chars": len(str(response or "")),
            "thinking_chars": len(str(thinking or "")),
        }
        match = None
        with self._lock:
            for old in reversed(self._items):
                if old["request_signature"] != record["request_signature"]:
                    continue
                if old["visible_hash"] != visible_hash:
                    continue
                if old["sampler_seed"] == record["sampler_seed"]:
                    continue
                match = dict(old)
                break
            self._items.append(record)
            while len(self._items) > self.limit:
                self._items.popleft()
        result = dict(record)
        result["matched_previous"] = match
        result["thinking_same_as_previous"] = (
            None if match is None else match.get("thinking_hash", "") == thinking_hash
        )
        return result
