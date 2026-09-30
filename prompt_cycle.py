"""Prompt Enhancer cycle state machine.

This module intentionally has no ComfyUI or Local LLM dependencies.  Keeping the
cycle cursor and shuffle semantics here makes the behavior deterministic to test
without importing the much larger Prompt Enhancer runtime.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

MAX_CYCLE_REVISION = 0x7FFFFFFF
VALID_CYCLE_MODES = frozenset({"fixed", "increment", "decrement", "shuffle", "random"})


def normalize_history_index(value: Any, count: int) -> int:
    """Clamp a serialized history index to the available prompt array."""
    if count <= 0:
        return 0
    try:
        index = int(value)
    except (TypeError, ValueError, OverflowError):
        index = 0
    return max(0, min(index, count - 1))


def normalize_cycle_revision(value: Any) -> int:
    """Normalize legacy/null cycle revisions into the supported integer range."""
    try:
        revision = int(value)
    except (TypeError, ValueError, OverflowError):
        revision = 0
    return max(0, min(revision, MAX_CYCLE_REVISION))


def normalize_cycle_mode(value: Any) -> str:
    mode = str(value or "fixed").strip().lower()
    return mode if mode in VALID_CYCLE_MODES else "fixed"


def parse_shuffle_state(value: Any, count: int, current_index: int) -> list[int]:
    """Return a de-duplicated, in-range shuffle bag excluding the active card."""
    raw: Any
    if isinstance(value, str):
        try:
            raw = json.loads(value or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = []
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raw = []

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []

    seen: set[int] = set()
    result: list[int] = []
    for item in raw:
        try:
            index = int(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= index < count and index != current_index and index not in seen:
            seen.add(index)
            result.append(index)
    return result


def prompt_cycle_signature(mode: Any, history: Sequence[Any], revision: Any = 0) -> tuple[str, tuple[str, ...], int]:
    return (
        normalize_cycle_mode(mode),
        tuple(str(item) for item in history),
        normalize_cycle_revision(revision),
    )


def next_prompt_index(
    mode: Any,
    count: int,
    current_index: int,
    shuffle_state: Any,
    *,
    rng: random.Random | random.SystemRandom | None = None,
) -> tuple[int, list[int]]:
    """Compute the next cursor for a stateless cycle invocation.

    Stateful queue-safe Shuffle behavior lives in :class:`PromptCycleStore`;
    this helper is also used for the empty-key fallback where no backend cursor
    may safely be shared.
    """
    current = normalize_history_index(current_index, count)
    if count <= 1:
        return current if count else 0, []

    normalized_mode = normalize_cycle_mode(mode)
    if normalized_mode == "increment":
        return (current + 1) % count, []
    if normalized_mode == "decrement":
        return (current - 1) % count, []

    if normalized_mode == "random":
        random_source = rng or random.SystemRandom()
        return random_source.randrange(count), []

    if normalized_mode == "shuffle":
        random_source = rng or random.SystemRandom()
        bag = parse_shuffle_state(shuffle_state, count, current)
        if not bag:
            bag = [index for index in range(count) if index != current]
            random_source.shuffle(bag)
        next_index = bag.pop(0) if bag else current
        return next_index, bag

    return current, parse_shuffle_state(shuffle_state, count, current)


@dataclass(slots=True)
class _CycleState:
    signature: tuple[str, tuple[str, ...], int]
    next_index: int
    shuffle: list[int]
    updated_at: float


class PromptCycleStore:
    """Workflow-isolated, queue-safe Prompt Cycle cursor store.

    The serialized X/Y index anchors a new cycle signature. Once that signature
    has been seen, the backend cursor is authoritative so multiple workflow items
    queued before earlier executions finish still consume different prompts. A
    history/mode/revision change creates a new signature and re-anchors the cursor
    to the user-visible X/Y index.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 24 * 60 * 60.0,
        clock: Callable[[], float] = time.monotonic,
        rng_factory: Callable[[], random.Random | random.SystemRandom] = random.SystemRandom,
    ) -> None:
        self._ttl_seconds = max(0.0, float(ttl_seconds))
        self._clock = clock
        self._rng_factory = rng_factory
        self._lock = threading.Lock()
        self._states: dict[str, _CycleState] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)

    def clear(self, unique_id: Any) -> None:
        key = str(unique_id or "").strip()
        if not key:
            return
        with self._lock:
            self._states.pop(key, None)

    def _prune_locked(self, now: float) -> None:
        stale = [
            key
            for key, state in self._states.items()
            if now - state.updated_at > self._ttl_seconds
        ]
        for key in stale:
            self._states.pop(key, None)

    def snapshot(
        self,
        unique_id: Any,
        mode: Any,
        history: Sequence[Any],
        revision: Any = 0,
    ) -> dict[str, Any]:
        """Read the live next cursor without advancing it."""
        key = str(unique_id or "").strip()
        clean_history = [str(item) for item in history]
        count = len(clean_history)
        if not key or count <= 0 or normalize_cycle_mode(mode) == "fixed":
            return {"valid": False}

        signature = prompt_cycle_signature(mode, clean_history, revision)
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            state = self._states.get(key)
            if not state or state.signature != signature:
                return {"valid": False}
            next_index = normalize_history_index(state.next_index, count)
            shuffle = parse_shuffle_state(state.shuffle, count, next_index)
            return {"valid": True, "next_index": next_index, "shuffle": shuffle}

    def advance(
        self,
        unique_id: Any,
        mode: Any,
        history: Sequence[Any],
        requested_index: Any,
        shuffle_state: Any,
        revision: Any = 0,
    ) -> tuple[int, int, list[int]]:
        """Return ``(index_used_now, next_index, next_shuffle_bag)``."""
        clean_history = [str(item) for item in history]
        count = len(clean_history)
        requested = normalize_history_index(requested_index, count)
        normalized_mode = normalize_cycle_mode(mode)
        key = str(unique_id or "").strip()

        if count <= 0:
            self.clear(key)
            return 0, 0, []

        # Fixed must never inherit a live backend cursor.  Keep this guarantee in
        # the store itself even though Prompt Enhancer also short-circuits Fixed.
        if normalized_mode == "fixed":
            self.clear(key)
            return requested, requested, []

        # Missing workflow ownership deliberately stays stateless rather than
        # risking cross-workflow cursor leakage.
        if not key:
            next_index, next_shuffle = next_prompt_index(
                normalized_mode,
                count,
                requested,
                shuffle_state,
                rng=self._rng_factory(),
            )
            return requested, next_index, next_shuffle

        signature = prompt_cycle_signature(normalized_mode, clean_history, revision)
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            state = self._states.get(key)
            state_matches = bool(state and state.signature == signature)
            if state_matches:
                current_index = normalize_history_index(state.next_index, count)
                current_shuffle: Any = state.shuffle
            else:
                current_index = requested
                current_shuffle = shuffle_state

            if normalized_mode == "shuffle" and count > 1:
                bag = parse_shuffle_state(current_shuffle, count, current_index)
                if not bag:
                    rng = self._rng_factory()
                    if state_matches:
                        # New deck: include every card exactly once.  Avoid only
                        # an immediate repeat at the old/new deck boundary.
                        bag = list(range(count))
                        rng.shuffle(bag)
                        if bag and bag[0] == current_index:
                            swap_index = next(
                                (i for i, value in enumerate(bag[1:], 1) if value != current_index),
                                None,
                            )
                            if swap_index is not None:
                                bag[0], bag[swap_index] = bag[swap_index], bag[0]
                    else:
                        # The user-selected X/Y entry is the first card in the
                        # initial deck, so queue each other prompt once.
                        bag = [index for index in range(count) if index != current_index]
                        rng.shuffle(bag)
                next_index = bag.pop(0) if bag else current_index
                next_shuffle = bag
            else:
                next_index, next_shuffle = next_prompt_index(
                    normalized_mode,
                    count,
                    current_index,
                    current_shuffle,
                    rng=self._rng_factory(),
                )

            self._states[key] = _CycleState(
                signature=signature,
                next_index=next_index,
                shuffle=list(next_shuffle),
                updated_at=now,
            )

        return current_index, next_index, list(next_shuffle)


PROMPT_CYCLE_STORE = PromptCycleStore()
