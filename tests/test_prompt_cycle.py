from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_cycle import (  # noqa: E402
    MAX_CYCLE_REVISION,
    PromptCycleStore,
    next_prompt_index,
    normalize_cycle_mode,
    normalize_cycle_revision,
    normalize_history_index,
    parse_shuffle_state,
    prompt_cycle_signature,
)


class Clock:
    def __init__(self, value=100.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


def seeded_factory(seed=12345):
    return lambda: random.Random(seed)


class PromptCycleTests(unittest.TestCase):
    def test_normalize_history_index(self):
        cases = [
            (None, 0, 0),
            (None, 3, 0),
            (-3, 3, 0),
            (99, 3, 2),
            ("2", 4, 2),
            ("bad", 4, 0),
        ]
        for value, count, expected in cases:
            with self.subTest(value=value, count=count):
                self.assertEqual(normalize_history_index(value, count), expected)

    def test_normalize_cycle_revision(self):
        cases = [
            (None, 0),
            ("", 0),
            ("bad", 0),
            (-1, 0),
            ("7", 7),
            (MAX_CYCLE_REVISION + 999, MAX_CYCLE_REVISION),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(normalize_cycle_revision(value), expected)

    def test_mode_and_signature_normalization(self):
        self.assertEqual(normalize_cycle_mode(" INCREMENT "), "increment")
        self.assertEqual(normalize_cycle_mode("not-a-mode"), "fixed")
        self.assertEqual(
            prompt_cycle_signature(" Increment ", [1, "b"], "9"),
            ("increment", ("1", "b"), 9),
        )

    def test_parse_shuffle_state_accepts_json_or_native_sequence_and_filters_bad_entries(self):
        expected = [0, 3]
        self.assertEqual(parse_shuffle_state('[0, 2, 3, 3, 99, -1, "bad"]', 4, 2), expected)
        self.assertEqual(parse_shuffle_state([0, 2, 3, 3, 99, -1, "bad"], 4, 2), expected)
        self.assertEqual(parse_shuffle_state({"not": "a list"}, 4, 2), [])

    def test_fixed_is_stateless_and_does_not_move(self):
        store = PromptCycleStore(rng_factory=seeded_factory())
        history = ["A", "B", "C"]
        self.assertEqual(store.advance("wf\x1fnode", "increment", history, 0, [], 1)[:2], (0, 1))
        self.assertEqual(len(store), 1)

        for _ in range(5):
            self.assertEqual(store.advance("wf\x1fnode", "fixed", history, 2, [], 1), (2, 2, []))
            self.assertEqual(store.snapshot("wf\x1fnode", "fixed", history, 1), {"valid": False})
            self.assertEqual(len(store), 0)

    def test_increment_rapid_queue_uses_backend_cursor_not_repeated_serialized_index(self):
        store = PromptCycleStore()
        history = ["A", "B", "C", "D"]
        used = [store.advance("wf\x1fnode", "increment", history, 0, [], 4)[0] for _ in range(10)]
        self.assertEqual(used, [0, 1, 2, 3, 0, 1, 2, 3, 0, 1])

    def test_decrement_wraps_across_rapid_queue(self):
        store = PromptCycleStore()
        history = ["A", "B", "C", "D"]
        used = [store.advance("wf\x1fnode", "decrement", history, 0, [], 4)[0] for _ in range(6)]
        self.assertEqual(used, [0, 3, 2, 1, 0, 3])

    def test_revision_change_makes_serialized_xy_authoritative_again(self):
        store = PromptCycleStore()
        history = ["A", "B", "C"]
        self.assertEqual(store.advance("wf\x1fnode", "increment", history, 0, [], 1)[0], 0)
        self.assertEqual(store.advance("wf\x1fnode", "increment", history, 0, [], 1)[0], 1)
        self.assertEqual(store.advance("wf\x1fnode", "increment", history, 2, [], 2)[0], 2)
        self.assertEqual(store.advance("wf\x1fnode", "increment", history, 2, [], 2)[0], 0)

    def test_history_change_resets_cursor_even_without_revision_change(self):
        store = PromptCycleStore()
        self.assertEqual(store.advance("wf\x1fnode", "increment", ["A", "B", "C"], 0, [], 1)[0], 0)
        self.assertEqual(store.advance("wf\x1fnode", "increment", ["A", "B", "C"], 0, [], 1)[0], 1)
        self.assertEqual(store.advance("wf\x1fnode", "increment", ["A", "B", "C", "D"], 3, [], 1)[0], 3)

    def test_mode_change_resets_cursor(self):
        store = PromptCycleStore()
        history = ["A", "B", "C"]
        store.advance("wf\x1fnode", "increment", history, 0, [], 1)
        store.advance("wf\x1fnode", "increment", history, 0, [], 1)
        self.assertEqual(store.advance("wf\x1fnode", "decrement", history, 2, [], 1)[0], 2)

    def test_snapshot_requires_exact_key_mode_history_and_revision(self):
        store = PromptCycleStore()
        history = ["A", "B", "C"]
        _, next_index, shuffle = store.advance("wf\x1fnode", "increment", history, 0, [], 8)
        self.assertEqual(
            store.snapshot("wf\x1fnode", "increment", history, 8),
            {"valid": True, "next_index": next_index, "shuffle": shuffle},
        )
        self.assertEqual(store.snapshot("other\x1fnode", "increment", history, 8), {"valid": False})
        self.assertEqual(store.snapshot("wf\x1fnode", "decrement", history, 8), {"valid": False})
        self.assertEqual(store.snapshot("wf\x1fnode", "increment", history + ["D"], 8), {"valid": False})
        self.assertEqual(store.snapshot("wf\x1fnode", "increment", history, 9), {"valid": False})

    def test_workflow_keys_are_isolated_even_with_same_graph_local_node_id(self):
        store = PromptCycleStore()
        history = ["A", "B", "C"]
        key_a = "workflow-a\x1fnode-12"
        key_b = "workflow-b\x1fnode-12"
        self.assertEqual(store.advance(key_a, "increment", history, 0, [], 1)[0], 0)
        self.assertEqual(store.advance(key_a, "increment", history, 0, [], 1)[0], 1)
        self.assertEqual(store.advance(key_b, "increment", history, 2, [], 1)[0], 2)
        self.assertEqual(store.advance(key_a, "increment", history, 0, [], 1)[0], 2)
        self.assertEqual(store.advance(key_b, "increment", history, 2, [], 1)[0], 0)

    def test_missing_owner_key_never_creates_shared_state(self):
        store = PromptCycleStore(rng_factory=seeded_factory())
        history = ["A", "B", "C"]
        first = store.advance("", "increment", history, 1, [], 1)
        second = store.advance("", "increment", history, 1, [], 1)
        self.assertEqual(first, (1, 2, []))
        self.assertEqual(second, (1, 2, []))
        self.assertEqual(len(store), 0)

    def test_shuffle_produces_complete_decks_and_no_boundary_repeat(self):
        store = PromptCycleStore(rng_factory=seeded_factory(7))
        history = ["A", "B", "C", "D"]
        used = [store.advance("wf\x1fnode", "shuffle", history, 0, [], 11)[0] for _ in range(16)]
        for offset in range(0, len(used), len(history)):
            self.assertEqual(set(used[offset : offset + len(history)]), set(range(len(history))))
        for boundary in range(len(history), len(used), len(history)):
            self.assertNotEqual(used[boundary - 1], used[boundary])

    def test_random_is_in_range_and_keeps_backend_cursor_between_queued_runs(self):
        store = PromptCycleStore(rng_factory=seeded_factory(99))
        history = ["A", "B", "C", "D", "E"]
        used = [store.advance("wf\x1fnode", "random", history, 3, [], 2)[0] for _ in range(20)]
        self.assertEqual(used[0], 3)
        self.assertTrue(all(0 <= index < len(history) for index in used))

    def test_ttl_prunes_abandoned_cycle_states(self):
        clock = Clock()
        store = PromptCycleStore(ttl_seconds=10, clock=clock)
        history = ["A", "B"]
        store.advance("wf-a\x1fnode", "increment", history, 0, [], 1)
        self.assertEqual(len(store), 1)
        clock.advance(11)
        self.assertEqual(store.snapshot("wf-a\x1fnode", "increment", history, 1), {"valid": False})
        self.assertEqual(len(store), 0)

    def test_concurrent_increment_advances_are_serialized_safely(self):
        store = PromptCycleStore()
        history = ["A", "B", "C", "D"]

        def run_one(_):
            return store.advance("wf\x1fnode", "increment", history, 0, [], 3)[0]

        with ThreadPoolExecutor(max_workers=12) as executor:
            used = list(executor.map(run_one, range(80)))
        self.assertEqual(Counter(used), Counter({0: 20, 1: 20, 2: 20, 3: 20}))

    def test_stateless_next_helper_wraps_and_fixed_keeps_current(self):
        self.assertEqual(next_prompt_index("increment", 3, 2, []), (0, []))
        self.assertEqual(next_prompt_index("decrement", 3, 0, []), (2, []))
        self.assertEqual(next_prompt_index("fixed", 3, 1, [0, 2]), (1, [0, 2]))


if __name__ == "__main__":
    unittest.main()
