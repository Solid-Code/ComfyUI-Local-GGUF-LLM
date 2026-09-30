from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cpu_tuning  # noqa: E402


class CpuTuningTests(unittest.TestCase):
    def test_parse_cpu_list_supports_ranges_and_singletons(self):
        self.assertEqual(
            cpu_tuning._parse_cpu_list("0-3,8,10-11"),
            {0, 1, 2, 3, 8, 10, 11},
        )

    def test_parse_cpu_list_normalizes_reverse_ranges(self):
        self.assertEqual(cpu_tuning._parse_cpu_list("3-1"), {1, 2, 3})

    def test_3990x_auto_profile_uses_physical_cores_for_decode_and_logical_for_batch(self):
        logical_ids = set(range(128))
        with mock.patch.object(cpu_tuning, "available_cpu_ids", return_value=logical_ids), \
             mock.patch.object(cpu_tuning, "physical_core_count", return_value=64), \
             mock.patch.object(cpu_tuning, "numa_node_count", return_value=1):
            profile = cpu_tuning.cpu_runtime_profile(0, 0, "Auto", cpu_only=True)
        self.assertEqual(profile["logical_cpus"], 128)
        self.assertEqual(profile["physical_cores"], 64)
        self.assertEqual(profile["threads"], 64)
        self.assertEqual(profile["threads_batch"], 128)
        self.assertTrue(profile["threads_auto"])
        self.assertTrue(profile["threads_batch_auto"])
        self.assertEqual(profile["numa_effective"], "Disabled")

    def test_explicit_thread_counts_are_preserved(self):
        with mock.patch.object(cpu_tuning, "available_cpu_ids", return_value=set(range(32))), \
             mock.patch.object(cpu_tuning, "physical_core_count", return_value=16), \
             mock.patch.object(cpu_tuning, "numa_node_count", return_value=1):
            profile = cpu_tuning.resolve_cpu_threads(12, 24)
        self.assertEqual(profile["threads"], 12)
        self.assertEqual(profile["threads_batch"], 24)
        self.assertFalse(profile["threads_auto"])
        self.assertFalse(profile["threads_batch_auto"])

    def test_auto_respects_process_affinity(self):
        with mock.patch.object(cpu_tuning, "available_cpu_ids", return_value=set(range(24))), \
             mock.patch.object(cpu_tuning, "physical_core_count", return_value=12), \
             mock.patch.object(cpu_tuning, "numa_node_count", return_value=1):
            profile = cpu_tuning.resolve_cpu_threads(0, 0)
        self.assertEqual(profile["threads"], 12)
        self.assertEqual(profile["threads_batch"], 24)

    def test_numa_auto_distributes_only_for_cpu_only_multi_node_hosts(self):
        self.assertEqual(cpu_tuning.resolve_numa_mode("Auto", cpu_only=True, nodes=2), "Distribute")
        self.assertEqual(cpu_tuning.resolve_numa_mode("Auto", cpu_only=True, nodes=1), "Disabled")
        self.assertEqual(cpu_tuning.resolve_numa_mode("Auto", cpu_only=False, nodes=4), "Disabled")

    def test_explicit_numa_strategy_is_preserved(self):
        for mode in ("Disabled", "Distribute", "Isolate", "Numactl"):
            with self.subTest(mode=mode):
                self.assertEqual(cpu_tuning.resolve_numa_mode(mode, cpu_only=True, nodes=1), mode)


if __name__ == "__main__":
    unittest.main()
