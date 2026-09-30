from __future__ import annotations

import unittest

from seed_utils import (
    LLAMA_DEFAULT_SEED,
    MAX_REQUEST_SEED,
    RecentSamplerSeedAllocator,
    llama_seed_from_u64,
    parse_openai_seed,
)


class SeedUtilsTests(unittest.TestCase):
    def test_explicit_seed_zero_is_deterministic_not_random(self):
        allocator = RecentSamplerSeedAllocator(randbits=lambda _bits: 123)
        request_seed, sampler_seed, randomized = parse_openai_seed(0, allocator)
        self.assertEqual(request_seed, 0)
        self.assertEqual(sampler_seed, llama_seed_from_u64(0))
        self.assertFalse(randomized)

    def test_omitted_null_and_minus_one_randomize(self):
        values = iter([11, 12, 13])
        allocator = RecentSamplerSeedAllocator(randbits=lambda _bits: next(values))
        a = parse_openai_seed(None, allocator)
        b = parse_openai_seed(None, allocator)
        c = parse_openai_seed(-1, allocator)
        self.assertTrue(a[2] and b[2] and c[2])
        self.assertEqual([a[0], b[0], c[0]], [11, 12, 13])

    def test_random_allocator_redraws_when_effective_sampler_seed_collides(self):
        # Two distinct 64-bit request seeds can map to the same native 32-bit
        # sampler seed. Inject a mapper that forces that condition so the guard is
        # deterministic and cheap to test.
        raw = iter([100, 200, 300])
        sampler_map = {100: 7, 200: 7, 300: 8}
        allocator = RecentSamplerSeedAllocator(
            recent_limit=16,
            randbits=lambda _bits: next(raw),
            mapper=lambda seed: sampler_map[seed],
        )
        self.assertEqual(allocator.allocate(), (100, 7))
        self.assertEqual(allocator.allocate(), (300, 8))

    def test_allocator_evicts_old_sampler_seeds_after_bounded_history(self):
        raw = iter([1, 2, 3, 4])
        allocator = RecentSamplerSeedAllocator(
            recent_limit=2,
            randbits=lambda _bits: next(raw),
            mapper=lambda seed: seed,
        )
        self.assertEqual(allocator.allocate(), (1, 1))
        self.assertEqual(allocator.allocate(), (2, 2))
        self.assertEqual(allocator.allocate(), (3, 3))
        # Sampler seed 1 is now outside the recent window and may be reused.
        allocator._randbits = lambda _bits: 1
        self.assertEqual(allocator.allocate(), (1, 1))


    def test_random_seed_avoids_recent_explicit_sampler_seed(self):
        explicit_sampler = llama_seed_from_u64(42)
        candidates = iter([42, 43])
        allocator = RecentSamplerSeedAllocator(randbits=lambda _bits: next(candidates))
        self.assertEqual(parse_openai_seed(42, allocator)[:2], (42, explicit_sampler))
        request_seed, sampler_seed = allocator.allocate()
        self.assertEqual(request_seed, 43)
        self.assertNotEqual(sampler_seed, explicit_sampler)

    def test_broken_entropy_source_fails_instead_of_reusing_sampler_seed(self):
        allocator = RecentSamplerSeedAllocator(
            randbits=lambda _bits: 1,
            mapper=lambda _seed: 7,
        )
        self.assertEqual(allocator.allocate(), (1, 7))
        with self.assertRaises(RuntimeError):
            allocator.allocate()

    def test_reserved_llama_default_seed_is_never_emitted_by_mapping(self):
        # Sample representative edge values; the helper also explicitly remaps
        # 0xFFFFFFFF if SplitMix64 ever produces it.
        for value in (0, 1, 2, 0xFFFFFFFF, 0x100000000, MAX_REQUEST_SEED):
            with self.subTest(value=value):
                self.assertNotEqual(llama_seed_from_u64(value), LLAMA_DEFAULT_SEED)

    def test_invalid_openai_seed_values_are_rejected(self):
        allocator = RecentSamplerSeedAllocator(randbits=lambda _bits: 1)
        for value in (True, 1.25, "abc", -2, MAX_REQUEST_SEED + 1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_openai_seed(value, allocator)


if __name__ == "__main__":
    unittest.main()
