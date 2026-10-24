"""Semantic checks for the paper's detector and the asynchronous simulator."""
import unittest
import numpy as np
from simulator import (detector_step, update_reward, push, clear_queue,
                       make_schedule, validate, simulate_adaptive, PROBE, VERIFY)


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([True, True, False])
        self.s = np.zeros(3, dtype=bool)

    def test_two_probes_only_create_pending(self):
        v, removed = detector_step(self.a, self.s, -1, 0, PROBE, False)
        self.assertEqual((v, removed), (-1, -1))
        v, removed = detector_step(self.a, self.s, v, 0, PROBE, False)
        self.assertEqual((v, removed), (0, -1))
        self.assertTrue(self.a[0])

    def test_collision_breaks_consecutive_probe_successes(self):
        detector_step(self.a, self.s, -1, 0, PROBE, False)
        detector_step(self.a, self.s, -1, 0, PROBE, True)
        v, removed = detector_step(self.a, self.s, -1, 0, PROBE, False)
        self.assertEqual((v, removed), (-1, -1))

    def test_failed_verification_preserves_occupied_set(self):
        self.s[:] = True
        v, removed = detector_step(self.a, self.s, 0, 0, VERIFY, True)
        self.assertEqual((v, removed), (-1, -1))
        np.testing.assert_array_equal(self.a, [True, True, False])
        np.testing.assert_array_equal(self.s, [False, True, True])

    def test_verified_release_resets_all_suspicion(self):
        self.s[:] = True
        v, removed = detector_step(self.a, self.s, 0, 0, VERIFY, False)
        self.assertEqual((v, removed), (-1, 0))
        np.testing.assert_array_equal(self.a, [False, True, False])
        self.assertFalse(self.s.any())

    def test_verification_never_updates_reward_statistics(self):
        n, avg = np.zeros((1, 3), dtype=np.int64), np.zeros((1, 3))
        u, l = np.full((1, 3), np.inf), np.full((1, 3), -np.inf)
        update_reward(n, avg, u, l, 0, 0, 1., False, VERIFY, .5)
        self.assertEqual(n.sum(), 0)
        update_reward(n, avg, u, l, 0, 0, 1., True, PROBE, .5)
        self.assertEqual(n.sum(), 0)
        update_reward(n, avg, u, l, 0, 0, .75, False, PROBE, .5)
        self.assertEqual(n[0, 0], 1)
        self.assertEqual(avg[0, 0], .75)

    def test_queue_is_rolling_sum_not_lifetime_count(self):
        q = np.zeros((1, 1, 3), dtype=np.int8)
        p, sums = np.zeros((1, 1), dtype=np.int64), np.zeros((1, 1), dtype=np.int64)
        for v in [1, 1, 1, 0, 0]:
            push(q, p, sums, 0, 0, v)
        self.assertEqual(sums[0, 0], 1)
        clear_queue(q, p, sums, 0, 0)
        self.assertEqual(sums[0, 0], 0)
        self.assertFalse(q.any())


class SimulationTests(unittest.TestCase):
    def test_exact_activity_denominator_and_asynchronous_pair_offset(self):
        starts, ends, dep = make_schedule()
        self.assertEqual(int(np.sum(ends - starts + 1)), 12_080_004)
        self.assertEqual(int(np.sum((starts <= 80_000) & (ends >= 80_000))), 10)
        self.assertEqual(int(np.sum((starts <= 100_001) & (ends >= 100_001))), 6)
        self.assertNotEqual(starts[0] % 2, starts[4] % 2)

    def test_determinism_metric_accounting_and_input_isolation(self):
        T = 2000
        mu = .1 + .05 * np.arange(19, -1, -1)
        starts, ends, dep = make_schedule(T)
        validate(T, mu, starts, ends, 10)
        points = np.array([0, dep, T], dtype=np.int64)
        noise = np.random.default_rng(2).normal(0, .5, (T, 10))
        state = np.random.SeedSequence(7).generate_state(10, dtype=np.uint64)
        before = noise.copy()
        args = (0, T, mu, starts, ends, 10, noise)
        a = simulate_adaptive(*args, state.copy(), .01, .02, 100, 500, points, dep)
        b = simulate_adaptive(*args, state.copy(), .01, .02, 100, 500, points, dep)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(noise, before)
        self.assertEqual(a[0][3], np.sum(ends - starts + 1))
        self.assertAlmostEqual(a[0][1], a[0][0] - a[1][1, 1])
        self.assertAlmostEqual(a[1][-1, 1], a[0][0])
        self.assertGreaterEqual(a[0][0], 0)
        self.assertLessEqual(a[0][2], a[0][3])
        self.assertTrue(np.all(np.diff(a[1][:, 1]) >= 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
