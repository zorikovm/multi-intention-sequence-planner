import unittest
import sys
import types

import numpy as np

try:
    import jax.numpy as jnp
except ModuleNotFoundError:  # Let pure planner tests run in lightweight environments.
    jnp = np
    fake_jax = types.ModuleType('jax')
    fake_random = types.SimpleNamespace(
        PRNGKey=lambda seed: int(seed),
        split=lambda key: (int(key) + 1, int(key) + 2),
    )
    fake_jax.random = fake_random
    fake_jax.numpy = np
    sys.modules['jax'] = fake_jax
    sys.modules['jax.numpy'] = np

from utils.multiswitch_planner import (
    MultiSwitchPlanner,
    PlannerConfig,
    _robust_ratio,
    select_landmarks_fps,
)


class _FakeNetwork:
    def select(self, name):
        if name != 'backward_repr':
            raise KeyError(name)

        def backward(observations):
            return jnp.asarray(observations)[..., :2]

        return backward


class _FakeAgent:
    config = {'frame_stack': None}
    network = _FakeNetwork()

    def successor_measure_extract(self, observations, z_goals, z_intents):
        del z_intents
        observations = jnp.asarray(observations)[..., :2]
        targets = jnp.asarray(z_goals)[..., :2]
        value = jnp.exp(-jnp.linalg.norm(observations - targets, axis=-1))
        return jnp.stack([value, value], axis=0)

    def sample_low_actions(self, observations, intentions, seed=None, temperature=0.0):
        del observations, intentions, seed, temperature
        return jnp.zeros(2)

    def sample_actions(self, observations, latents, seed=None, temperature=0.0):
        del observations, latents, seed, temperature
        return jnp.ones(2)


class MultiSwitchPlannerTest(unittest.TestCase):
    def test_robust_ratio_penalizes_disagreement(self):
        numerator = np.asarray([[0.8], [0.2]])
        denominator = np.ones((2, 1))
        unpenalized = _robust_ratio(numerator, denominator, 0.0)[0]
        penalized = _robust_ratio(numerator, denominator, 1.0)[0]
        self.assertLess(penalized, unpenalized)

    def test_landmarks_are_dataset_states(self):
        observations = np.arange(20, dtype=np.float32).reshape(10, 2)
        landmarks, _, indices = select_landmarks_fps(
            observations,
            observations,
            num_landmarks=4,
            num_candidates=10,
            seed=0,
        )
        np.testing.assert_array_equal(landmarks, observations[indices])

    def test_planner_finds_and_executes_multihop_route(self):
        positions = np.stack([np.arange(6), np.zeros(6)], axis=-1).astype(np.float32)
        planner = MultiSwitchPlanner(
            _FakeAgent(),
            dataset_observations=positions,
            dataset_positions=positions,
            config=PlannerConfig(
                num_landmarks=6,
                landmark_candidates=6,
                num_neighbors=2,
                max_waypoints=6,
                min_reachability=1e-8,
                uncertainty_penalty=0.0,
                switch_cost=0.0,
                waypoint_tolerance=0.1,
                allow_direct_goal=False,
                use_high_actor_for_waypoints=False,
                min_route_excess=0.0,
                route_stride=1,
                seed=0,
            ),
        )
        planner.reset(np.asarray([-1.0, 0.0]), np.asarray([6.0, 0.0]))
        self.assertGreater(planner.get_metrics()['initial_route_waypoints'], 1.0)
        np.testing.assert_array_equal(planner.sample_action(np.asarray([-1.0, 0.0])), np.zeros(2))


if __name__ == '__main__':
    unittest.main()
