"""Test-time multi-intention planning with frozen FB representations."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PlannerConfig:
    num_landmarks: int = 256
    landmark_candidates: int = 20_000
    num_neighbors: int = 12
    max_waypoints: int = 32
    min_reachability: float = 1e-6
    uncertainty_penalty: float = 0.5
    switch_cost: float = 0.02
    waypoint_tolerance: float = 1.75
    terminal_tolerance: float = 0.5
    max_subgoal_steps: int = 120
    stall_steps: int = 40
    replan_on_waypoint: bool = False
    allow_direct_goal: bool = False
    use_high_actor_for_waypoints: bool = True
    min_route_waypoints: int = 0
    min_route_detour: float = 0.0
    max_route_detour: float = float('inf')
    min_route_excess: float = 22.0
    max_route_excess: float = float('inf')
    max_replans_before_fallback: int = -1
    route_stride: int = 3
    seed: int = 0
    inference_batch_size: int = 4096


def _as_ensemble(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values[None, :]
    if values.ndim != 2:
        raise ValueError(f'Expected successor values with 1 or 2 dims, got {values.shape}.')
    return values


def _robust_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    uncertainty_penalty: float,
    eps: float = 1e-8,
) -> np.ndarray:
    numerator = _as_ensemble(numerator)
    denominator = _as_ensemble(denominator)
    ratios = numerator / np.maximum(denominator, eps)
    valid = np.isfinite(ratios) & (ratios > 0.0)
    ratios = np.where(valid, np.clip(ratios, eps, 1.0), eps)
    log_ratios = np.log(ratios)
    robust_log = log_ratios.mean(axis=0) - uncertainty_penalty * log_ratios.std(axis=0)
    return np.exp(robust_log)


def select_landmarks_fps(
    observations: np.ndarray,
    positions: np.ndarray,
    num_landmarks: int,
    num_candidates: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = np.asarray(observations)
    positions = np.asarray(positions, dtype=np.float32)
    if len(observations) != len(positions):
        raise ValueError('Observations and positions must have equal length.')
    if len(observations) == 0:
        raise ValueError('Cannot select landmarks from an empty dataset.')

    rng = np.random.default_rng(seed)
    candidate_count = min(int(num_candidates), len(observations))
    candidate_idxs = rng.choice(len(observations), size=candidate_count, replace=False)
    candidate_positions = positions[candidate_idxs]

    landmark_count = min(int(num_landmarks), candidate_count)
    center = candidate_positions.mean(axis=0)
    first = int(np.argmax(np.sum((candidate_positions - center) ** 2, axis=-1)))
    selected = [first]
    min_sq_dist = np.sum((candidate_positions - candidate_positions[first]) ** 2, axis=-1)

    for _ in range(1, landmark_count):
        nxt = int(np.argmax(min_sq_dist))
        selected.append(nxt)
        sq_dist = np.sum((candidate_positions - candidate_positions[nxt]) ** 2, axis=-1)
        min_sq_dist = np.minimum(min_sq_dist, sq_dist)

    selected_idxs = candidate_idxs[np.asarray(selected)]
    return observations[selected_idxs], positions[selected_idxs], selected_idxs


class MultiSwitchPlanner:
    def __init__(
        self,
        agent,
        dataset_observations: np.ndarray,
        dataset_positions: np.ndarray,
        config: PlannerConfig,
    ):
        if not hasattr(agent, 'sample_low_actions'):
            raise TypeError('MultiSwitchPlanner requires an agent with sample_low_actions().')

        self.agent = agent
        self.config = config
        self.rng = jax.random.PRNGKey(config.seed)
        self.landmarks, self.landmark_positions, self.landmark_indices = select_landmarks_fps(
            dataset_observations,
            dataset_positions,
            config.num_landmarks,
            config.landmark_candidates,
            config.seed,
        )
        self.landmark_latents = self._encode(self.landmarks)
        self.landmark_denominators = self._self_successor(self.landmarks, self.landmark_latents)
        self.base_edges = self._build_landmark_graph()

        self.goal: Optional[np.ndarray] = None
        self.goal_position: Optional[np.ndarray] = None
        self.goal_latent: Optional[np.ndarray] = None
        self.goal_denominator: Optional[np.ndarray] = None
        self.task_latent: Optional[np.ndarray] = None
        self._cached_goal_edges: Optional[Dict[int, Tuple[float, float]]] = None
        self.route: List[int] = []
        self.active_node: Optional[int] = None
        self.active_target: Optional[np.ndarray] = None
        self.active_position: Optional[np.ndarray] = None
        self.active_latent: Optional[np.ndarray] = None
        self.steps_on_target = 0
        self.steps_without_progress = 0
        self.best_target_distance = np.inf
        self.metrics: Dict[str, float] = {}
        self.enabled = True

    @property
    def num_landmarks(self) -> int:
        return len(self.landmarks)

    def _position(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation)
        frame_stack = self.agent.config.get('frame_stack')
        if frame_stack is not None and int(frame_stack) > 1:
            observation = observation.reshape(int(frame_stack), -1)[-1]
        return np.asarray(observation[:2], dtype=np.float32)

    def _encode(self, observations: np.ndarray) -> np.ndarray:
        values = self.agent.network.select('backward_repr')(jnp.asarray(observations))
        return np.asarray(values)

    def _self_successor(self, states: np.ndarray, latents: np.ndarray) -> np.ndarray:
        chunks = []
        batch_size = self.config.inference_batch_size
        for start in range(0, len(states), batch_size):
            stop = min(start + batch_size, len(states))
            values = self.agent.successor_measure_extract(
                jnp.asarray(states[start:stop]),
                jnp.asarray(latents[start:stop]),
                jnp.asarray(latents[start:stop]),
            )
            chunks.append(_as_ensemble(np.asarray(values)))
        return np.concatenate(chunks, axis=1)

    def _reachability(self, sources, target_latents, target_denominators):
        outputs = []
        batch_size = self.config.inference_batch_size
        for start in range(0, len(sources), batch_size):
            stop = min(start + batch_size, len(sources))
            numerator = self.agent.successor_measure_extract(
                jnp.asarray(sources[start:stop]),
                jnp.asarray(target_latents[start:stop]),
                jnp.asarray(target_latents[start:stop]),
            )
            outputs.append(
                _robust_ratio(
                    np.asarray(numerator),
                    target_denominators[:, start:stop],
                    self.config.uncertainty_penalty,
                )
            )
        return np.concatenate(outputs, axis=0)

    @staticmethod
    def _nearest_indices(position, candidates, count):
        sq_dist = np.sum((candidates - position[None, :]) ** 2, axis=-1)
        count = min(int(count), len(candidates))
        if count == len(candidates):
            return np.argsort(sq_dist)
        idxs = np.argpartition(sq_dist, count - 1)[:count]
        return idxs[np.argsort(sq_dist[idxs])]

    def _build_landmark_graph(self):
        n = self.num_landmarks
        neighbor_count = min(self.config.num_neighbors + 1, n)
        sq_dist = np.sum(
            (self.landmark_positions[:, None, :] - self.landmark_positions[None, :, :]) ** 2,
            axis=-1,
        )
        nearest = np.argpartition(sq_dist, neighbor_count - 1, axis=1)[:, :neighbor_count]
        source_idxs, target_idxs = [], []
        for source in range(n):
            ordered = nearest[source][np.argsort(sq_dist[source, nearest[source]])]
            ordered = ordered[ordered != source][: self.config.num_neighbors]
            source_idxs.extend([source] * len(ordered))
            target_idxs.extend(ordered.tolist())
        source_idxs = np.asarray(source_idxs, dtype=np.int32)
        target_idxs = np.asarray(target_idxs, dtype=np.int32)
        reaches = self._reachability(
            self.landmarks[source_idxs],
            self.landmark_latents[target_idxs],
            self.landmark_denominators[:, target_idxs],
        )
        edges = [[] for _ in range(n)]
        for source, target, reach in zip(source_idxs, target_idxs, reaches):
            if reach < self.config.min_reachability:
                continue
            cost = -float(np.log(reach)) + self.config.switch_cost
            edges[int(source)].append((int(target), cost, float(reach)))
        return edges

    def reset(self, observation, goal, task_latent=None):
        self.goal = np.asarray(goal)
        self.goal_position = self._position(goal)
        self.goal_latent = self._encode(self.goal[None])[0]
        self.goal_denominator = self._self_successor(self.goal[None], self.goal_latent[None])
        self.task_latent = np.asarray(task_latent) if task_latent is not None else self.goal_latent
        self._cached_goal_edges = None
        self.route = []
        self.active_node = None
        self.active_target = None
        self.active_position = None
        self.active_latent = None
        self.steps_on_target = 0
        self.steps_without_progress = 0
        self.best_target_distance = np.inf
        self.metrics = {
            'plans': 0.0,
            'replans': 0.0,
            'waypoints_reached': 0.0,
            'fallback_actions': 0.0,
            'initial_route_waypoints': 0.0,
            'planner_enabled': 0.0,
            'initial_route_xy_length': 0.0,
            'initial_route_detour': 0.0,
            'initial_route_excess': 0.0,
            'executed_route_waypoints': 0.0,
            'failed_targets': 0.0,
            'stall_replans': 0.0,
            'timeout_replans': 0.0,
            'planner_abandoned': 0.0,
        }
        self.enabled = True
        self._plan(np.asarray(observation), is_replan=False)

    def _goal_edges(self):
        if self._cached_goal_edges is not None:
            return self._cached_goal_edges
        targets = self._nearest_indices(
            self.goal_position, self.landmark_positions, self.config.num_neighbors
        )
        goal_latents = np.repeat(self.goal_latent[None], len(targets), axis=0)
        denoms = np.repeat(self.goal_denominator, len(targets), axis=1)
        reaches = self._reachability(self.landmarks[targets], goal_latents, denoms)
        self._cached_goal_edges = {
            int(source): (-float(np.log(reach)) + self.config.switch_cost, float(reach))
            for source, reach in zip(targets, reaches)
            if reach >= self.config.min_reachability
        }
        return self._cached_goal_edges

    def _start_edges(self, observation):
        position = self._position(observation)
        targets = self._nearest_indices(position, self.landmark_positions, self.config.num_neighbors)
        sources = np.repeat(observation[None], len(targets), axis=0)
        reaches = self._reachability(
            sources,
            self.landmark_latents[targets],
            self.landmark_denominators[:, targets],
        )
        edges = [
            (int(target), -float(np.log(reach)) + self.config.switch_cost, float(reach))
            for target, reach in zip(targets, reaches)
            if reach >= self.config.min_reachability
        ]
        if self.config.allow_direct_goal:
            direct_reach = self._reachability(
                observation[None], self.goal_latent[None], self.goal_denominator
            )[0]
            if direct_reach >= self.config.min_reachability:
                edges.append(
                    (
                        self.num_landmarks,
                        -float(np.log(direct_reach)) + self.config.switch_cost,
                        float(direct_reach),
                    )
                )
        return edges

    def _shortest_route(self, observation):
        n = self.num_landmarks
        start_node, goal_node = n + 1, n
        goal_edges = self._goal_edges()
        start_edges = self._start_edges(observation)

        def neighbors(node):
            if node == start_node:
                return start_edges
            if 0 <= node < n:
                result = list(self.base_edges[node])
                if node in goal_edges:
                    cost, reach = goal_edges[node]
                    result.append((goal_node, cost, reach))
                return result
            return []

        initial = (start_node, 0)
        queue = [(0.0, start_node, 0)]
        best = {initial: 0.0}
        parent = {}
        final_state = None
        while queue:
            cost, node, used = heapq.heappop(queue)
            state = (node, used)
            if cost > best.get(state, np.inf):
                continue
            if node == goal_node:
                final_state = state
                break
            for nxt, edge_cost, _ in neighbors(node):
                next_used = used + (1 if 0 <= nxt < n else 0)
                if next_used > self.config.max_waypoints:
                    continue
                next_state = (nxt, next_used)
                next_cost = cost + edge_cost
                if next_cost < best.get(next_state, np.inf):
                    best[next_state] = next_cost
                    parent[next_state] = state
                    heapq.heappush(queue, (next_cost, nxt, next_used))
        if final_state is None:
            return []
        nodes = []
        state = final_state
        while state != initial:
            nodes.append(state[0])
            state = parent[state]
        nodes.reverse()
        return nodes

    def _set_active_target(self):
        if not self.route:
            self.active_node = None
            self.active_target = self.active_position = self.active_latent = None
            return
        target = self.route[0]
        self.active_node = target
        if target == self.num_landmarks:
            self.active_target = self.goal
            self.active_position = self.goal_position
            self.active_latent = self.goal_latent
        else:
            self.active_target = self.landmarks[target]
            self.active_position = self.landmark_positions[target]
            self.active_latent = self.landmark_latents[target]
        self.steps_on_target = 0
        self.steps_without_progress = 0
        self.best_target_distance = np.inf

    def _plan(self, observation, is_replan):
        candidate_route = self._shortest_route(observation)
        self.metrics['plans'] += 1.0
        if is_replan:
            self.metrics['replans'] += 1.0
        else:
            route_waypoints = sum(node < self.num_landmarks for node in candidate_route)
            route_positions = [self._position(observation)]
            for node in candidate_route:
                route_positions.append(
                    self.goal_position if node == self.num_landmarks else self.landmark_positions[node]
                )
            if len(route_positions) > 1:
                route_positions = np.asarray(route_positions)
                route_xy_length = float(np.linalg.norm(np.diff(route_positions, axis=0), axis=-1).sum())
                direct_xy = float(np.linalg.norm(route_positions[0] - route_positions[-1]))
                route_detour = route_xy_length / max(direct_xy, 1e-6)
                route_excess = route_xy_length - direct_xy
            else:
                route_xy_length = route_detour = route_excess = 0.0
            self.metrics['initial_route_waypoints'] = float(route_waypoints)
            self.metrics['initial_route_xy_length'] = route_xy_length
            self.metrics['initial_route_detour'] = route_detour
            self.metrics['initial_route_excess'] = route_excess
            self.enabled = (
                route_waypoints >= self.config.min_route_waypoints
                and route_detour >= self.config.min_route_detour
                and route_detour <= self.config.max_route_detour
                and route_excess >= self.config.min_route_excess
                and route_excess <= self.config.max_route_excess
            )
            self.metrics['planner_enabled'] = float(self.enabled)
        if self.enabled and self.config.route_stride > 1 and candidate_route:
            goal_node = self.num_landmarks
            landmark_route = [node for node in candidate_route if node != goal_node]
            compressed_route = landmark_route[self.config.route_stride - 1 :: self.config.route_stride]
            if candidate_route[-1] == goal_node:
                compressed_route.append(goal_node)
            self.route = compressed_route
        else:
            self.route = candidate_route if self.enabled else []
        if not is_replan:
            self.metrics['executed_route_waypoints'] = float(
                sum(node < self.num_landmarks for node in self.route)
            )
        self._set_active_target()

    def _advance_or_replan(self, observation):
        self.metrics['waypoints_reached'] += 1.0
        if self.route:
            self.route.pop(0)
        if self.config.replan_on_waypoint:
            self._plan(observation, is_replan=True)
        else:
            self._set_active_target()

    def _disable_planner(self):
        self.enabled = False
        self.route = []
        self.active_node = None
        self.active_target = self.active_position = self.active_latent = None
        self.metrics['planner_abandoned'] = 1.0

    def sample_action(self, observation, temperature=0.0):
        if self.goal is None:
            raise RuntimeError('Call reset() before sample_action().')
        observation = np.asarray(observation)
        if self.active_target is not None:
            distance = float(np.linalg.norm(self._position(observation) - self.active_position))
            tolerance = (
                self.config.terminal_tolerance
                if self.active_node == self.num_landmarks
                else self.config.waypoint_tolerance
            )
            if distance <= tolerance:
                self._advance_or_replan(observation)
            else:
                if distance + 1e-4 < self.best_target_distance:
                    self.best_target_distance = distance
                    self.steps_without_progress = 0
                else:
                    self.steps_without_progress += 1

        timed_out = self.active_target is not None and self.steps_on_target >= self.config.max_subgoal_steps
        stalled = self.active_target is not None and self.steps_without_progress >= self.config.stall_steps
        if timed_out or stalled:
            self.metrics['failed_targets'] += 1.0
            if stalled:
                self.metrics['stall_replans'] += 1.0
            else:
                self.metrics['timeout_replans'] += 1.0
            if (
                self.config.max_replans_before_fallback >= 0
                and self.metrics['replans'] >= self.config.max_replans_before_fallback
            ):
                self._disable_planner()
            else:
                self._plan(observation, is_replan=True)

        self.rng, key = jax.random.split(self.rng)
        if self.active_latent is None:
            self.metrics['fallback_actions'] += 1.0
            action = self.agent.sample_actions(
                jnp.asarray(observation),
                jnp.asarray(self.task_latent),
                seed=key,
                temperature=temperature,
            )
        elif self.config.use_high_actor_for_waypoints:
            action = self.agent.sample_actions(
                jnp.asarray(observation),
                jnp.asarray(self.active_latent),
                seed=key,
                temperature=temperature,
            )
            self.steps_on_target += 1
        else:
            action = self.agent.sample_low_actions(
                jnp.asarray(observation),
                jnp.asarray(self.active_latent),
                seed=key,
                temperature=temperature,
            )
            self.steps_on_target += 1
        return np.asarray(action)

    def get_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)
