#!/usr/bin/env python3
"""Render the five Medium tasks and one paired T4 trajectory."""

import importlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'glfw')

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.pyplot as plt
import ml_collections
import numpy as np

from agents import agents
from agents.fbpiswitch import get_config
from scripts.render_large_examples import add_markers, draw_maze, replay
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets, relabel_dataset
from utils.flax_utils import restore_agent
from utils.multiswitch_planner import MultiSwitchPlanner, PlannerConfig


ENV_NAME = 'ogbench-antmaze-medium-navigate-v0'
TASK_ID = 4
EVAL_SEED = 0
EPISODE = 1
EPISODE_SEED = 40_001


def load_experiment():
    checkpoint_dir = Path('artifacts/checkpoints/medium')
    with open(checkpoint_dir / 'flags.json') as file:
        saved = json.load(file)
    config = get_config()
    config.unlock()
    config.update(ml_collections.ConfigDict(saved['agent']))
    config.lock()

    env, train_data, val_data = make_env_and_datasets(
        ENV_NAME,
        frame_stack=config['frame_stack'],
        add_info=True,
        no_renderer=True,
        dataset_dir='artifacts/data',
    )
    env.unwrapped._add_noise_to_goal = False
    raw_train = Dataset.create(**train_data)
    raw_val = Dataset.create(**val_data)
    dataset_class = getattr(importlib.import_module('utils.datasets'), config['dataset_class'])
    train = dataset_class(raw_train, config)
    agent = agents[config['agent_name']].create(0, train.sample(1), config)
    agent = restore_agent(agent, str(checkpoint_dir), None)
    return env, raw_train, raw_val, dataset_class, config, agent


def infer_latent(env, raw_val, dataset_class, config, agent):
    env.reset(options={'task_id': TASK_ID})
    relabeled = relabel_dataset(ENV_NAME, env, raw_val)
    dataset = dataset_class(Dataset.create(**relabeled), config)
    count = int(config.get('num_zero_shot_samples', 100_000))
    batch = dataset.sample(
        count,
        idxs=np.arange(count),
        relabeling=False,
        augmentation=False,
    )
    return np.asarray(agent.infer_latent(batch))


def make_planner(agent, raw_train):
    return MultiSwitchPlanner(
        agent,
        np.asarray(raw_train['observations']),
        np.asarray(raw_train['qpos'])[:, :2],
        PlannerConfig(
            num_landmarks=256,
            landmark_candidates=20_000,
            num_neighbors=12,
            max_waypoints=32,
            min_reachability=1e-6,
            uncertainty_penalty=0.5,
            switch_cost=0.02,
            waypoint_tolerance=1.75,
            terminal_tolerance=1.75,
            max_subgoal_steps=120,
            stall_steps=40,
            allow_direct_goal=False,
            use_high_actor_for_waypoints=True,
            min_route_excess=22.0,
            route_stride=3,
            seed=0,
        ),
    )


def render_overview(output_dir, env):
    maze_map = np.asarray(env.unwrapped.maze_map)
    maze_unit = float(env.unwrapped._maze_unit)
    tasks = env.unwrapped.task_infos
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    for task_id, ax in enumerate(axes, 1):
        draw_maze(ax, maze_map, maze_unit)
        task = tasks[task_id - 1]
        add_markers(ax, np.asarray(task['init_xy']), np.asarray(task['goal_xy']))
        ax.set_title(f'T{task_id}', fontsize=13, weight='bold')
    fig.suptitle('Задачи AntMaze Medium', fontsize=16, weight='bold')
    path = output_dir / 'antmaze_medium_tasks.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_pair(output_dir, env, baseline, planner):
    maze_map = np.asarray(env.unwrapped.maze_map)
    maze_unit = float(env.unwrapped._maze_unit)
    task = env.unwrapped.task_infos[TASK_ID - 1]
    start = baseline[0][0]
    goal = np.asarray(task['goal_xy'])
    colors = ('#7445db', '#f59e0b')

    def setup():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
        titles = (
            f'Исходный контроллер | успех={baseline[1]}',
            f'Планировщик | успех={planner[1]}',
        )
        for ax, title in zip(axes, titles):
            draw_maze(ax, maze_map, maze_unit)
            add_markers(ax, start, goal)
            ax.set_title(title, fontsize=12, weight='bold')
        fig.suptitle(f'AntMaze Medium, T4, seed эпизода {EPISODE_SEED}', fontsize=14, weight='bold')
        return fig, axes

    fig, axes = setup()
    for ax, trajectory, color in zip(axes, (baseline[0], planner[0]), colors):
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2)
        ax.scatter(*trajectory[-1], s=42, c=color, edgecolor='white', zorder=7)
    image_path = output_dir / 'medium_t4_seed40001.png'
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    fig, axes = setup()
    lines, dots, labels = [], [], []
    for ax, color in zip(axes, colors):
        line, = ax.plot([], [], color=color, linewidth=2)
        dot = ax.scatter([], [], s=42, c=color, edgecolor='white', zorder=7)
        label = ax.text(0.98, 0.98, '', transform=ax.transAxes, ha='right', va='top')
        lines.append(line)
        dots.append(dot)
        labels.append(label)

    frame_skip = 4
    trajectories = (baseline[0], planner[0])
    steps = (baseline[2], planner[2])
    frame_count = (max(map(len, trajectories)) - 1) // frame_skip + 2

    def update(frame):
        for trajectory, line, dot, label, total in zip(trajectories, lines, dots, labels, steps):
            stop = min(frame * frame_skip + 1, len(trajectory))
            shown = trajectory[:stop]
            line.set_data(shown[:, 0], shown[:, 1])
            dot.set_offsets(shown[-1:])
            label.set_text(f'шаг {min(stop - 1, total)} / {total}')
        return (*lines, *dots, *labels)

    animation = FuncAnimation(fig, update, frames=frame_count, interval=1000 / 30, blit=False)
    video_path = output_dir / 'medium_t4_seed40001.mp4'
    animation.save(video_path, writer=FFMpegWriter(fps=30, codec='libx264', bitrate=1800))
    plt.close(fig)
    return image_path, video_path


def main():
    output_dir = Path('media')
    output_dir.mkdir(exist_ok=True)
    env, raw_train, raw_val, dataset_class, config, agent = load_experiment()
    render_overview(output_dir, env)
    latent = infer_latent(env, raw_val, dataset_class, config, agent)
    baseline = replay(agent, env, TASK_ID, EVAL_SEED, EPISODE, latent, planner=None)
    planner = replay(
        agent,
        env,
        TASK_ID,
        EVAL_SEED,
        EPISODE,
        latent,
        planner=make_planner(agent, raw_train),
    )
    image_path, video_path = render_pair(output_dir, env, baseline, planner)
    record = {
        'task_id': TASK_ID,
        'eval_seed': EVAL_SEED,
        'episode': EPISODE,
        'episode_seed': EPISODE_SEED,
        'baseline_success': baseline[1],
        'planner_success': planner[1],
        'baseline_steps': baseline[2],
        'planner_steps': planner[2],
        'image': image_path.name,
        'video': video_path.name,
    }
    with open(output_dir / 'medium_example.json', 'w') as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
