#!/usr/bin/env python3
"""Replay selected Large episodes and render top-down paired trajectories."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'glfw')

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import jax
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Rectangle
import ml_collections
import numpy as np

from agents import agents
from agents.fbpiswitch import get_config
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets, relabel_dataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.multiswitch_planner import MultiSwitchPlanner, PlannerConfig


DEFAULT_CASES = (
    '4:1:2:large_t4_seed1040002',
    '5:1:2:large_t5_seed1050002',
    '5:0:3:large_t5_seed50003',
)


def parse_case(value):
    task_id, eval_seed, episode, name = value.split(':', 3)
    return int(task_id), int(eval_seed), int(episode), name


def load_experiment(dataset_dir, checkpoint_dir):
    with open(Path(checkpoint_dir) / 'flags.json') as file:
        saved = json.load(file)
    config = get_config()
    config.unlock()
    config.update(ml_collections.ConfigDict(saved['agent']))
    config.lock()

    env, train_data, val_data = make_env_and_datasets(
        'ogbench-antmaze-large-navigate-v0',
        frame_stack=config['frame_stack'],
        add_info=True,
        no_renderer=True,
        dataset_dir=dataset_dir,
    )
    env.unwrapped._add_noise_to_goal = False

    raw_train = Dataset.create(**train_data)
    raw_val = Dataset.create(**val_data)
    dataset_class = getattr(importlib.import_module('utils.datasets'), config['dataset_class'])
    train = dataset_class(raw_train, config)
    example = train.sample(1)
    agent = agents[config['agent_name']].create(0, example, config)
    agent = restore_agent(agent, checkpoint_dir, None)
    return env, raw_train, raw_val, dataset_class, config, agent


def infer_task_latent(env, raw_val, dataset_class, config, agent, task_id):
    env.reset(options={'task_id': task_id})
    relabeled = relabel_dataset(
        'ogbench-antmaze-large-navigate-v0', env, raw_val
    )
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
    observations = np.asarray(raw_train['observations'])
    positions = np.asarray(raw_train['qpos'])[:, :2]
    return MultiSwitchPlanner(
        agent,
        observations,
        positions,
        PlannerConfig(
            num_landmarks=256,
            landmark_candidates=20_000,
            num_neighbors=12,
            max_waypoints=32,
            min_reachability=1e-6,
            uncertainty_penalty=0.5,
            switch_cost=0.02,
            waypoint_tolerance=1.75,
            terminal_tolerance=0.5,
            max_subgoal_steps=120,
            stall_steps=40,
            allow_direct_goal=False,
            use_high_actor_for_waypoints=True,
            min_route_waypoints=20,
            min_route_detour=0.0,
            max_route_detour=4.0,
            min_route_excess=22.0,
            max_replans_before_fallback=0,
            route_stride=3,
            seed=0,
        ),
    )


def replay(agent, env, task_id, eval_seed, episode, latent, planner):
    _, trajectories, _ = evaluate(
        agent=agent,
        env=env,
        task_id=task_id,
        inferred_latent=latent,
        num_eval_episodes=episode + 1,
        num_video_episodes=0,
        eval_temperature=0.0,
        planner=planner,
        eval_seed=eval_seed,
    )
    trajectory = trajectories[episode]
    xy = np.asarray(trajectory['observation'])[:, :2]
    final_info = trajectory['info'][-1]
    return xy, int(final_info.get('success', 0.0)), len(trajectory['action'])


def draw_maze(ax, maze_map, maze_unit):
    rows, cols = maze_map.shape
    for row in range(rows):
        for col in range(cols):
            x = (col - 1) * maze_unit
            y = (row - 1) * maze_unit
            color = '#202631' if maze_map[row, col] else '#f4f6f8'
            ax.add_patch(
                Rectangle(
                    (x - maze_unit / 2, y - maze_unit / 2),
                    maze_unit,
                    maze_unit,
                    facecolor=color,
                    edgecolor='#cbd2da',
                    linewidth=0.45,
                )
            )
    ax.set_xlim(-1.5 * maze_unit, (cols - 1.5) * maze_unit)
    ax.set_ylim((rows - 1.5) * maze_unit, -1.5 * maze_unit)
    ax.set_aspect('equal')
    ax.axis('off')


def add_markers(ax, start, goal):
    ax.scatter(*start, s=95, c='#24a148', edgecolor='white', linewidth=1.5, zorder=5)
    ax.scatter(*goal, s=105, c='#d73027', edgecolor='white', linewidth=1.5, zorder=5)
    ax.text(start[0], start[1], 'S', ha='center', va='center', color='white', weight='bold', zorder=6)
    ax.text(goal[0], goal[1], 'G', ha='center', va='center', color='white', weight='bold', zorder=6)


def render_pair(output_dir, name, task_id, episode_seed, maze_map, maze_unit, baseline, hybrid):
    baseline_xy, baseline_success, baseline_steps = baseline
    hybrid_xy, hybrid_success, hybrid_steps = hybrid
    start = baseline_xy[0]
    goal = np.asarray(TASKS[task_id]['goal_xy'])
    colors = ('#7445db', '#f59e0b')

    def setup_figure():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
        titles = (
            f'Исходный контроллер | success={baseline_success}',
            f'Планировщик с возвратом | success={hybrid_success}',
        )
        for ax, title in zip(axes, titles):
            draw_maze(ax, maze_map, maze_unit)
            add_markers(ax, start, goal)
            ax.set_title(title, fontsize=12, weight='bold')
        fig.suptitle(f'AntMaze Large, T{task_id}, seed эпизода {episode_seed}', fontsize=14, weight='bold')
        return fig, axes

    fig, axes = setup_figure()
    axes[0].plot(baseline_xy[:, 0], baseline_xy[:, 1], color=colors[0], linewidth=2)
    axes[1].plot(hybrid_xy[:, 0], hybrid_xy[:, 1], color=colors[1], linewidth=2)
    axes[0].scatter(*baseline_xy[-1], s=42, c=colors[0], edgecolor='white', zorder=7)
    axes[1].scatter(*hybrid_xy[-1], s=42, c=colors[1], edgecolor='white', zorder=7)
    png_path = output_dir / f'{name}.png'
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    fig, axes = setup_figure()
    lines, dots, step_texts = [], [], []
    for ax, color in zip(axes, colors):
        line, = ax.plot([], [], color=color, linewidth=2)
        dot = ax.scatter([], [], s=42, c=color, edgecolor='white', zorder=7)
        step_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, ha='right', va='top')
        lines.append(line)
        dots.append(dot)
        step_texts.append(step_text)

    frame_skip = 4
    frame_count = (max(len(baseline_xy), len(hybrid_xy)) - 1) // frame_skip + 2

    def update(frame):
        for trajectory, line, dot, text_box, total_steps in zip(
            (baseline_xy, hybrid_xy), lines, dots, step_texts, (baseline_steps, hybrid_steps)
        ):
            stop = min(frame * frame_skip + 1, len(trajectory))
            shown = trajectory[:stop]
            line.set_data(shown[:, 0], shown[:, 1])
            dot.set_offsets(shown[-1:])
            text_box.set_text(f'шаг {min(stop - 1, total_steps)} / {total_steps}')
        return (*lines, *dots, *step_texts)

    animation = FuncAnimation(fig, update, frames=frame_count, interval=1000 / 30, blit=False)
    mp4_path = output_dir / f'{name}.mp4'
    animation.save(mp4_path, writer=FFMpegWriter(fps=30, codec='libx264', bitrate=1800))
    plt.close(fig)
    return png_path, mp4_path


def render_task_overview(output_dir, maze_map, maze_unit):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for ax, task_id in zip(axes, (2, 4, 5)):
        draw_maze(ax, maze_map, maze_unit)
        task = TASKS[task_id]
        add_markers(ax, np.asarray(task['init_xy']), np.asarray(task['goal_xy']))
        ax.set_title(f'T{task_id}', fontsize=14, weight='bold')
    fig.suptitle('Задачи финальной проверки AntMaze Large', fontsize=16, weight='bold')
    path = output_dir / 'antmaze_large_tasks.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', default='artifacts/data')
    parser.add_argument('--checkpoint-dir', default='artifacts/checkpoints/large')
    parser.add_argument('--output-dir', default='media')
    parser.add_argument('--case', action='append', dest='cases')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env, raw_train, raw_val, dataset_class, config, agent = load_experiment(
        args.dataset_dir, args.checkpoint_dir
    )
    global TASKS
    TASKS = {index + 1: task for index, task in enumerate(env.unwrapped.task_infos)}
    maze_map = np.asarray(env.unwrapped.maze_map)
    maze_unit = float(env.unwrapped._maze_unit)
    render_task_overview(output_dir, maze_map, maze_unit)

    records = []
    for case_value in args.cases or DEFAULT_CASES:
        task_id, eval_seed, episode, name = parse_case(case_value)
        latent = infer_task_latent(env, raw_val, dataset_class, config, agent, task_id)
        baseline = replay(agent, env, task_id, eval_seed, episode, latent, planner=None)
        planner = make_planner(agent, raw_train)
        hybrid = replay(agent, env, task_id, eval_seed, episode, latent, planner=planner)
        episode_seed = eval_seed * 1_000_000 + task_id * 10_000 + episode
        png_path, mp4_path = render_pair(
            output_dir,
            name,
            task_id,
            episode_seed,
            maze_map,
            maze_unit,
            baseline,
            hybrid,
        )
        records.append(
            {
                'task_id': task_id,
                'eval_seed': eval_seed,
                'episode': episode,
                'episode_seed': episode_seed,
                'baseline_success': baseline[1],
                'planner_success': hybrid[1],
                'baseline_steps': baseline[2],
                'planner_steps': hybrid[2],
                'image': png_path.name,
                'video': mp4_path.name,
            }
        )
        print(json.dumps(records[-1], ensure_ascii=False))

    with open(output_dir / 'large_examples.json', 'w') as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
