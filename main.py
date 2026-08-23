import json
import importlib
import os
# Keep EGL as the default used by the original experiments, while allowing
# CPU-only/headless hosts to select another MuJoCo backend externally.
os.environ.setdefault("MUJOCO_GL", "egl")
import random
import time
from collections import defaultdict

import numpy as np
np.in1d = np.isin
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets, relabel_dataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.multiswitch_planner import MultiSwitchPlanner, PlannerConfig


FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'experiments', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'ogbench-antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string(
    'dataset_dir',
    None,
    'Optional local OGBench dataset directory. Useful on read-only or offline hosts.',
)
flags.DEFINE_string('save_dir', 'exp_logs', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1_000_000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5_000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 100_000, 'Saving interval.')
flags.DEFINE_string('complex_task_name', None, 'None for GCRL tasks, "regions" for more complex region-based rewards')
flags.DEFINE_string(
    'eval_tasks',
    None,
    'Optional comma-separated task IDs for targeted screening; default evaluates all tasks.',
)

flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_bool(
    'headless_no_renderer',
    False,
    'Disable MuJoCo renderer creation for state-only CPU evaluation (videos unavailable).',
)

flags.DEFINE_bool('eval_only', False, 'Skip gradient updates and evaluate the restored checkpoint once.')
flags.DEFINE_bool(
    'multiswitch',
    False,
    'Add test-time sequence planning, with the released high actor as executor and fallback.',
)
flags.DEFINE_integer('multiswitch_landmarks', 256, 'Number of offline landmark states.')
flags.DEFINE_integer(
    'multiswitch_planner_seed',
    0,
    'Seed for offline landmark construction, independent of evaluation seed.',
)
flags.DEFINE_float(
    'multiswitch_min_route_detour',
    0.0,
    'Use sequence planning only above this graph-length / straight-line-distance ratio.',
)
flags.DEFINE_float(
    'multiswitch_min_route_excess',
    22.0,
    'Use sequence planning only above this route length minus straight-line distance.',
)
flags.DEFINE_integer('multiswitch_candidates', 20_000, 'Dataset states considered by landmark FPS.')
flags.DEFINE_integer('multiswitch_neighbors', 12, 'Spatial neighbors considered per graph vertex.')
flags.DEFINE_integer('multiswitch_max_waypoints', 32, 'Maximum intermediate intentions in a route.')
flags.DEFINE_float('multiswitch_min_reachability', 1e-6, 'Minimum FB reachability retained as an edge.')
flags.DEFINE_float('multiswitch_uncertainty_penalty', 0.5, 'Penalty for disagreement between forward heads.')
flags.DEFINE_float('multiswitch_switch_cost', 0.02, 'Additive cost per intention switch.')
flags.DEFINE_float('multiswitch_waypoint_tolerance', 1.75, 'XY distance used to mark a waypoint reached.')
flags.DEFINE_integer('multiswitch_max_subgoal_steps', 120, 'Replan after this many steps on one intention.')
flags.DEFINE_integer('multiswitch_stall_steps', 40, 'Replan after this many steps without XY progress.')
flags.DEFINE_bool('multiswitch_replan_on_waypoint', False, 'Replan after reaching each waypoint.')
flags.DEFINE_bool('multiswitch_allow_direct_goal', False, 'Include a non-local start-to-goal edge in graph search.')
flags.DEFINE_bool(
    'multiswitch_use_high_actor_for_waypoints',
    True,
    'Execute each planned local goal through the released single-switch high actor.',
)
flags.DEFINE_integer(
    'multiswitch_min_route_waypoints',
    0,
    'Use sequence planning only when the initial route has at least this many landmarks.',
)
flags.DEFINE_integer(
    'multiswitch_route_stride',
    3,
    'Execute every N-th planned landmark while retaining the full route for complexity gating.',
)

config_flags.DEFINE_config_file('agent', 'agents/fbpiswitch.py', lock_config=False)


def main(_):
    # Set up logger.   
    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    if FLAGS.enable_wandb:
        setup_wandb(
            wandb_output_dir=FLAGS.save_dir,
            project='FB pi-Switch', group=FLAGS.wandb_run_group, name=exp_name,
        )
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)
        
    config = FLAGS.agent
    # Set up environment and dataset.
    dataset_kwargs = {}
    if FLAGS.dataset_dir is not None:
        dataset_kwargs['dataset_dir'] = FLAGS.dataset_dir
    eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=config['frame_stack'],
        add_info=True,
        no_renderer=FLAGS.headless_no_renderer,
        **dataset_kwargs,
    )
    eval_env.unwrapped._add_noise_to_goal = False

    if config.get('num_zero_shot_samples') is not None:
        num_zero_shot_samples = config['num_zero_shot_samples']
    else:
        num_zero_shot_samples = 100_000

    train_dataset = Dataset.create(**train_dataset)
    if val_dataset is not None:
        val_dataset = Dataset.create(**val_dataset)
        zero_shot_dataset_dict = val_dataset
    else:
        zero_shot_dataset_dict = train_dataset

    dataset_module = importlib.import_module('utils.datasets')
    dataset_class = getattr(dataset_module, config['dataset_class'])
    train_dataset = dataset_class(train_dataset, config)
    if val_dataset is not None:
        val_dataset = dataset_class(val_dataset, config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # During hierarchical training, copy the frozen representation and actor
    # from a separate checkpoint. A fully restored evaluation checkpoint
    # already contains these modules and does not need this second restore.
    if config['agent_name'] in ["hfb", "fbpiswitch"]:
        if config.get('frozen_path'):
            agent = agent.load_agent_from_frozen(FLAGS, config, example_batch)
        elif FLAGS.restore_path is None:
            raise ValueError(
                'A hierarchical agent needs either --restore_path for a full checkpoint '
                'or --agent.frozen_path for hierarchical post-training.'
            )

    planner = None
    if FLAGS.multiswitch:
        if config['agent_name'] != 'fbpiswitch':
            raise ValueError('--multiswitch currently requires --agent=agents/fbpiswitch.py.')
        raw_dataset = train_dataset.dataset
        dataset_observations = np.asarray(raw_dataset['observations'])
        if 'qpos' in raw_dataset:
            dataset_positions = np.asarray(raw_dataset['qpos'])[:, :2]
        else:
            dataset_positions = dataset_observations[:, :2]
        planner = MultiSwitchPlanner(
            agent,
            dataset_observations,
            dataset_positions,
            PlannerConfig(
                num_landmarks=FLAGS.multiswitch_landmarks,
                landmark_candidates=FLAGS.multiswitch_candidates,
                num_neighbors=FLAGS.multiswitch_neighbors,
                max_waypoints=FLAGS.multiswitch_max_waypoints,
                min_reachability=FLAGS.multiswitch_min_reachability,
                uncertainty_penalty=FLAGS.multiswitch_uncertainty_penalty,
                switch_cost=FLAGS.multiswitch_switch_cost,
                waypoint_tolerance=FLAGS.multiswitch_waypoint_tolerance,
                max_subgoal_steps=FLAGS.multiswitch_max_subgoal_steps,
                stall_steps=FLAGS.multiswitch_stall_steps,
                replan_on_waypoint=FLAGS.multiswitch_replan_on_waypoint,
                allow_direct_goal=FLAGS.multiswitch_allow_direct_goal,
                use_high_actor_for_waypoints=FLAGS.multiswitch_use_high_actor_for_waypoints,
                min_route_waypoints=FLAGS.multiswitch_min_route_waypoints,
                min_route_detour=FLAGS.multiswitch_min_route_detour,
                min_route_excess=FLAGS.multiswitch_min_route_excess,
                route_stride=FLAGS.multiswitch_route_stride,
                seed=FLAGS.multiswitch_planner_seed,
            ),
        )
    
    
    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    loop_steps = 1 if FLAGS.eval_only else FLAGS.train_steps
    for i in tqdm.tqdm(range(1, loop_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        if not FLAGS.eval_only:
            batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] in ['icvf', 'iqlintentions', 'fbpiswitch_nonh']:
                intention_batch = train_dataset.sample(config['batch_size'])
                batch.update(
                    intention_rewards=intention_batch['rewards'],
                    intention_masks=intention_batch['masks'],
                    intention_goals=intention_batch['value_goals'],
                )
                agent, update_info = agent.update(batch)
            else:
                agent, update_info = agent.update(batch)

        # Log metrics.
        if not FLAGS.eval_only and i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'], augmentation=False)
                if config['agent_name'] in ['icvf', 'iqlintentions', 'fbpiswitch_nonh']:
                    val_intention_batch = val_dataset.sample(config['batch_size'], augmentation=False)
                    val_batch.update(
                        intention_rewards=val_intention_batch['rewards'],
                        intention_masks=val_intention_batch['masks'],
                        intention_goals=val_intention_batch['value_goals'],
                    )
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            if FLAGS.enable_wandb:
                wandb.log(train_metrics, step=i)

            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            renders = []
            eval_metrics = {}
            episode_outcomes = []
            overall_metrics = defaultdict(list)
            task_infos = eval_env.unwrapped.task_infos if hasattr(eval_env.unwrapped, 'task_infos') else eval_env.task_infos

            num_tasks = len(task_infos)
            if FLAGS.eval_tasks:
                task_ids = [int(value) for value in FLAGS.eval_tasks.split(',')]
                if any(task_id < 1 or task_id > num_tasks for task_id in task_ids):
                    raise ValueError(f'--eval_tasks must contain IDs in [1, {num_tasks}].')
            else:
                task_ids = range(1, num_tasks + 1)
            for task_id in tqdm.tqdm(task_ids):
                
                if config['agent_name'] not in ["hiql", "iql", "iqlbilinear", "iqlintentions"]:
                    env_name = FLAGS.env_name
                    eval_env.reset(options=dict(task_id=task_id))
                    zero_shot_dataset = relabel_dataset(
                        env_name, 
                        eval_env, 
                        zero_shot_dataset_dict, 
                        complex_task_name=FLAGS.complex_task_name,
                    )
                    zero_shot_dataset = dataset_class(Dataset.create(**zero_shot_dataset), config)
    
                    assert zero_shot_dataset.size >= num_zero_shot_samples
                    zero_shot_batch = zero_shot_dataset.sample(num_zero_shot_samples, 
                                                               idxs=np.arange(num_zero_shot_samples),
                                                               relabeling=False,
                                                               augmentation=False)
                    inferred_latent = agent.infer_latent(zero_shot_batch)
                    inferred_latent = np.asarray(inferred_latent)
            
                else:
                    inferred_latent = None

                eval_info, cur_trajs, cur_renders = evaluate(
                    agent=agent,
                    env=eval_env,
                    task_id=task_id,
                    inferred_latent=inferred_latent,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                    complex_task_name=FLAGS.complex_task_name,
                    planner=planner,
                    eval_seed=FLAGS.seed,
                )
                renders.extend(cur_renders)
                for episode_idx, trajectory in enumerate(cur_trajs):
                    final_info = trajectory['info'][-1]
                    episode_outcomes.append(
                        {
                            'task_id': task_id,
                            'episode': episode_idx,
                            'episode_seed': (
                                int(FLAGS.seed) * 1_000_000
                                + 10_000 * int(task_id)
                                + episode_idx
                            ),
                            'success': float(final_info.get('success', 0.0)),
                        }
                    )
                if FLAGS.complex_task_name=='regions':
                    metric_names = ['total_discounted_return', 'total_return', 'goal_reached', 'in_goal_return', 'to_goal_return', 'in_goal_discounted_return', 'to_goal_discounted_return']
                else:
                    metric_names = ['success']
                metric_names.extend(k for k in eval_info if k.startswith('planner.'))
                eval_metrics.update(
                    {f'evaluation/{task_id}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
                    
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            if FLAGS.enable_wandb:
                wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
            with open(os.path.join(FLAGS.save_dir, 'episode_outcomes.jsonl'), 'w') as f:
                for record in episode_outcomes:
                    f.write(json.dumps(record) + '\n')

        # Save agent.
        if not FLAGS.eval_only and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
