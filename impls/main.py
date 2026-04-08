import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents.acro import ACROAgent
from agents import acro

from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset, ACRODataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb


from utils.encoders import get_acro_encoder

from ogbench.utils import ImageDistractionWrapper


FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'ogbench', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('steps_pre_train_acro', 100000, 'Number of steps to pre-train ACRO encoder.')

flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)


flags.DEFINE_string('dataset_path_train', None, 'Path to dataset for train.')
flags.DEFINE_string('dataset_path_val', None, 'Path to dataset for val.')
flags.DEFINE_integer('using_distractions_dataset', 0, 'Determines whether we use the distraction env or normal env')
flags.DEFINE_integer('use_acro_rep', 0, 'Whether to use ACRO as representation or to use state.')
flags.DEFINE_integer('use_acro_for_reward', 0, 'Whether to use ACRO as reward.')

flags.DEFINE_string('exp_name', "Default_Exp_Name", 'Name the experiment will show on WANDB')
flags.DEFINE_string('acro_restore_path', None, 'Path to saved ACRO model weights.')


def main(_):
    # Set up logger.
    exp_name = FLAGS.exp_name

    setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent

    if ((FLAGS.dataset_path_train) and (FLAGS.dataset_path_val)): # Use our dataset for training
        print("Using specified data")
        env, _, _ = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'], dataset_path=FLAGS.dataset_path_train)

        data_train = np.load(FLAGS.dataset_path_train)
        data_val = np.load(FLAGS.dataset_path_val)
        train_dataset = dict(data_train)   # keys: observations, actions, rewards, terminals, etc.
        # val_dataset = dict(data_val)           # or split off a slice if you want validation
        val_dataset = None # making None because val_dataset has issue where data['terminals'][-1] errors out <-- need to figure out why but in mean time we can just try training with train dataset only

    else:
        # Allow their code to download the dataset corresponding to the specified env_name
        print("Data NOT specified. Going to use auto-download OGBench data based on env_name")

        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])


    # Make sure to use distracted env if we are training on the distraction dataset
    # This is important because we need the observation space to match that of the data we're using
    if (FLAGS.using_distractions_dataset):
        env = ImageDistractionWrapper(
            env,
            distracting_images_dir='/work/10993/rohanpatel01/vista/DAVIS/JPEGImages/480p/',
        )


    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'ACRODataset': ACRODataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)


    # # see what shape the train_dataset elements are. We might have a bug here because next_v_t in HIQL algo is mishaped.
    # # next_v_t should be (1024, ) I THINK but is really (1024, 64, 128). Not sure which is correct but I think should be (1024, ) bc that's the shape of the other stuff
    # #       that next_v_t is being operated with
    # batch = train_dataset.sample(4)
    # for k, v in batch.items():
    #     print(k, v.shape, v.dtype)

    acro_val_dataset = None

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Pre-train ACRO
    if FLAGS.use_acro_for_reward or FLAGS.use_acro_rep:
        acro_config = acro.get_config()
        example_batch = train_dataset.sample(1)

        # breakpoint()
        if config['discrete']:
            # Fill with the maximum action to let the agent know the action space size.
            example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

        train_dataset_acro = dict(data_train)
        val_dataset_acro = dict(data_val)

        acro_train_dataset = ACRODataset(Dataset.create(**train_dataset_acro), acro_config)
        if val_dataset is not None:
            acro_val_dataset = ACRODataset(Dataset.create(**val_dataset_acro), acro_config)

        # Load pretrained ACRO
        # TODO: Note just pulling the acro_config will only load the deafult values we see in acro.py:get_config()
        #       rather than pulling whatever values we would've overriden. So we need to come back and fix this
        acro_agent = ACROAgent.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], acro_config)

        if FLAGS.acro_restore_path:
            acro_agent = restore_agent(acro_agent, FLAGS.acro_restore_path)
        else:
            # Train ACRO
            train_loop(acro_agent, acro_train_dataset, acro_val_dataset, acro_config, env, step_offset=0)
            print("ACRO done pre-training")
        # Extract encoder definition and params
        # acro_encoder = acro_agent.network.select('encoder')
        acro_encoder = get_acro_encoder(acro_agent)

    else:
        acro_encoder = None


    # Done Pre-training ACRO, Run actual algorithm

    # Pass in ACRO's encoder to the algorithm we want to run
    # ex: HIQL
    example_batch = train_dataset.sample(1)

    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    if FLAGS.use_acro_for_reward or FLAGS.use_acro_rep:
        agent = agent_class.create(
            FLAGS.seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
            acro_encoder=acro_encoder,
        )
    else:
        agent = agent_class.create(
            FLAGS.seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # train HIQL
    if FLAGS.use_acro_for_reward or FLAGS.use_acro_rep:
        train_loop(agent, train_dataset, val_dataset, config, env, step_offset=FLAGS.steps_pre_train_acro)
    else:
        train_loop(agent, train_dataset, val_dataset, config, env)
    print("Done training HIQL")

def train_loop(agent, train_dataset, val_dataset, config, env, step_offset=0):
    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()


    if agent.config['agent_name'] == 'acro':
        # Set number of timesteps to pre-train ACRO
        train_steps = FLAGS.steps_pre_train_acro

    else:
        # train_steps to train actual agent
        train_steps = FLAGS.train_steps

    for i in tqdm.tqdm(range(1, train_steps + 1), smoothing=0.1, dynamic_ncols=True):
    # for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        global_step = step_offset + i
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])

        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=global_step)
            train_logger.log(train_metrics, step=global_step)

        # Evaluate agent. But do not evaluate when we are pre-training the ACRO encoder
        if (agent.config['agent_name'] != 'acro') and (i == 1 or i % FLAGS.eval_interval == 0):
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=global_step)
            eval_logger.log(eval_metrics, step=global_step)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()






if __name__ == '__main__':
    app.run(main)

