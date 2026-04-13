from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
import wandb
from absl import flags
FLAGS = flags.FLAGS

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def L2_Rewards_from_acro(acro_agent, train_dataset, val_dataset):
    
    results = {}
    
    for dataset_name, dataset in [('train', train_dataset), ('val', val_dataset)]:
        
        breakpoint()

        all_l2_distances = []
        
        batch_size = acro_agent.config['batch_size']

        # TODO: Note: dataset.sample(batch_size) does NOT return a trajecotry. It just returns a batch of observations.
        #       We need to look into how we can get a trajectory
        batch_traj = dataset.sample(batch_size)
        observations = batch_traj['observations']  # Shape: (batch_size, H, W, C)
        
        encoded_obs = acro_agent.network.select('encoder')(observations)    # should be [batch_size, traj_len, latent_dim]
        
        # Get the last observation of each trajectory and replicate it
        # Assuming trajectories are concatenated, get last index of each trajectory
        # You may need to adjust this based on how your dataset structures batch samples
        last = encoded_obs[:, -1:, :]
        encoded_last_obs = jax.numpy.broadcast_to(last, encoded_obs.shape)  # should be [batch_size, traj_len, latent_dim]


        # last_obs_encoded = encoded_obs[-batch_size:]  # Last observation from each trajectory
        # last_obs_replicated = np.repeat(last_obs_encoded, len(encoded_obs) // batch_size, axis=0)  # Shape: (batch_size * traj_len, latent_dim)
        
        # Compute -L2 distance from each observation to the corresponding trajectory's last observation
        l2_distances = -np.linalg.norm(encoded_obs - encoded_last_obs, axis=1)  # Shape: (batch_size * traj_len,)
        
        # Compute statistics
        mean_dist = np.mean(l2_distances)
        std_dist = np.std(l2_distances)
        min_dist = np.min(l2_distances)
        max_dist = np.max(l2_distances)
        
        # Log statistics
        wandb.log({
            f'acro_eval/{dataset_name}_l2_mean': mean_dist,
            f'acro_eval/{dataset_name}_l2_std': std_dist,
            f'acro_eval/{dataset_name}_l2_min': min_dist,
            f'acro_eval/{dataset_name}_l2_max': max_dist,
        })
        
        results[dataset_name] = {
            'all_distances': l2_distances,
            'mean': mean_dist,
            'std': std_dist,
            'min': min_dist,
            'max': max_dist,
        }
    
    return results
    



def evaluate_acro(acro_agent, train_dataset, val_dataset):

    results = {}
    results.update(L2_Rewards_from_acro(acro_agent, train_dataset, val_dataset))



    





def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        # print(f"[eval] observation shape: {observation.shape}, dtype: {observation.dtype}")

        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')

        done = False
        step = 0
        render = []
        while not done:

            # TODO: Need to pass observations and goal through ACRO encoder before passing through actor_fn?

            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):

                frame = env.render().copy()

                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
