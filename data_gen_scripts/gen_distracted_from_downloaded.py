import numpy as np
from ogbench.utils import make_env_and_datasets, ImageDistractionWrapper
from absl import app, flags
from ml_collections import config_flags
import os
from tqdm import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string('distraction_images_dir', None, 'Directory of DAVIS distraction images.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_list('specific_distractor', None, 'Specific distractor(s) for the evaluation') # --specific_distractor=bear,dog
flags.DEFINE_integer('num_traj', 1, 'Number of trajectories to generate.')


def main(_):


    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    env = ImageDistractionWrapper(env, specific_distractor=FLAGS.specific_distractor, distracting_images_dir=FLAGS.distraction_images_dir)
    env.reset()
    

    # Generate distracted version of dataset for train dataset
    observations = []
    next_observations = []
    MAX_NUM_TRAJECTORIES = FLAGS.num_traj
    save_dir = '/data/rohanp/ogbench/data_gen_scripts/data/single_dog_distractor_dataset'

    train_traj_count = 0
    val_traj_count = 0

    train_bar = tqdm(total=MAX_NUM_TRAJECTORIES)


    for i, (obs, next_obs) in enumerate((zip(train_dataset['observations'], train_dataset['next_observations']))):
        # break   # TODO: just here for debugging - remove later
        obs = env._add_distraction(obs)
        observations.append(obs)

        # Just want to step forward the environment so the next distractor loads
        env.step(train_dataset['actions'][i])    

        next_obs = env._add_distraction(next_obs)
        next_observations.append(next_obs)

        # Note: We reset after we give distraction to next_observation because next_observation (when terminals[i] == 1) will be the last observation
        #       of the traj. So we want to reset after so the distraction remains consistent and doesn't reset at the last frame
    
        # reset so the distractor video resets  
        if train_dataset['terminals'][i]:
            env.reset()
            train_traj_count += 1
            if (train_traj_count >= MAX_NUM_TRAJECTORIES):
                break
            train_bar.update(1)

        
    n_train = len(observations)
    train_dataset['observations'] = np.array(observations)
    train_dataset['next_observations'] = np.array(next_observations)
    for key in train_dataset:
        if key not in ('observations', 'next_observations'):
            train_dataset[key] = train_dataset[key][:n_train]

    assert 'terminals' in train_dataset, "terminals key missing!"
    assert np.sum(train_dataset['terminals'] == 1) > 0, "No terminals in train_dataset!"






    ##################################################################################################################################################################################################################







    # Generate distracted version of dataset for val dataset
    observations_val = []
    next_observations_val = []

    val_bar = tqdm(total=MAX_NUM_TRAJECTORIES)



    for i, (obs, next_obs) in enumerate((zip(val_dataset['observations'], val_dataset['next_observations']))):

        obs = env._add_distraction(obs)
        observations_val.append(obs)

        # ignore output because we just want to augment downloaded dataset with our distractions
        env.step(val_dataset['actions'][i])    # Just want to step forward the environment so the next distractor loads


        next_obs = env._add_distraction(next_obs)
        next_observations_val.append(next_obs)
        
        # reset so the distractor video resets
        if val_dataset['terminals'][i]:
            env.reset()
            val_traj_count += 1
            if (val_traj_count >= MAX_NUM_TRAJECTORIES):
                break
            val_bar.update(1)



    # TODO: make sure to grab the other fields of the dataset based on how many observations we sampled in the above loop
    n_val = len(observations_val)
    val_dataset['observations'] = np.array(observations_val)
    val_dataset['next_observations'] = np.array(next_observations_val)
    for key in val_dataset:
        if key not in ('observations', 'next_observations'):
            val_dataset[key] = val_dataset[key][:n_val]


    assert 'terminals' in val_dataset, "terminals key missing!"
    assert np.sum(val_dataset['terminals'] == 1) > 0, "No terminals in val_dataset!"


    # Save generated distracted dataset
    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, 'visual-antmaze-medium-stitch-v0-distracted.npz'),
        **train_dataset  # unpacks all keys: observations, actions, terminals, next_observations, qpos, qvel, etc.
    )
    np.savez(
        os.path.join(save_dir, 'visual-antmaze-medium-stitch-v0-distracted-val.npz'),
        **val_dataset
    )

    print('Done')


if __name__ == '__main__':
    app.run(main)