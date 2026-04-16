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


config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)

# DEBUG_MAX_STEPS = 400

def main(_):


    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name) # , frame_stack=config['frame_stack']

    env = ImageDistractionWrapper(env, specific_distractor=FLAGS.specific_distractor, distracting_images_dir=FLAGS.distraction_images_dir)
    env.reset()
    

    # Generate distracted version of dataset for train dataset
    observations = []
    next_observations = []
    MAX_NUM_TRAJECTORIES = 256  # 1 batch
    train_traj_count = 0
    val_traj_count = 0

    train_bar = tqdm(total=MAX_NUM_TRAJECTORIES)

    while (train_traj_count < MAX_NUM_TRAJECTORIES):
        for i, (obs, next_obs) in enumerate((zip(train_dataset['observations'], train_dataset['next_observations']))):
            # break   # TODO: just here for debugging - remove later
            obs = env._add_distraction(obs)
            observations.append(obs)

            # ignore output because we just want to augment downloaded dataset with our distractions
            env.step(train_dataset['actions'][i])    # Just want to step forward the environment so the next distractor loads

            next_obs = env._add_distraction(next_obs)
            next_observations.append(next_obs)

            # Note: We reset after we give distraction to next_observation because next_observation (when terminals[i] == 1) will be the last observation
            #       of the traj. So we want to reset after so the distraction remains consistent and doesn't reset at the last frame
        
            # reset so the distractor video resets  
            if train_dataset['terminals'][i]:
                env.reset()
                train_traj_count += 1
                train_bar.update(1)
                # if train_traj_count >= MAX_NUM_TRAJECTORIES:  # TODO: I just want a small dataset that I can debug

        
    # TODO: make sure to grab the other fields of the dataset based on how many observations we sampled in the above loop
    n_train = len(observations)
    train_dataset['observations'] = np.array(observations)
    train_dataset['next_observations'] = np.array(next_observations)
    for key in train_dataset:
        if key not in ('observations', 'next_observations'):
            train_dataset[key] = train_dataset[key][:n_train]

    assert 'terminals' in train_dataset, "terminals key missing!"
    assert np.sum(train_dataset['terminals'] == 1) > 0, "No terminals in train_dataset!"



    # Generate distracted version of dataset for val dataset
    observations_val = []
    next_observations_val = []

    val_bar = tqdm(total=MAX_NUM_TRAJECTORIES)


    while (val_traj_count < MAX_NUM_TRAJECTORIES):
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
                val_bar.update(1)
                # if val_traj_count >= MAX_NUM_TRAJECTORIES:  # TODO: I just want a small dataset that I can debug
                #     break  


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
    output_dir = '/data/rohanp/ogbench/data_gen_scripts/data/small_bear_distracted_dataset'
    os.makedirs(output_dir, exist_ok=True)


    # TODO: Just for debugging
    # for key in train_dataset:
    #     train_dataset[key] = train_dataset[key][:DEBUG_MAX_STEPS]

    

    np.savez(
        os.path.join(output_dir, 'visual-antmaze-medium-stitch-v0-distracted.npz'),
        **train_dataset  # unpacks all keys: observations, actions, terminals, next_observations, qpos, qvel, etc.
    )
    np.savez(
        os.path.join(output_dir, 'visual-antmaze-medium-stitch-v0-distracted-val.npz'),
        **val_dataset
    )

    print('Done')


if __name__ == '__main__':
    app.run(main)
