import imageio
import numpy as np
from utils.env_utils import make_env_and_datasets
import os

def sample_trajectories_and_test_mp4(acro_dataset, num_trajectories=1, output_dir="check_sample_traj"):
    """
    Sample a single trajectory from an ACRODataset and save it as an mp4.
    """
    os.makedirs(output_dir, exist_ok=True)
 
    num_available_trajs = len(acro_dataset.terminal_locs)
    
    if num_available_trajs == 0:
        print("No trajectories available in dataset")
        return
    
    # Grab a single trajectory
    traj_idx = np.random.randint(0, num_available_trajs)
    
    start = acro_dataset.initial_locs[traj_idx]
    end = acro_dataset.terminal_locs[traj_idx]
    idxs = np.arange(start, end + 1)
 
    observations = acro_dataset.get_observations(idxs)
 
    # Extract last frame if frame stacking is used
    if acro_dataset.config.get('frame_stack') is not None:
        n_stack = acro_dataset.config['frame_stack']
        c_per_frame = observations.shape[-1] // n_stack
        observations = observations[..., -c_per_frame:]
 
    obs = np.array(observations)
    
    # Normalize to uint8
    if obs.dtype != np.uint8:
        obs = np.clip(obs, 0, 1)  # Assuming normalized to [0, 1]
        obs = (obs * 255).astype(np.uint8)
 
    T, H, W, C = obs.shape
 
    # Handle different channel formats
    if C == 1:
        obs = np.repeat(obs, 3, axis=-1)
    elif C == 4:
        obs = obs[..., :3]  # Drop alpha channel
    elif C != 3:
        raise ValueError(f"Unexpected channel count: {C} for trajectory {traj_idx}")
 
    # Save as mp4
    output_path = os.path.join(output_dir, f"trajectory_{traj_idx:06d}.mp4")
    
    try:
        imageio.mimsave(output_path, obs, fps=30, codec='libx264')
        print(f"Saved trajectory {traj_idx} ({T} frames) to {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save video: {e}")