import cv2
import numpy as np
from utils.env_utils import make_env_and_datasets
import os

# def sample_trajectories_and_test_mp4(acro_dataset, num_trajectories=1, output_dir="check_sample_traj"):
def sample_trajectories_and_test_mp4(acro_dataset, num_trajectories=1, output_dir="check_sample_traj"):
    """
    Sample complete trajectories from an ACRODataset and save each as an mp4.
    """
    os.makedirs(output_dir, exist_ok=True)
 
    num_available_trajs = len(acro_dataset.terminal_locs)
    traj_indices = np.random.choice(
        num_available_trajs,
        size=min(num_trajectories, num_available_trajs),
        replace=False
    )
 
    for traj_num, traj_idx in enumerate(traj_indices):
        start = acro_dataset.initial_locs[traj_idx]
        end = acro_dataset.terminal_locs[traj_idx]
        idxs = np.arange(start, end + 1)  # Assumes terminal_locs is inclusive
 
        observations = acro_dataset.get_observations(idxs)
 
        if acro_dataset.config.get('frame_stack') is not None:
            n_stack = acro_dataset.config['frame_stack']
            c_per_frame = observations.shape[-1] // n_stack
            observations = observations[..., -c_per_frame:]
 
        obs = np.array(observations)
        if obs.dtype != np.uint8:
            obs = ((obs - obs.min()) / (obs.max() - obs.min() + 1e-8) * 255).astype(np.uint8)
 
        T, H, W, C = obs.shape
 
        # Handle channel count before any further processing
        if C == 1:
            obs = np.repeat(obs, 3, axis=-1)
        elif C == 4:
            obs = obs[..., :3]  # Drop alpha channel
        elif C != 3:
            raise ValueError(f"Unexpected channel count: {C} for trajectory {traj_idx}")
 
        # Ensure even dimensions for mp4v codec
        if H % 2 != 0:
            H = H + 1
            obs = np.pad(obs, ((0, 0), (0, 1), (0, 0), (0, 0)))
        if W % 2 != 0:
            W = W + 1
            obs = np.pad(obs, ((0, 0), (0, 0), (0, 1), (0, 0)))
 
        output_path = os.path.join(output_dir, f"trajectory_{traj_idx:06d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H))
 
        if not writer.isOpened():
            print(f"ERROR: Failed to open VideoWriter for {output_path}")
            continue
 
        for frame in obs:
            writer.write(frame[..., ::-1])  # RGB -> BGR for OpenCV
 
        writer.release()
        print(f"Saved trajectory {traj_idx} ({T} frames) to {output_path}")



# def trajectory_to_mp4(acro_dataset, output_path="trajectory.mp4", traj_idx=None, fps=30):
#     """
#     Render a single trajectory's observations to an mp4.
    
#     Args:
#         acro_dataset: Your ACRODataset instance
#         output_path: Where to save the .mp4
#         traj_idx: Which trajectory to render (random if None)
#         fps: Frames per second
#     """
#     # Pick a trajectory
#     if traj_idx is None:
#         traj_idx = np.random.randint(len(acro_dataset.terminal_locs))

#     start = acro_dataset.initial_locs[traj_idx]
#     end = acro_dataset.terminal_locs[traj_idx]
#     idxs = np.arange(start, end + 1)

#     # Use get_observations so frame stacking is handled correctly
#     observations = acro_dataset.get_observations(idxs)  # shape: (T, H, W, C) or (T, H, W, stack*C)

#     # If frame-stacked (e.g. 4 grayscale frames concatenated on channel dim),
#     # just take the last frame's channels
#     if acro_dataset.config['frame_stack'] is not None:
#         n_stack = acro_dataset.config['frame_stack']
#         # observations shape: (T, H, W, stack*C) — grab only the last frame
#         c_per_frame = observations.shape[-1] // n_stack
#         observations = observations[..., -c_per_frame:]  # (T, H, W, C)

#     # Normalize to uint8 if needed
#     obs = np.array(observations)
#     if obs.dtype != np.uint8:
#         obs = ((obs - obs.min()) / (obs.max() - obs.min() + 1e-8) * 255).astype(np.uint8)

#     T, H, W, C = obs.shape

#     # If grayscale, convert to BGR for OpenCV
#     if C == 1:
#         obs = np.repeat(obs, 3, axis=-1)

#     # Write the video
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

#     for frame in obs:
#         writer.write(frame[..., ::-1])  # RGB -> BGR for OpenCV

#     writer.release()
#     print(f"Saved {T} frames to {output_path}")



