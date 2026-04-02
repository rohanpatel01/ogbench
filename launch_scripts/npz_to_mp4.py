

import numpy as np
import imageio
import os

DATA_FILE = '/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data/distracted_from_downloaded/new/visual-antmaze-medium-stitch-v0-distracted.npz'
OUTPUT_DIR = 'trajectories_videos/distracted_from_downloaded/new'
MAX_VIDEOS = 10




# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("loading data...")
data = np.load(DATA_FILE)
print("done loading data")

obs = data['observations']
# Use terminals or timeouts to find boundaries. 
# In D4RL/AntMaze, terminals or 'done' signals indicate the end of a trajectory.
terminals = data['terminals']
episode_ends = np.where(terminals)[0]

print(f"Total frames: {obs.shape[0]}")
print(f"Number of episodes to process: {len(episode_ends)}")
start_idx = 0

count_videos_generated = 0
for i, end_idx in enumerate(episode_ends):

    if count_videos_generated >= MAX_VIDEOS:
        break
    else:
        count_videos_generated += 1


    # Slice the specific episode
    episode_frames = obs[start_idx : end_idx + 1]
    
    # 1. Handle Channel Ordering (NCHW to NHWC)
    # If shape is (Frames, 3, H, W), transpose to (Frames, H, W, 3)
    if episode_frames.ndim == 4 and episode_frames.shape[1] == 3:
        episode_frames = episode_frames.transpose(0, 2, 3, 1)
    
    # 2. Ensure Data Type is uint8
    if episode_frames.dtype != np.uint8:
        # Assuming float 0.0-1.0 if not uint8
        episode_frames = (episode_frames * 255).clip(0, 255).astype(np.uint8)

    # 3. Save the video
    video_name = os.path.join(OUTPUT_DIR, f'trajectory_{i:04d}.mp4')
    imageio.mimsave(video_name, episode_frames, fps=20)
    
    # Update start_idx for the next loop
    print(f"Saved {video_name} (Frames: {len(episode_frames)})")
    start_idx = end_idx + 1

print(f"\nDone — All videos saved to the '{OUTPUT_DIR}' folder.")