

import numpy as np
import imageio

DATA_FILE='../data_gen_scripts/data/visual-antmaze-medium-stitch-v0-distracted.npz'


data = np.load(DATA_FILE)
obs = data['observations']

print("Observations shape:", obs.shape)
print("Dtype:", obs.dtype)

terminals = data['terminals']
episode_ends = np.where(terminals)[0]
print(f"Number of episodes: {len(episode_ends)}")
print(f"First episode end index: {episode_ends[0]}")

episode_frames = obs[:episode_ends[0] + 1]
print(f"Episode frames shape: {episode_frames.shape}")

# Fix channel order if needed
if episode_frames.ndim == 4 and episode_frames.shape[1] in (1, 3):
    print("Transposing from channels-first to channels-last")
    episode_frames = episode_frames.transpose(0, 2, 3, 1)

print(f"Final frames shape: {episode_frames.shape}")
print(f"Pixel value range: {episode_frames.min()} - {episode_frames.max()}")

# Ensure uint8
if episode_frames.dtype != np.uint8:
    print(f"Converting from {episode_frames.dtype} to uint8")
    episode_frames = (episode_frames * 255).clip(0, 255).astype(np.uint8)

imageio.mimsave('sample_episode.mp4', episode_frames, fps=60)
print("Done — saved to sample_episode.mp4")