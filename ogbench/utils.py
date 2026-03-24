import os
import urllib.request

import gymnasium
import numpy as np
from gymnasium.spaces import Box
from PIL import Image
from tqdm import tqdm

from ogbench.relabel_utils import add_oracle_reps, relabel_dataset



DEFAULT_DATASET_DIR = '/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data' # Changed to match where my data is.   Orig: ~/.ogbench/data
DATASET_URL = 'https://rail.eecs.berkeley.edu/datasets/ogbench'

def load_dataset(dataset_path, ob_dtype=np.float32, action_dtype=np.float32, compact_dataset=False, add_info=False):
    """Load OGBench dataset.

    Args:
        dataset_path: Path to the dataset file.
        ob_dtype: dtype for observations.
        action_dtype: dtype for actions.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').
        add_info: Whether to add observation information ('qpos', 'qvel', and 'button_states') to the dataset.

    Returns:
        Dictionary containing the dataset. The dictionary contains the following keys: 'observations', 'actions',
        'terminals', and 'next_observations' (if `compact_dataset` is False) or 'valids' (if `compact_dataset` is True).
        If `add_info` is True, the dictionary may also contain additional keys for observation information.
    """
    file = np.load(dataset_path)

    dataset = dict()
    for k in ['observations', 'actions', 'terminals']:
        if k == 'observations':
            dtype = ob_dtype
        elif k == 'actions':
            dtype = action_dtype
        else:
            dtype = np.float32
        dataset[k] = file[k][...].astype(dtype, copy=False)

    if add_info:
        # Read observation information.
        info_keys = []
        for k in ['qpos', 'qvel', 'button_states']:
            if k in file:
                dataset[k] = file[k][...]
                info_keys.append(k)

    # Example:
    # Assume each trajectory has length 4, and (s0, a0, s1), (s1, a1, s2), (s2, a2, s3), (s3, a3, s4) are transition
    # tuples. Note that (s4, a4, s0) is *not* a valid transition tuple, and a4 does not have a corresponding next state.
    # At this point, `dataset` loaded from the file has the following structure:
    #                  |<--- traj 1 --->|  |<--- traj 2 --->|  ...
    # -------------------------------------------------------------
    # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
    # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
    # 'terminals'   : [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  1, ...]

    if compact_dataset:
        # Compact dataset: We need to invalidate the last state of each trajectory so that we can safely get
        # `next_observations[t]` by using `observations[t + 1]`.
        # Our goal is to have the following structure:
        #                  |<--- traj 1 --->|  |<--- traj 2 --->|  ...
        # -------------------------------------------------------------
        # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
        # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
        # 'terminals'   : [ 0,  0,  0,  1,  1,  0,  0,  0,  1,  1, ...]
        # 'valids'      : [ 1,  1,  1,  1,  0,  1,  1,  1,  1,  0, ...]

        dataset['valids'] = 1.0 - dataset['terminals']
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0).astype(np.float32)
    else:
        # Regular dataset: Generate `next_observations` by shifting `observations`.
        # Our goal is to have the following structure:
        #                       |<- traj 1 ->|  |<- traj 2 ->|  ...
        # ----------------------------------------------------------
        # 'observations'     : [s0, s1, s2, s3, s0, s1, s2, s3, ...]
        # 'actions'          : [a0, a1, a2, a3, a0, a1, a2, a3, ...]
        # 'next_observations': [s1, s2, s3, s4, s1, s2, s3, s4, ...]
        # 'terminals'        : [ 0,  0,  0,  1,  0,  0,  0,  1, ...]

        ob_mask = (1.0 - dataset['terminals']).astype(bool)
        next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
        dataset['next_observations'] = dataset['observations'][next_ob_mask]
        dataset['observations'] = dataset['observations'][ob_mask]
        dataset['actions'] = dataset['actions'][ob_mask]
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)

        if add_info:
            for k in info_keys:
                dataset[k] = dataset[k][ob_mask]

    return dataset


def download_datasets(dataset_names, dataset_dir=DEFAULT_DATASET_DIR):
    """Download OGBench datasets.

    Args:
        dataset_names: List of dataset names to download.
        dataset_dir: Directory to save the datasets.
    """
    # Make dataset directory.
    dataset_dir = os.path.expanduser(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download datasets.
    dataset_file_names = []
    for dataset_name in dataset_names:
        dataset_file_names.append(f'{dataset_name}.npz')
        dataset_file_names.append(f'{dataset_name}-val.npz')
    for dataset_file_name in dataset_file_names:
        dataset_file_path = os.path.join(dataset_dir, dataset_file_name)
        # breakpoint()
        if not os.path.exists(dataset_file_path):
            dataset_url = f'{DATASET_URL}/{dataset_file_name}'
            print('Downloading dataset from:', dataset_url)
            response = urllib.request.urlopen(dataset_url)
            tmp_dataset_file_path = f'{dataset_file_path}.tmp'
            with tqdm.wrapattr(
                open(tmp_dataset_file_path, 'wb'),
                'write',
                miniters=1,
                desc=dataset_url.split('/')[-1],
                total=getattr(response, 'length', None),
            ) as file:
                for chunk in response:
                    file.write(chunk)
            os.rename(tmp_dataset_file_path, dataset_file_path)


def make_env_and_datasets(
    dataset_name,
    dataset_dir=DEFAULT_DATASET_DIR,
    dataset_path=None,
    compact_dataset=False,
    env_only=False,
    dataset_only=False,
    cur_env=None,
    add_info=False,
    **env_kwargs,
):
    """Make OGBench environment and load datasets.

    Args:
        dataset_name: Dataset name.
        dataset_dir: Directory to save/load the datasets.
        dataset_path: (Optional) Path to the dataset file.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').
        env_only: Whether to return only the environment.
        dataset_only: Whether to return only the datasets.
        cur_env: Current environment (only used when `dataset_only` is True).
        add_info: Whether to add observation information ('qpos', 'qvel', and 'button_states') to the datasets.
        **env_kwargs: Keyword arguments to pass to the environment.
    """
    # Make environment.
    splits = dataset_name.split('-')
    dataset_add_info = add_info
    env = cur_env
    # breakpoint()    #TODO
    if 'singletask' in splits:
        # Single-task environment.
        pos = splits.index('singletask')
        env_name = '-'.join(splits[: pos - 1] + splits[pos:])  # Remove the dataset type.
        if not dataset_only:
            env = gymnasium.make(env_name, **env_kwargs)
        dataset_name = '-'.join(splits[:pos] + splits[-1:])  # Remove the words 'singletask' and 'task\d' (if exists).
        dataset_add_info = True
    elif 'oraclerep' in splits:
        # Environment with oracle goal representations.
        env_name = '-'.join(splits[:-3] + splits[-1:])  # Remove the dataset type and the word 'oraclerep'.
        if not dataset_only:
            env = gymnasium.make(env_name, use_oracle_rep=True, **env_kwargs)
        dataset_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the word 'oraclerep'.
        dataset_add_info = True
    else:
        # Original, goal-conditioned environment.
        env_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the dataset type.
        if not dataset_only:
            env = gymnasium.make(env_name, **env_kwargs)

    if env_only:
        return env

    # Load datasets.
    if dataset_path is None:
        dataset_dir = os.path.expanduser(dataset_dir)
        download_datasets([dataset_name], dataset_dir)
        train_dataset_path = os.path.join(dataset_dir, f'{dataset_name}.npz')
        val_dataset_path = os.path.join(dataset_dir, f'{dataset_name}-val.npz')
    else:
        train_dataset_path = dataset_path
        val_dataset_path = dataset_path.replace('.npz', '-val.npz')

    ob_dtype = np.uint8 if ('visual' in env_name or 'powderworld' in env_name) else np.float32
    action_dtype = np.int32 if 'powderworld' in env_name else np.float32
    train_dataset = load_dataset(
        train_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=dataset_add_info,
    )
    val_dataset = load_dataset(
        val_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=dataset_add_info,
    )

    if 'singletask' in splits:
        # Add reward information to the datasets.
        relabel_dataset(env_name, env, train_dataset)
        relabel_dataset(env_name, env, val_dataset)

    if 'oraclerep' in splits:
        # Add oracle goal representations to the datasets.
        add_oracle_reps(env_name, env, train_dataset)
        add_oracle_reps(env_name, env, val_dataset)

    if not add_info:
        # Remove information keys.
        for k in ['qpos', 'qvel', 'button_states']:
            if k in train_dataset:
                del train_dataset[k]
            if k in val_dataset:
                del val_dataset[k]

    if dataset_only:
        return train_dataset, val_dataset
    else:
        return env, train_dataset, val_dataset


class ImageDistractionWrapper(gymnasium.Wrapper):
    """Wrapper that adds distracting video from image sequences placed beside the observation.

    Loads image sequences from folders inside distracting_images/, scales them to match the
    observation size, and concatenates them horizontally (original | distraction). The
    distraction advances one frame per environment step, creating a video effect.
    """

    def __init__(
        self,
        env,
        distracting_images_dir=None,
        position='right',
        difficulty='easy',
        specific_distractor=None,

    ):
        """Initialize the wrapper.

        Args:
            env: Environment with visual (image) observations.
            distracting_images_dir: Path to the distracting_images folder. Defaults to
                <package_root>/distracting_images.
            folder_names: List of subfolder names to use (e.g. ['bike-packing', 'bus']).
                If None, discovers all subfolders and randomly picks one per episode.
            position: Where to place the distraction: 'right' or 'left'.
            difficulty: Difficulty of the distraction: 'easy', 'medium', or 'hard'.
        """
        super().__init__(env)
        if distracting_images_dir is None:
            pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            distracting_images_dir = os.path.join(pkg_root, 'distracting_images')

        self._folder_names_difficulty_map={
            'easy': ['bike-packing', 'bear', 'blackswan'],
            'medium': [ "bear", "bus", "crossing", "dogs-scale", "hike", "koala", "mallard-water", "parkour", "scooter-gray", "surf",
                        "bike-packing", "camel", "dance-jump", "drift-chicane", "hockey", "lab-coat", "mbike-trick", "pigs", "sheep", "swing",
            ],
            'hard': [   "bear", "bus", "crossing", "dogs-scale", "hike", "koala", "mallard-water", "parkour", "scooter-gray", "surf",
                        "bike-packing", "camel", "dance-jump", "drift-chicane", "hockey", "lab-coat", "mbike-trick", "pigs", "sheep", "swing",
                        "blackswan", "car-roundabout", "dance-twirl", "drift-straight", "horsejump-high", "lady-running", "miami-surf", "planes-water", "shooting", "tennis",
                        "bmx-bumps", "car-shadow", "dancing", "drift-turn", "horsejump-low", "libby", "motocross-bumps", "rallye", "skate-park", "tractor-sand",
                        "bmx-trees", "car-turn", "disc-jockey", "drone", "india", "lindy-hop", "motocross-jump", "rhino", "snowboard", "train",
                        "boat", "cat-girl", "dog", "elephant", "judo", "loading", "motorbike", "rollerblade", "soapbox", "tuk-tuk",
                        "boxing-fisheye", "classic-car", "dog-agility", "flamingo", "kid-football", "longboard", "night-race", "schoolgirls", "soccerball", "upside-down",
                        "breakdance", "color-run", "dog-gooses", "goat", "kite-surf", "lucia", "paragliding", "scooter-black", "stroller", "varanus-cage",
                        "breakdance-flare", "cows", "dogs-jump", "gold-fish", "kite-walk", "mallard-fly", "paragliding-launch", "scooter-board", "stunt", "walking"
            ]
        }
        self._distracting_images_dir = os.path.expanduser(distracting_images_dir)

        if specific_distractor is None:
            self._folder_names = self._folder_names_difficulty_map[difficulty]
        else:
            self._folder_names = [specific_distractor]

        self._position = position
        self._image_sequences = {}  # folder_name -> list of (H, W, 3) arrays
        self._current_folder = None
        self._current_frame_idx = 0
        self._playback_direction = 1  # +1 forward, -1 backward
        self._ob_shape = None
        self._observation_space = None
        for folder_name in self._folder_names:
            self._load_sequence(folder_name)

    def _discover_folders(self):
        """Discover available subfolders in distracting_images_dir."""
        if not os.path.isdir(self._distracting_images_dir):
            return []
        folders = []
        for name in sorted(os.listdir(self._distracting_images_dir)):
            path = os.path.join(self._distracting_images_dir, name)
            if os.path.isdir(path):
                folders.append(name)
        return folders

    def _load_sequence(self, folder_name):
        """Load image sequence from a folder. Returns list of (H, W, 3) uint8 arrays."""
        if folder_name in self._image_sequences:
            return self._image_sequences[folder_name]
        folder_path = os.path.join(self._distracting_images_dir, folder_name)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f'Distraction folder not found: {folder_path}')
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = []
        for f in os.listdir(folder_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                files.append(f)
        files.sort()
        if not files:
            raise ValueError(f'No images found in {folder_path}')
        frames = []
        for f in files:
            path = os.path.join(folder_path, f)
            img = Image.open(path).convert('RGB')
            frames.append(np.array(img))
        self._image_sequences[folder_name] = frames
        return frames

    def _get_distraction_frame(self, target_h, target_w):
        """Get current distraction frame scaled to (target_h, target_w, 3)."""
        if self._current_folder is None:
            return None
        frames = self._image_sequences[self._current_folder]
        frame = frames[self._current_frame_idx]
        img = Image.fromarray(frame)
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return np.array(img)

    def _add_distraction(self, ob):
        """Concatenate distraction video to the side of the observation."""
        if not isinstance(ob, np.ndarray) or ob.ndim < 2:
            return ob
        if ob.dtype != np.uint8 or ob.shape[-1] not in (3, 6):
            return ob
        h, w = ob.shape[:2]
        n_channels = ob.shape[-1]
        # Distraction is always 3-channel; we scale to match observation size
        dist_frame = self._get_distraction_frame(h, w)
        if dist_frame is None:
            return ob
        if n_channels == 6:
            dist_frame = np.concatenate([dist_frame, dist_frame], axis=-1)
        if self._position == 'right':
            ob = np.concatenate([ob, dist_frame], axis=1)
        else:
            ob = np.concatenate([dist_frame, ob], axis=1)
        return ob

    def _update_observation_space(self, ob_shape):
        """Update observation space to reflect new width."""
        if self._ob_shape == ob_shape:
            return
        self._ob_shape = ob_shape
        h, w, c = ob_shape
        new_w = 2 * w
        self._observation_space = Box(low=0, high=255, shape=(h, new_w, c), dtype=np.uint8)

    def reset(self, *args, **kwargs):
        folders = self._folder_names if self._folder_names else self._discover_folders()
        if not folders:
            raise ValueError(
                f'No distraction folders found in {self._distracting_images_dir}. '
                'Set folder_names or add subfolders with images.'
            )
        folder_idx = self.env.np_random.integers(0, len(folders))
        self._current_folder = folders[folder_idx]
        frames = self._load_sequence(self._current_folder)
        self._current_frame_idx = int(self.env.np_random.integers(0, len(frames)))
        self._playback_direction = 1
        ob, info = self.env.reset(*args, **kwargs)
        if isinstance(ob, np.ndarray) and ob.ndim >= 2 and ob.dtype == np.uint8:
            self._update_observation_space(ob.shape)
            ob = self._add_distraction(ob)

        # Added this so during evaluation we also apply distraction to goal
        if 'goal' in info and isinstance(info['goal'], np.ndarray):
            info['goal'] = self._add_distraction(info['goal'])
            
        return ob, info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        if self._current_folder is not None:
            n = len(self._image_sequences[self._current_folder])
            self._current_frame_idx += self._playback_direction
            if self._current_frame_idx >= n:
                self._current_frame_idx = n - 2
                self._playback_direction = -1
            elif self._current_frame_idx < 0:
                self._current_frame_idx = 1
                self._playback_direction = 1
        if isinstance(ob, np.ndarray) and ob.ndim >= 2 and ob.dtype == np.uint8:
            ob = self._add_distraction(ob)
        return ob, reward, terminated, truncated, info

    @property
    def observation_space(self):
        if self._observation_space is not None:
            return self._observation_space
        return self.env.observation_space