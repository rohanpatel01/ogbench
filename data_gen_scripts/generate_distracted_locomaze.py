import glob
import json
import pathlib
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from agents import SACAgent
from tqdm import trange
from utils.evaluation import supply_rng
from utils.flax_utils import restore_agent

import ogbench.locomaze  # noqa

from ogbench.utils import ImageDistractionWrapper

FLAGS = flags.FLAGS

# Note: I changed the default values here to match the ones OGBench used to generate their data for visual-antmaze-medium-stitch-v0
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'visual-antmaze-medium', 'Environment name.')          # visual-antmaze-medium
flags.DEFINE_string('dataset_type', 'stitch', 'Dataset type.')                                  # stitch
flags.DEFINE_string('restore_path', 'experts/ant', 'Expert agent restore path.')
flags.DEFINE_integer('restore_epoch', 400000, 'Expert agent restore epoch.')
flags.DEFINE_float('noise', 0.2, 'Gaussian action noise level.')
flags.DEFINE_integer('num_episodes', 5000, 'Number of episodes.')                               # 5000     # TODO: change back once we get distractions working correctly
flags.DEFINE_integer('max_episode_steps', 200, 'Maximum number of steps in an episode.')        # 200

# Flags for distraction wrapper
flags.DEFINE_string('distraction', 'none', 'Distraction wrapper: none, circle, or image.')
flags.DEFINE_string('distraction_images_dir', None, 'Path to distracting_images folder (image wrapper). Default: <repo>/distracting_images.')
flags.DEFINE_string('distraction_difficulty', 'easy', 'Difficulty of the distraction: easy, medium, or hard.')
flags.DEFINE_string('save_dir', None, 'Save path.')
flags.DEFINE_integer('save_period', 1000000, 'Defines after how many episodes generated to save them')
flags.DEFINE_string('save_file_name', 'none', 'Name of the .npz file that will contain the generated data')




def main(_):
    assert FLAGS.dataset_type in ['path', 'navigate', 'stitch', 'explore']
    # 'path': Reach a single goal and stay there.
    # 'navigate': Repeatedly reach randomly sampled goals in a single episode.
    # 'stitch': Reach a nearby goal that is 4 cells away and stay there.
    # 'explore': Repeatedly follow random directions sampled every 10 steps.

    # TODO: Read contents of command line and populate flags based on what we want
    # For now I'll just hardcode the flags above just to get progress on adding distractions and such to this dataset

    print("Starting main")
    print("env_name: ", FLAGS.env_name)
    print("dataset_type: ", FLAGS.dataset_type)
    print("num_episodes: ", FLAGS.num_episodes)

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        max_episode_steps=FLAGS.max_episode_steps,
    )

    ob_dim = env.observation_space.shape[0]

    distracted_envs = None
    if FLAGS.distraction != 'none':
        if 'visual' not in FLAGS.env_name:
            raise ValueError(
                f'Distraction wrappers require a visual environment (e.g. visual-cube-single-v0). '
                f'Got env_name={FLAGS.env_name}.'
            )
        if FLAGS.distraction == 'image':
            distracting_dir = FLAGS.distraction_images_dir
            if distracting_dir is None:
                distracting_dir = pathlib.Path(__file__).resolve().parent.parent / 'distracting_images' # default distraction dir?
            else:
                # Setup Distracted environments we'll use during data collection
                distracted_envs = {
                    'left': ImageDistractionWrapper(
                                env,
                                distracting_images_dir='/work/10993/rohanpatel01/vista/DAVIS/JPEGImages/480p/',
                                difficulty=FLAGS.distraction_difficulty,
                                specific_distractor='bear'
                            ),

                    'right': ImageDistractionWrapper(
                                env,
                                distracting_images_dir='/work/10993/rohanpatel01/vista/DAVIS/JPEGImages/480p/',
                                difficulty=FLAGS.distraction_difficulty,
                                specific_distractor='dog'
                            )
                }
            

    # Initialize oracle agent.
    if 'point' in FLAGS.env_name:

        def actor_fn(ob, temperature):
            return ob[-2:]
    else:
        # Load agent config.
        restore_path = FLAGS.restore_path
        candidates = glob.glob(restore_path)
        assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

        with open(candidates[0] + '/flags.json', 'r') as f:
            agent_config = json.load(f)['agent']

        # Load agent.
        agent = SACAgent.create(
            FLAGS.seed,
            np.zeros(ob_dim),
            env.action_space.sample(),
            agent_config,
        )
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        actor_fn = supply_rng(agent.sample_actions, rng=agent.rng)

    # Store all empty cells and vertex cells.
    all_cells = []
    vertex_cells = []
    maze_map = env.unwrapped.maze_map
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 0:
                all_cells.append((i, j))

                # Exclude hallway cells.
                if (
                    maze_map[i - 1, j] == 0
                    and maze_map[i + 1, j] == 0
                    and maze_map[i, j - 1] == 1
                    and maze_map[i, j + 1] == 1
                ):
                    continue
                if (
                    maze_map[i, j - 1] == 0
                    and maze_map[i, j + 1] == 0
                    and maze_map[i - 1, j] == 1
                    and maze_map[i + 1, j] == 1
                ):
                    continue

                vertex_cells.append((i, j))

    # Collect data.
    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        if FLAGS.dataset_type in ['path', 'navigate', 'explore']:
            # Sample an initial state from all cells.
            init_ij = all_cells[np.random.randint(len(all_cells))]
            # Sample a goal state from vertex cells.
            goal_ij = vertex_cells[np.random.randint(len(vertex_cells))]
        elif FLAGS.dataset_type == 'stitch':
            # Sample an initial state from all cells.
            init_ij = all_cells[np.random.randint(len(all_cells))]

            # Perform BFS to find adjacent cells.
            adj_cells = []
            adj_steps = 4  # Target distance from the initial cell.
            bfs_map = maze_map.copy()
            for i in range(bfs_map.shape[0]):
                for j in range(bfs_map.shape[1]):
                    bfs_map[i][j] = -1
            bfs_map[init_ij[0], init_ij[1]] = 0
            queue = [init_ij]
            while len(queue) > 0:
                i, j = queue.pop(0)
                for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < bfs_map.shape[0]
                        and 0 <= nj < bfs_map.shape[1]
                        and maze_map[ni, nj] == 0
                        and bfs_map[ni, nj] == -1
                    ):
                        bfs_map[ni][nj] = bfs_map[i][j] + 1
                        queue.append((ni, nj))
                        if bfs_map[ni][nj] == adj_steps:
                            adj_cells.append((ni, nj))

            # Sample a goal state from adjacent cells.
            goal_ij = adj_cells[np.random.randint(len(adj_cells))] if len(adj_cells) > 0 else init_ij
        else:
            raise ValueError(f'Unsupported dataset_type: {FLAGS.dataset_type}')


        # Use environment with specific distractor video based on starting location of agent (init_ij)
        # middle_width = maze_map.shape[1] // 2

        maze_height = maze_map.shape[0]
        maze_width = maze_map.shape[1]

        # print("Maze map shape: ", maze_map.shape)

        # Add distractor based on if the starting location is above or below the main diagonal
        init_row = init_ij[0]
        init_col = init_ij[1]

        norm_row = init_row / maze_height
        norm_col = init_col / maze_width

        # print("###")
        unique_distractor = 'left' if norm_col >= norm_row else 'right'
        # print("Episode: ", ep_idx)
        # print("init_row: ", init_row)
        # print("init_col: ", init_col)
        
        # print("norm_col: ", norm_col)
        # print("norm_row: ", norm_row)
        # print("unique_distractor: ", unique_distractor)

        # print("maze_height: ", maze_height)
        # print("maze_width: ", maze_width)
        # print("###")


        env = distracted_envs[unique_distractor]

        ob, _ = env.reset(options=dict(task_info=dict(init_ij=init_ij, goal_ij=goal_ij)))
        
        done = False
        step = 0

        cur_subgoal_dir = None  # Current subgoal direction (only for 'explore').

        while not done:
            if FLAGS.dataset_type == 'explore':
                # Sample a random direction every 10 steps.
                if step % 10 == 0:
                    cur_subgoal_dir = np.random.randn(2)
                    cur_subgoal_dir = cur_subgoal_dir / (np.linalg.norm(cur_subgoal_dir) + 1e-6)
                subgoal_dir = cur_subgoal_dir
            else:
                # Get the oracle subgoal and compute the direction.
                subgoal_xy, _ = env.unwrapped.get_oracle_subgoal(env.unwrapped.get_xy(), env.unwrapped.cur_goal_xy)
                subgoal_dir = subgoal_xy - env.unwrapped.get_xy()
                subgoal_dir = subgoal_dir / (np.linalg.norm(subgoal_dir) + 1e-6)

            agent_ob = env.unwrapped.get_ob(ob_type='states')
            # Exclude the agent's position and add the subgoal direction.
            agent_ob = np.concatenate([agent_ob[2:], subgoal_dir])
            action = actor_fn(agent_ob, temperature=0)
            # Add Gaussian noise to the action.
            action = action + np.random.normal(0, FLAGS.noise, action.shape)
            action = np.clip(action, -1, 1)

            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = info['success']

            # Sample a new goal state when the current goal is reached.
            if success and FLAGS.dataset_type == 'navigate':
                goal_ij = vertex_cells[np.random.randint(len(vertex_cells))]
                env.unwrapped.set_goal(goal_ij)

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])

            ob = next_ob
            step += 1

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step
        

        # See if we should periodically save the dataset (also save the dataset in an outer folder with num episodes)
        if (ep_idx % FLAGS.save_period == 0):
            save_data('intermediate_save_' + str(ep_idx) + '/', dataset, total_train_steps)


    # Save one final time with all data
    save_data('final_save_' + str(ep_idx) + '/', dataset, total_train_steps)
    print('Total steps:', total_steps)


def save_data(save_subfolder_name: str, dataset, total_train_steps):
    train_path = FLAGS.save_dir + save_subfolder_name + FLAGS.save_file_name
    val_path = FLAGS.save_dir + save_subfolder_name + FLAGS.save_file_name.replace('.npz', '-val.npz')
    

    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminals':
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)





if __name__ == '__main__':
    app.run(main)
