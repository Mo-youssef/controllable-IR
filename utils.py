import numpy as np
import pfrl
import os
from pfrl import explorer

import gym
import griddly
from gym import spaces
from gym_minigrid.wrappers import ReseedWrapper, RGBImgObsWrapper, ImgObsWrapper
from pfrl.wrappers import atari_wrappers
import torch.nn as nn

try:
    import cv2

    cv2.ocl.setUseOpenCL(False)
    _is_cv2_available = True
except Exception:
    _is_cv2_available = False

if 'GDY-Clusters-Sparse-v0' not in [env.id for env in gym.envs.registry.all()]:
    griddly.GymWrapperFactory().build_gym_from_yaml(
        'Clusters-Sparse',
        os.path.join(os.getcwd(), 'sparse_clusters.yml'),
        level=0,
        player_observer_type=griddly.gd.ObserverType.SPRITE_2D,
    )
if 'GDY-Clusters-Semi-Sparse-v0' not in [env.id for env in gym.envs.registry.all()]:
    griddly.GymWrapperFactory().build_gym_from_yaml(
        'Clusters-Semi-Sparse',
        os.path.join(os.getcwd(), 'semi_sparse_clusters.yml'),
        level=0,
        player_observer_type=griddly.gd.ObserverType.SPRITE_2D,
    )
if 'GDY-Clusters-Semi-Sparse-Wall-v0' not in [env.id for env in gym.envs.registry.all()]:
    griddly.GymWrapperFactory().build_gym_from_yaml(
        'Clusters-Semi-Sparse-Wall',
        os.path.join(os.getcwd(), 'semi_sparse_clusters_wall.yml'),
        level=0,
        player_observer_type=griddly.gd.ObserverType.SPRITE_2D,
    )
if 'GDY-Clusters-Semi-Sparse-Wall-No-v0' not in [env.id for env in gym.envs.registry.all()]:
    griddly.GymWrapperFactory().build_gym_from_yaml(
        'Clusters-Semi-Sparse-Wall-No',
        os.path.join(os.getcwd(), 'semi_sparse_clusters_wall_no.yml'),
        level=0,
        player_observer_type=griddly.gd.ObserverType.SPRITE_2D,
    )

class RandomSelectionEpsilonGreedy(explorer.Explorer):

    def __init__(
        self,
        start_epsilon,
        end_epsilon,
        num_epsilon,
        epsilon_interval,
        random_action_func
    ):
        assert start_epsilon >= 0 and start_epsilon <= 1
        assert end_epsilon >= 0 and end_epsilon <= 1
        assert num_epsilon >= 0
        self.random_action_func = random_action_func
        self.epsilon = start_epsilon
        self.epsilon_interval = epsilon_interval
        self.epsilon_range = np.linspace(
            start_epsilon, end_epsilon, num_epsilon)

    def select_action_epsilon_greedily(self, epsilon, random_action_func, greedy_action_func):
        if np.random.rand() < epsilon:
            return random_action_func(), False
        else:
            return greedy_action_func(), True

    def compute_epsilon(self, t):
        if t % self.epsilon_interval == 0:
          return np.random.choice(self.epsilon_range)
        return self.epsilon

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        a, _ = self.select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func
        )
        return a

    def __repr__(self):
        return "RandomSelectionEpsilonGreedy(epsilon={})".format(self.epsilon)

def mini_grid_wrapper(env_id, max_frames=0, clip_rewards=True, frame_stack=True):
    env = gym.make(env_id)
    env = ReseedWrapper(env, seeds=[0])
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    if max_frames:
        env = pfrl.wrappers.ContinuingTimeLimit(
            env, max_episode_steps=max_frames)
    # env = atari_wrappers.MaxAndSkipEnv(env, skip=0)
    env = atari_wrappers.wrap_deepmind(
        env, episode_life=False, clip_rewards=clip_rewards, frame_stack=frame_stack)
    return env

class ResizeFrame(gym.ObservationWrapper):

    def __init__(self, env, target_shape):
        if not _is_cv2_available:
            raise RuntimeError(
                "Cannot import cv2 module. Please install OpenCV-Python to use"
                " WarpFrame."
            )
        gym.ObservationWrapper.__init__(self, env)

        self.target_shape = target_shape
        self.channels = min(env.observation_space.shape)
        self.channel_pos = np.argmin(env.observation_space.shape)
        if self.channel_pos == 0:
            shape = (self.channels, target_shape[0], target_shape[1])
        else:
            shape = (target_shape[0], target_shape[1], self.channels)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape, dtype=env.observation_space.dtype
        )

    def observation(self, frame):
        resized = cv2.resize(
            frame, self.target_shape, interpolation=cv2.INTER_AREA
        )
        return resized

class ChannelOrder(gym.ObservationWrapper):

    def __init__(self, env, channel_order):
        gym.ObservationWrapper.__init__(self, env)

        self.channels = min(env.observation_space.shape)
        self.channel_pos = np.argmin(env.observation_space.shape)
        if self.channel_pos == 0:
            self.width = env.observation_space.shape[1]
            self.height = env.observation_space.shape[2]
        else:
            self.width = env.observation_space.shape[0]
            self.height = env.observation_space.shape[1]

        shape = {
            "hwc": (self.height, self.width, self.channels),
            "chw": (self.channels, self.height, self.width),
        }
        self.no_need = shape[channel_order] == env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape[channel_order], dtype=env.observation_space.dtype
        )

    def observation(self, frame):
        if self.no_need:
            return frame

        return frame.transpose(1, 2, 0) if self.channel_pos == 0 else frame.transpose(2, 0, 1)

class RewardScaling(gym.Wrapper):
    def __init__(self, env, scale, event):
        """Take action on reset for envs that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        self.scale = scale
        self.event = event
        
    def _get_griddly_events(self, info):
        events = list()
        for item in info['History']:
            events.append('{}-{}-{}'.format(item['SourceObjectName'], item['ActionName'], item['DestinationObjectName']))
        return events or ['nothing']

    def step(self, ac):
        obs, reward, done, info = self.env.step(ac)
        events = self._get_griddly_events(info)
        if np.any([1 if self.event in event else 0 for event in events]):
            assert reward < 0, "No punishement for stuck boxes"
            reward *= self.scale
        return obs, reward, done, info
    
class ReturnState(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, ac):
        obs, reward, done, info = self.env.step(ac)
        return obs, reward, done, (info, self.env.get_state())

def griddly_wrapper(env_id, max_frames=0, clip_rewards=True, frame_stack=True, obs_shape=(84,84), test=False, punishment=1):
    env = gym.make(env_id)
    env.enable_history(True)
    env.reset()
    env = ChannelOrder(env, channel_order='hwc')
    env = ResizeFrame(env, (84,84))
    env = ChannelOrder(env, channel_order='chw')
    env = RewardScaling(env, scale=punishment, event="box-move-near_wall")
    if max_frames:
        env = pfrl.wrappers.ContinuingTimeLimit(
            env, max_episode_steps=max_frames)
    # env = atari_wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=clip_rewards, frame_stack=frame_stack)
    if test:
        env = ReturnState(env)
    return env

def wrap_env(env_id, max_frames=5000, clip_rewards=True, episode_life=True, frame_stack=True, obs_shape=(84,84), test=False, punishment=1):
    if env_id.startswith('MiniGrid'):
        env = mini_grid_wrapper(
            env_id, max_frames=max_frames, clip_rewards=clip_rewards, frame_stack=frame_stack)
    elif env_id.startswith('GDY'):
        env = griddly_wrapper(
            env_id, max_frames=max_frames, clip_rewards=clip_rewards, frame_stack=frame_stack, obs_shape=obs_shape, test=test, punishment=punishment)
    else:
        env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari(
            env_id, max_frames=max_frames), episode_life=True, clip_rewards=clip_rewards, frame_stack=frame_stack)
    return env

class MovingVariance():
    def __init__(self, eps=1e-10):
        self.num = 0
        self.eps = eps
    def push_val(self, x):
        self.num += 1
        if self.num == 1:
            self.old_m = x
            self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m)/self.num
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
            self.old_m = self.new_m
            self.old_s = self.new_s
    def push(self, vec):
        try:
            if len(vec) > 0:
                for x in vec:
                    self.push_val(x)
        except TypeError:
            self.push_val(x)

    def num_samples(self):
        return self.num
    def mean(self):
        return self.new_m if self.num else 0
    def variance(self):
        return self.new_s / (self.num - 1) if self.num > 1 else 0
    def std(self):
        return (np.sqrt(self.variance()) + self.eps) if self.num > 1 else 1

def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer