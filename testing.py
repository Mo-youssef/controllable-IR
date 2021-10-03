#%%
import gym
from utils import ChannelOrder, ResizeFrame
import pfrl 
import matplotlib.pyplot as plt
import griddly
#%%
env = gym.make('GDY-Clusters-v0')
env.reset()
env = ChannelOrder(env, channel_order='hwc')
env = ResizeFrame(env, (84,84))
env = ChannelOrder(env, channel_order='chw')
env = pfrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=10800)
env.enable_history(True)
#%%
observation = env.reset()
#%%
plt.imshow(observation.transpose(1,2,0))
plt.show()
#%%
observation, reward, done, info = env.step(0)
# %% Register environment
import pfrl
import os
import matplotlib as plt

import gym
import griddly

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
# %%
env = gym.make('GDY-Clusters-Semi-Sparse-Wall-v0')
# %%
# observation, reward, done, info = env.step(0)
# print(reward)
# plt.imshow(observation.transpose(1,2,0))
# plt.show()

# del gym.envs.registry.env_specs['GDY-Clusters-Semi-Sparse-Wall-v0']
# griddly.GymWrapperFactory().build_gym_from_yaml(
#     'Clusters-Semi-Sparse-Wall',
#     os.path.join(os.getcwd(), 'semi_sparse_clusters_wall.yml'),
#     level=0,
#     player_observer_type=griddly.gd.ObserverType.SPRITE_2D,
# )

# env = gym.make('GDY-Clusters-Semi-Sparse-Wall-v0')

# env.reset()