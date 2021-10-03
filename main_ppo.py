import gym
import numpy as np
import datetime
import random
import sys
import pickle
import os
from pfrl.agents import ppo
import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wandb
import logging
from collections import deque

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import pfrl
from pfrl import agents
from pfrl import nn as pnn
from pfrl import utils
from pfrl import experiments
from pfrl.experiments.evaluator import save_agent
from evaluator import Evaluator

import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from models.ppo_model import PPO_model
from models.DQN_model import Embedding_fn, Embedding_full
from utils import wrap_env
from ppo_ir_modules import NGU_module, CTRL_module

import pdb
# import ppo_params

from ppo_parser import create_parser
ppo_params = create_parser().parse_args()
ppo_params.IR_module = 'None'
if ppo_params.ngu_reward:
    ppo_params.IR_module = 'NGU'
elif ppo_params.ctrl_reward:
    ppo_params.IR_module = 'CTRL'

print(ppo_params)
# pdb.set_trace()
if ppo_params.env_name.startswith('GDY'):
    import griddly


# experiment = (f"{ppo_params.env_alias}_maxFrames-{ppo_params.max_frames}_LR-{ppo_params.lr}"
#               f"_BS-{ppo_params.batch_size}_discount-{ppo_params.discount}_numEnvs-{ppo_params.num_envs}"
#               f"_epochs-{ppo_params.epochs}_seed-{ppo_params.seed}" #)
#               f"{'_NGU_e' if ppo_params.ngu_reward else ''}{ppo_params.ngu_embed_size if ppo_params.ngu_reward else ''}"
#               f"{'_CTRL_hs' if ppo_params.ctrl_reward else ''}{ppo_params.ctrl_hidden_size if ppo_params.ctrl_reward else ''}"
#               f"{'_CTRL_ls' if ppo_params.ctrl_reward else ''}{ppo_params.ctrl_latent_size if ppo_params.ctrl_reward else ''}_s{ppo_params.seed}")
experiment = (f"{ppo_params.env_alias}_seed-{ppo_params.seed}"
              f"{'_NGU' if ppo_params.ngu_reward else ''}"
              f"{'_CTRL' if ppo_params.ctrl_reward else ''}")

if ppo_params.wandb:
    wandb.init(
            project=ppo_params.wandb_project,
            entity='youssef101',
            name=experiment,
            config=ppo_params,
            )
    ppo_params = wandb.config
    unique_id = wandb.run.id

torch.manual_seed(ppo_params.seed)
torch.cuda.manual_seed(ppo_params.seed)
np.random.seed(ppo_params.seed)
random.seed(ppo_params.seed)

# Creating batch envs to run in parallel
process_seeds = np.arange(ppo_params.num_envs) + ppo_params.seed * ppo_params.num_envs
assert process_seeds.max() < 2 ** 32
# pdb.set_trace()
def make_env(idx, test, frame_stack=True, punishment=1):
    process_seed = int(process_seeds[idx])
    env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
    env = wrap_env(ppo_params.env_name, max_frames=ppo_params.eval_max_frames if test else ppo_params.max_frames, 
                    clip_rewards=ppo_params.clip_rewards, frame_stack=frame_stack, obs_shape=(84,84), test=test, punishment=punishment)
    env.seed(env_seed)
    return env
def make_batch_env(test):
    vec_env = pfrl.envs.MultiprocessVectorEnv(
        [
            (lambda: make_env(idx, test, frame_stack=False, punishment=ppo_params.punishment_scale))
            for idx, env in enumerate(range(ppo_params.num_envs))
        ]
    )
    if not ppo_params.no_frame_stack:
        vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
    return vec_env

sample_env = make_batch_env(test=False)
print("Observation space", sample_env.observation_space)
print("Action space", sample_env.action_space)
n_actions = sample_env.action_space.n
obs_n_channels = sample_env.observation_space.low.shape[0]
print('Observation num channels: ', obs_n_channels)
del sample_env

print("Loading model ................")
model = PPO_model(obs_n_channels, n_actions, recurrent=ppo_params.recurrent)
print("done")
  
optimizer = torch.optim.Adam(model.parameters(), lr=ppo_params.lr, eps=1e-8) # eps=1.5*10**-4)

agent = agents.PPO(
            model,
            optimizer,
            gpu=0,
            phi=lambda x: np.asarray(x, dtype=np.float32) / 255,
            update_interval=ppo_params.update_interval,
            minibatch_size=ppo_params.batch_size,
            epochs=ppo_params.epochs,
            clip_eps=0.2,
            clip_eps_vf=None,
            standardize_advantages=True,
            entropy_coef=0.01,
            max_grad_norm=ppo_params.grad_norm,
            gamma=ppo_params.discount,
            value_func_coef=0.5, # value_weight in oriol's code
            recurrent=ppo_params.recurrent,
        )
print(f'Num of GPUs available = {torch.cuda.device_count()}')

env = make_batch_env(test=False)
num_envs = env.num_envs
if ppo_params.evaluate:
    eval_env = make_batch_env(test=True) if ppo_params.eval_num_envs > 1 else make_env(idx=0, test=True, frame_stack=False, punishment=ppo_params.punishment_scale) 
else:
    eval_env = None

os.makedirs(ppo_params.outdir, exist_ok=True)
logger = logging.getLogger(__name__)

intrinsic_reward = ppo_params.ngu_reward or ppo_params.ctrl_reward
episodic_ir_module = None
if ppo_params.ngu_reward:
    encoder = lambda: Embedding_fn(embedding_size=ppo_params.ngu_embed_size, input_channels=obs_n_channels)
    episodic_ir_module = NGU_module(encoder, Embedding_full, torch.optim.Adam, ppo_params.ngu_lr, agent,
                                    n_actions, ppo_params.max_frames, ppo_params.ngu_embed_size,
                                    ppo_params.ngu_k_neighbors, ppo_params.ngu_update_schedule, num_envs,
                                    ppo_params.ngu_mem, ppo_params.batch_size, ppo_params.ir_model_copy)
    episodic_ir_module.reset(np.ones(num_envs))
elif ppo_params.ctrl_reward:
    model_args = {
        'input_channels': obs_n_channels,
        'num_actions': env.action_space.n,
        'hidden_size': ppo_params.ctrl_hidden_size,
        'channels': ppo_params.ctrl_channels,
        'latent_size': ppo_params.ctrl_latent_size,
        'encoder_out': ppo_params.ctrl_encoder_out,
    }
    episodic_ir_module = CTRL_module(model_args, torch.optim.Adam, ppo_params.ngu_lr, agent, 
                            ppo_params.ctrl_weight_normal, ppo_params.max_frames, ppo_params.ngu_k_neighbors, 
                            ppo_params.ngu_update_schedule, num_envs, ppo_params.ngu_mem,
                            ppo_params.batch_size, ppo_params.ir_model_copy)

    episodic_ir_module.reset(np.ones(num_envs))

evaluator = Evaluator(
    env=eval_env,
    agent=agent,
    num_episodes=ppo_params.eval_n_runs,
    eval_frequency=ppo_params.eval_interval,
    video_frequency=ppo_params.video_every,
    intrinsic_model=episodic_ir_module,
    unique_id=unique_id,
) if ppo_params.evaluate else None

return_window_size = 100
recent_returns = deque(maxlen=return_window_size)
recent_ireturns = deque(maxlen=return_window_size)

episode_r = np.zeros(num_envs, dtype=np.float64)
episode_ir = np.zeros(num_envs, dtype=np.float64)
episode_idx = np.zeros(num_envs, dtype="i")
episode_len = np.zeros(num_envs, dtype="i")

# o_0, r_0
obss = env.reset()
new_obs = [np.array(o) for o in obss]  # [np.array(o)[-1:] for o in obss] in case of frame_stack


step_offset = ppo_params.warmup
if hasattr(agent, "t"):
    print('Agent has attribute t')
    agent.t = step_offset

# t = step_offset
t = 0
steps = 0
episode_len_queue = deque(maxlen=100)
# eval_stats_history = []  # List of evaluation episode stats dict
print(model)
print('TRAINING begins ....................')
try:
    while True:
        # a_t
        actions = agent.batch_act(obss)
        # o_{t+1}, r_{t+1}
        
        old_obs = new_obs
        obss, rs, dones, infos = env.step(actions)  # obss is list of lazyframes , rs & dones are tuples, actions is np.array, infos is a tuple of dicts
        reward = np.array(rs, dtype=float)
        

        # pdb.set_trace()
        episode_r += rs
        episode_len += 1

        # Compute mask for done and reset
        if ppo_params.max_frames is None:
            resets = np.zeros(num_envs, dtype=bool)
        else:
            resets = episode_len == ppo_params.max_frames
        resets = np.logical_or(
            resets, [info.get("needs_reset", False) for info in infos]
        )
        # Make mask. 0 if done/reset, 1 if pass
        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)
        steps += 1
        if intrinsic_reward:
            new_obs = [np.array(o) for o in obss]  # [np.array(o)[-1:] for o in obss] in case of frame_stack
            mem_items = list(zip(old_obs, new_obs, actions, infos))
            episodic_ir_module.memory.add(mem_items)

            episodic_ir_module.reset(end)
            recent_ireturns.extend(episode_ir[end])
            episode_ir[end] = 0

            if (t > ppo_params.ir_warmup):
                episodic_memory_stats = episodic_ir_module.train(steps)         

            if ppo_params.IR_module == 'NGU':
                reward_int = episodic_ir_module.compute_reward(obss) 
            elif ppo_params.IR_module == 'CTRL':
                reward_int = episodic_ir_module.compute_reward(old_obs, actions)
            else:
                print(f'IR module:{ppo_params.IR_module} is not implemented yet')
            # pdb.set_trace()
            reward += ppo_params.ir_beta*reward_int if (t > ppo_params.warmup) else 0
            episode_ir += reward_int


        # Agent observes the consequences
        agent.batch_observe(obss, reward, dones, resets)



        # For episodes that ends, do the following:
        #   1. increment the episode count
        #   2. record the return
        #   3. clear the record of rewards
        #   4. clear the record of the number of steps
        #   5. reset the env to start a new episode
        # 3-5 are skipped when training is already finished.
        episode_idx += end
        recent_returns.extend(episode_r[end])

        for _ in range(num_envs):
            t += 1
            if ppo_params.checkpoint_frequency and t % ppo_params.checkpoint_frequency == 0:
                save_agent(agent, t, ppo_params.outdir+'/'+unique_id, logger, suffix='_agent')
                torch.save(episodic_ir_module.test_embedding_fn, ppo_params.outdir+'/'+unique_id+f'/{t}_ir_model.pt')
                # with open(unique_id+'_test_ir_save.pkl', 'wb') as f:
                #     pickle.dump(episodic_ir_module.test_embedding_fn, f)
                print('AGENT and IR saved .....')

        if (
            ppo_params.log_interval is not None
            and t >= ppo_params.log_interval
            and t % ppo_params.log_interval < num_envs
        ):
            log_str = (f"[{datetime.datetime.now():%Y-%m-%d %T}] ppo:{ppo_params.outdir} step:{t} episode:{np.sum(episode_idx)}"
                       f" last_R: {recent_returns[-1] if recent_returns else np.nan} average_R:{np.mean(recent_returns) if recent_returns else np.nan:.4f}"
                       f" episode_len: {np.mean(episode_len_queue):.4f}")
            logger.info(log_str)
            print(log_str, flush=True)
            logger.info("statistics: {}".format(agent.get_statistics()))
            if ppo_params.wandb:
                wandb.log({'env_steps': t, 'agent_train_steps': agent.n_updates, 'last_R': recent_returns[-1] if recent_returns else np.nan,
                        'episode':np.sum(episode_idx), 'average_R':np.mean(recent_returns) if recent_returns else np.nan, 
                        'episode_len': np.mean(episode_len_queue), 'max_R': np.max(recent_returns) if recent_returns else np.nan,})

                if intrinsic_reward:
                    wandb.log({'env_steps': t, 'last_IR': recent_ireturns[-1] if recent_ireturns else np.nan,
                            'average_IR':np.mean(recent_ireturns) if recent_ireturns else np.nan})
                    if (episodic_ir_module.accum_counter > 0):
                        if ppo_params.ngu_reward:
                            wandb.log({'env_steps': t, 'NGU_steps': episodic_ir_module.train_steps, 'NGU_loss': episodic_ir_module.accum_loss/episodic_ir_module.accum_counter,
                                'NGU_accuracy': episodic_ir_module.accum_acc/episodic_ir_module.accum_counter,
                                # 'confusion_matrix': episodic_ir_module.confusion_matrix()})
                            })
                        elif ppo_params.ctrl_reward:
                            wandb.log({'env_steps': t, 'CTRL_steps': episodic_ir_module.train_steps, 'CTRL_loss': episodic_ir_module.accum_loss/episodic_ir_module.accum_counter,
                                'CTRL_recon_loss': episodic_ir_module.accum_recon/episodic_ir_module.accum_counter,
                                'CTRL_normal_loss': episodic_ir_module.accum_norm/episodic_ir_module.accum_counter,
                                # 'ctrl_example_masks': episodic_ir_module.sample_imgs(size=5)})
                            })
                        episodic_ir_module.reset_stats()
                        
        if evaluator and t >= ppo_params.warmup:
            eval_stats = evaluator.evaluate_if_necessary(step=t)
            if eval_stats is not None:
                agent_stats = dict(agent.get_statistics())
                # eval_stats_history.append(eval_stats)
                if ppo_params.wandb:
                    wandb.log(dict(env_steps=t, agent_train_steps=agent.n_updates,
                                episode=np.sum(episode_idx), **eval_stats, **agent_stats,
                            ))
                # if (
                #     ppo_params.successful_score is not None
                #     and evaluator.max_score >= ppo_params.successful_score
                # ):
                #     env.close()
                #     evaluator.env.close()
                #     break

        if t >= ppo_params.max_steps:
            # env.close()
            break

        # Start new episodes if needed
        if np.sum(end) > 0:
            episode_len_queue.append(np.mean(episode_len[end]))
        episode_r[end] = 0
        episode_len[end] = 0
        obss = env.reset(not_end)

except (Exception, KeyboardInterrupt):
    # Save the current model before being killed
    # save_agent(agent, t, ppo_params.outdir, logger, suffix="_except")
    print('agent saved')
    env.close()
    print('env closed')
    if evaluator:
        evaluator.env.close()
    raise
else:
    # Save the final model
    # save_agent(agent, t, ppo_params.outdir, logger, suffix="_finish")
    env.close()
    if evaluator:
        evaluator.env.close()


# print(eval_stats_history)