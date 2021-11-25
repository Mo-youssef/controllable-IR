import numpy as np
from collections import deque
import copy
from skimage import color
import matplotlib.pyplot as plt
import PIL

import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl.replay_buffer import batch_experiences

from models.DQN_model import Net, RNDNet
from utils import MovingVariance
from models.disentangle_network import DisentangleNetwork
from models.controlled_network import ControlledNetwork

from pfrl.utils.random import sample_n_k
import wandb

import pdb

class MainMemory():
    def __init__(self, max_size, batch_size, device):
        self.data = deque(maxlen=max_size)
        self.max_size = max_size
        self.device = device
        self.batch_size = batch_size
    def uniform_sample(self):
        indices = sample_n_k(len(self.data), self.batch_size)
        sampled = [self.data[i] for i in indices]
        return sampled # list of tuples of form (old obs, new obs, action, info)
    def add(self, data):
        for item in data:
            self.data.append(item) 
            
class EpisodicMemory():
    def __init__(self, max_size, embedding_size, k_neighbors, eps=0.01, C=0.001, psi=0.008):
        self.num = 0
        self.max_size = max_size
        self.memory = np.zeros((max_size, embedding_size))
        self.k_neighbors = k_neighbors
        self.running_sum = 0
        self.running_num = 0
        self.eps = eps
        self.C = C
        self.psi = psi
        self.max_sim = 8
        self.embedding_size = embedding_size

    def reset(self):
        self.num = 0
        # self.running_sum = 0
        # self.running_num = 0

    def add_item(self, embedding):
        self.memory[self.num % self.max_size] = np.array(embedding).ravel()
        self.num = (self.num + 1) % self.max_size

    def score(self, embedding):
        if self.num < self.k_neighbors:
            return 1 
        test = np.array(embedding).ravel()[None, :]
        dists = np.sum((test - self.memory[:self.num])**2, axis=1)
        k_dist = np.partition(dists, self.k_neighbors+1)[:self.k_neighbors+1] if len(
            dists) > self.k_neighbors+1 else np.sort(dists)[:self.k_neighbors+1]
        k_dist = np.delete(k_dist, k_dist.argmin()) if len(
            k_dist) > 1 else k_dist
        self.running_sum += np.sum(k_dist)
        self.running_num += len(k_dist)
        running_mean = self.running_sum / self.running_num
        dist_normalized = k_dist / max(running_mean, 1e-8)   # (running_mean if abs(running_mean - 0)>self.psi else self.psi )
        dist_normalized = np.maximum(dist_normalized - self.psi, 0)
        dist_kernel = self.eps / (self.eps + dist_normalized)
        sim = np.sqrt(np.sum(dist_kernel)) + self.C
        return 0 if sim >= self.max_sim else 1/sim

class Eval_intrinsic_module():
    def __init__(self, encoder, agent, episodic_max_size,
                 embedding_size, k_neighbors, num_envs):
        self.agent = agent
        self.test_embedding_fn = encoder
        self.num_envs = num_envs
        self.episodic_memory = [EpisodicMemory(episodic_max_size, embedding_size, k_neighbors) for _ in range(num_envs)]
        self.name = 'TOBEASSIGNED'

    def reset(self, end_vec):
        for i, e in enumerate(end_vec):
            if e:
                self.episodic_memory[i].reset()

    def __call__(self, obss, actions=None, BS=1024):
        assert self.name != 'TOBEASSIGNED', 'Please assign a name to the module'
        all_embeddings = []
        for i in range(int(np.ceil(len(obss)/BS))):
            obs = obss[i*BS:(i+1)*BS]
            obs = self.agent.phi(obs)
            obs = torch.as_tensor(obs, device=self.agent.device, dtype=torch.float32)
            self.test_embedding_fn.eval()
            with torch.no_grad():
                if actions is None:
                    embeddings = self.test_embedding_fn(obs)
                else:
                    action = actions[i*BS:(i+1)*BS]
                    action = np.array(action)
                    action = torch.as_tensor(action, device=self.agent.device, dtype=torch.int)
                    if self.name == 'CTRL':
                        _, _, embeddings, _ = self.test_embedding_fn(obs, action)
                    elif self.name == 'ALL_CTRL':
                        _, embeddings = self.test_embedding_fn(obs, action)
                    else:
                        raise ValueError('Module name is not assigned')
            embeddings = embeddings.cpu().detach().numpy()
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)
        
    def compute_reward(self, obs_vec, action_vec=None):
        reward_vec = []
        embeddings = []
        if action_vec is not None:
            obs = np.array(obs_vec)
            obs = self.agent.phi(obs) # [:, -1:, :, :] 
            obs = torch.as_tensor(obs, device=self.agent.device,
                                dtype=torch.float32)
            action = np.array(action_vec)
            action = torch.as_tensor(action, device=self.agent.device,
                                dtype=torch.int)
            self.test_embedding_fn.eval()
            with torch.no_grad():
                _, _, embeddings, _ = self.test_embedding_fn(obs, action)
            embeddings = embeddings.cpu().detach().numpy()
            reward_vec = []
            for i, embedding in enumerate(embeddings):
                self.episodic_memory[i].add_item(embedding)
                reward = self.episodic_memory[i].score(embedding)
                reward_vec.append(reward)
        else:
            obs = np.array(obs_vec)
            obs = self.agent.phi(obs) # [:, -1:, :, :] incldue this in case of stack frame, make sure to modify how elements are added to the memory in main_ppo
            obs = torch.as_tensor(obs, device=self.agent.device, dtype=torch.float32)
            self.test_embedding_fn.eval()
            with torch.no_grad():
                embeddings = self.test_embedding_fn(obs)
            embeddings = embeddings.cpu().detach().numpy()
            reward_vec = []
            for i, embedding in enumerate(embeddings):
                self.episodic_memory[i].add_item(embedding)
                reward = self.episodic_memory[i].score(embedding)
                reward_vec.append(reward)

        return np.array(reward_vec), embeddings

class NGU_module():
    name = 'NGU'
    def __init__(self, embedding_fn, embedding_model, optimizer,
                 lr, agent, n_actions, episodic_max_size, embedding_size,
                 k_neighbors, update_schedule, num_envs, mem_size, batch_size, ir_model_copy):
        # this class takes the agent as argument to get parameters not for any calculations
        self.agent = agent
        self.embedding_fn = embedding_fn().to(self.agent.device)
        self.test_embedding_fn = copy.deepcopy(self.embedding_fn).to(self.agent.device)
        assert id(self.test_embedding_fn) != id(self.embedding_fn), 'test and train IR models are the same'
        self.embedding_model = embedding_model(
            self.embedding_fn, n_actions).to(self.agent.device)
        self.optimizer = optimizer(
            self.embedding_model.parameters(), lr=lr)
        self.episodic_memory = [EpisodicMemory(episodic_max_size, embedding_size, k_neighbors) for _ in range(num_envs)]
        self.memory = MainMemory(mem_size, batch_size, agent.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_envs = num_envs
        self.train_steps = 0
        self.update_schedule = update_schedule
        self.update_frequency = list(update_schedule.values())[0]
        self.model_copy = ir_model_copy
        self.accum_acc = 0
        self.accum_loss = 0
        self.accum_counter = 0
        self.confusion_matrix_queue = deque(maxlen=100)

    def reset(self, end_vec):
        for i, e in enumerate(end_vec):
            if e:
                self.episodic_memory[i].reset()

    def compute_reward(self, obs_vec):
        # get embeddings
        obs = np.array(obs_vec)
        obs = self.agent.phi(obs) # [:, -1:, :, :] incldue this in case of stack frame, make sure to modify how elements are added to the memory in main_ppo
        obs = torch.as_tensor(obs, device=self.agent.device, dtype=torch.float32)
        self.test_embedding_fn.eval()
        with torch.no_grad():
            embeddings = self.test_embedding_fn(obs)
        embeddings = embeddings.cpu().detach().numpy()
        reward_vec = []
        for i, embedding in enumerate(embeddings):
            # add memory to episodic memory and get score
            self.episodic_memory[i].add_item(embedding)
            reward = self.episodic_memory[i].score(embedding)

            reward_vec.append(reward)

        return np.array(reward_vec)

    def reset_stats(self):
        self.accum_acc = 0
        self.accum_loss = 0
        self.accum_counter = 0

    def train(self, time_step):
        if time_step % self.model_copy == 0:
            # print(f'model copy at t = {time_step}')
            self.test_embedding_fn.load_state_dict(self.embedding_fn.state_dict())

        self.update_frequency = self.update_schedule[time_step] if time_step in self.update_schedule.keys() else self.update_frequency
        if time_step % self.update_frequency != 0:  # take into account multiple envs
            return dict()
            
        self.train_steps += 1
        self.embedding_model.train()

        experiences = self.memory.uniform_sample()  # list of tuples of form (old obs, new obs, action, info)
        exp_batch = {
            "state": torch.as_tensor(self.agent.phi([elem[0] for elem in experiences]), device=self.agent.device),
            "next_state": torch.as_tensor(self.agent.phi([elem[1] for elem in experiences]), device=self.agent.device),
            "action": torch.as_tensor([elem[2] for elem in experiences], device=self.agent.device),
        }
        # pdb.set_trace()
        observations = exp_batch["state"]
        next_observations = exp_batch["next_state"]
        actions = exp_batch["action"]
        output = self.embedding_model(observations, next_observations)
        loss = self.loss_fn(output, actions)
        self.mean_loss = loss.item()
        preds = torch.argmax(output.data, 1)
        self.mean_acc = (preds == actions).sum().item() / self.agent.minibatch_size
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.accum_loss += self.mean_loss
        self.accum_acc += self.mean_acc
        self.accum_counter += 1

        self.confusion_matrix_queue.append((actions, preds))
        # self.true_actions = actions
        # self.pred_actions = preds
        # self.confusion_matrix = wandb.plot.confusion_matrix(y_true=actions.cpu().numpy(), preds=preds.cpu().numpy())

        return dict(loss=float(self.mean_loss), acc=float(self.mean_acc), confusion_matrix=self.confusion_matrix)

    def confusion_matrix(self):
        true_actions = np.ravel([item[0].cpu().numpy() for item in self.confusion_matrix_queue])
        pred_actions = np.ravel([item[1].cpu().numpy() for item in self.confusion_matrix_queue])
        return wandb.plot.confusion_matrix(y_true=true_actions, preds=pred_actions, 
                                            class_names=['noop','up','left','down','right'], title='NGU model performance')

class CTRL_module():
    name = 'CTRL'
    def __init__(self, model_args, optimizer, lr, agent, weight_normal,
                 episodic_max_size, k_neighbors, update_schedule,
                 num_envs, mem_size, batch_size, ir_model_copy):
        self.agent = agent
        self.weight_normal = weight_normal
        self.disentangle_network = DisentangleNetwork(**model_args).to(self.agent.device)
        self.test_embedding_fn = copy.deepcopy(self.disentangle_network).to(self.agent.device)
        self.optimizer = optimizer(self.disentangle_network.parameters(), lr=lr)
        self.episodic_memory = [EpisodicMemory(episodic_max_size, model_args['latent_size'], k_neighbors) for _ in range(num_envs)]
        self.memory = MainMemory(mem_size, batch_size, agent.device)
        self.num_envs = num_envs
        self.train_steps = 0
        self.update_schedule = update_schedule
        self.update_frequency = list(update_schedule.values())[0]
        self.model_copy = ir_model_copy
        self.accum_recon = 0
        self.accum_norm = 0
        self.accum_loss = 0
        self.accum_counter = 0

    def reset(self, end_vec):
        for i, e in enumerate(end_vec):
            if e:
                self.episodic_memory[i].reset()

    def compute_reward(self, obs_vec, action_vec):
        # get embedding
        obs = np.array(obs_vec)
        obs = self.agent.phi(obs) # [:, -1:, :, :] 
        obs = torch.as_tensor(obs, device=self.agent.device,
                            dtype=torch.float32)
        action = np.array(action_vec)
        action = torch.as_tensor(action, device=self.agent.device,
                            dtype=torch.int)
        self.test_embedding_fn.eval()
        with torch.no_grad():
            _, _, embeddings, _ = self.test_embedding_fn(obs, action)
        embeddings = embeddings.cpu().detach().numpy()
        reward_vec = []
        for i, embedding in enumerate(embeddings):
            # add memory to episodic memory and get score
            self.episodic_memory[i].add_item(embedding)
            reward = self.episodic_memory[i].score(embedding)

            reward_vec.append(reward)
        
        return np.array(reward_vec)

    def reset_stats(self):
        self.accum_recon = 0
        self.accum_norm = 0
        self.accum_loss = 0
        self.accum_counter = 0


    def train(self, time_step):
        if time_step % self.model_copy == 0:
            self.test_embedding_fn.load_state_dict(self.disentangle_network.state_dict())

        self.update_frequency = self.update_schedule[time_step] if time_step in self.update_schedule.keys() else self.update_frequency
        if time_step % self.update_frequency != 0:  # take into account multiple envs
            return dict()
            
        self.train_steps += 1
        self.disentangle_network.train()

        experiences = self.memory.uniform_sample()  # list of tuples of form (old obs, new obs, action, info)
        exp_batch = {
            "state": torch.as_tensor(self.agent.phi([elem[0] for elem in experiences]), device=self.agent.device),
            "next_state": torch.as_tensor(self.agent.phi([elem[1] for elem in experiences]), device=self.agent.device),
            "action": torch.as_tensor([elem[2] for elem in experiences], device=self.agent.device),
            "info": [elem[3] for elem in experiences],
        }
            
        observations = exp_batch["state"]
        next_observations = exp_batch["next_state"]
        actions = exp_batch["action"]
        assert observations.shape == next_observations.shape, f'mismatch in dims. observarions = {observations.shape}, next_observarions = {next_observations.shape}'
        total_effect = next_observations - observations

        # pdb.set_trace()
        controllable_effect, normal_effect, _, _ = self.disentangle_network(observations, actions)
        normal_reconstruction_loss = self.weight_normal * F.mse_loss(normal_effect, total_effect)
        total_reconstruction_loss = F.mse_loss(normal_effect + controllable_effect, total_effect)
        loss = total_reconstruction_loss + normal_reconstruction_loss

        self.next_observations = next_observations # for mask visualizations
        self.controllable_effect = controllable_effect # for mask visualizations
        self.normal_effect = normal_effect # for mask visualizations
        self.actions = actions
        self.infos = exp_batch['info']
        # self.wandb_imgs = sample_imgs_clusters(next_observations.detach().cpu().numpy(), controllable_effect.detach().cpu().numpy(), size=5)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.accum_loss += loss.item()
        self.accum_recon += total_reconstruction_loss.item()
        self.accum_norm += normal_reconstruction_loss.item()
        self.accum_counter += 1

        return dict(loss=float(loss.item()), total_loss=float(self.accum_recon), normal_loss=float(self.accum_norm))
    
    def sample_imgs_butterflies(self, size=5):
        next_observations = self.next_observations.detach().cpu().numpy()
        controllable_effect = self.controllable_effect.detach().cpu().numpy()
        normal_effect = self.normal_effect.detach().cpu().numpy()
        actions = self.actions.detach().cpu().numpy()

        infos = []
        events = []
        for info in self.infos:
            a_move_box = False
            event_list = []
            for item in info['History']:
                a_move_box = (a_move_box or (('catcher' in item['SourceObjectName']) and ('move' in item['ActionName']) and ('butterfly' in item['DestinationObjectName'])))
                event_list.append(f"{item['SourceObjectName']}-{item['ActionName']}-{item['DestinationObjectName']}")
            infos.append(a_move_box)
            events.append(event_list)
        events = np.array(events, dtype=object)
        infos = np.array(infos)

        indices = choose_idx(size, actions, infos)
        main_frames = next_observations[indices]
        masks = controllable_effect[indices]
        norm_masks = normal_effect[indices]
        actions = actions[indices]
        events = events[indices]
        full_imgs = [combine_frames(main_frame, mask, action, event, norm_mask) for main_frame, mask, norm_mask, action, event in zip(main_frames, masks, norm_masks, actions, events)]
        wandb_imgs = [wandb.Image(img) for img in full_imgs]
        return wandb_imgs

    def sample_imgs_clusters(self, size=5):
        next_observations = self.next_observations.detach().cpu().numpy()
        controllable_effect = self.controllable_effect.detach().cpu().numpy()
        actions = self.actions.detach().cpu().numpy()

        infos = []
        events = []
        for info in self.infos:
            a_move_box = False
            event_list = []
            for item in info['History']:
                a_move_box = (a_move_box or (('avatar' in item['SourceObjectName']) and ('move' in item['ActionName']) and ('box' in item['DestinationObjectName'])))
                event_list.append(f"{item['SourceObjectName']}-{item['ActionName']}-{item['DestinationObjectName']}")
            infos.append(a_move_box)
            events.append(event_list)
        events = np.array(events, dtype=object)
        infos = np.array(infos)

        indices = choose_idx(size, actions, infos)
        # indices = np.random.choice(range(len(next_observations)), size=size, replace=False)
        main_frames = next_observations[indices]
        masks = controllable_effect[indices]
        actions = actions[indices]
        events = events[indices]
        full_imgs = [combine_frames(main_frame, mask, action, event) for main_frame, mask, action, event in zip(main_frames, masks, actions, events)]
        wandb_imgs = [wandb.Image(img) for img in full_imgs]
        return wandb_imgs
    
class ALL_CTRL_module():
    name = 'ALL_CTRL'
    def __init__(self, model_args, optimizer, lr, agent, weight_normal,
                 episodic_max_size, k_neighbors, update_schedule,
                 num_envs, mem_size, batch_size, ir_model_copy):
        self.agent = agent
        self.weight_normal = weight_normal
        self.disentangle_network = ControlledNetwork(**model_args).to(self.agent.device)
        self.test_embedding_fn = copy.deepcopy(self.disentangle_network).to(self.agent.device)
        self.optimizer = optimizer(self.disentangle_network.parameters(), lr=lr)
        self.episodic_memory = [EpisodicMemory(episodic_max_size, model_args['latent_size'], k_neighbors) for _ in range(num_envs)]
        self.memory = MainMemory(mem_size, batch_size, agent.device)
        self.num_envs = num_envs
        self.train_steps = 0
        self.update_schedule = update_schedule
        self.update_frequency = list(update_schedule.values())[0]
        self.model_copy = ir_model_copy
        self.accum_recon = 0
        self.accum_norm = 0
        self.accum_loss = 0
        self.accum_counter = 0

    def reset(self, end_vec):
        for i, e in enumerate(end_vec):
            if e:
                self.episodic_memory[i].reset()

    def compute_reward(self, obs_vec, action_vec):
        # get embedding
        obs = np.array(obs_vec)
        obs = self.agent.phi(obs) # [:, -1:, :, :] 
        obs = torch.as_tensor(obs, device=self.agent.device,
                            dtype=torch.float32)
        action = np.array(action_vec)
        action = torch.as_tensor(action, device=self.agent.device,
                            dtype=torch.int)
        self.test_embedding_fn.eval()
        with torch.no_grad():
            _, embeddings = self.test_embedding_fn(obs, action)
        embeddings = embeddings.cpu().detach().numpy()
        reward_vec = []
        for i, embedding in enumerate(embeddings):
            # add memory to episodic memory and get score
            self.episodic_memory[i].add_item(embedding)
            reward = self.episodic_memory[i].score(embedding)

            reward_vec.append(reward)
        
        return np.array(reward_vec)

    def reset_stats(self):
        self.accum_recon = 0
        self.accum_norm = 0
        self.accum_loss = 0
        self.accum_counter = 0


    def train(self, time_step):
        if time_step % self.model_copy == 0:
            self.test_embedding_fn.load_state_dict(self.disentangle_network.state_dict())

        self.update_frequency = self.update_schedule[time_step] if time_step in self.update_schedule.keys() else self.update_frequency
        if time_step % self.update_frequency != 0:  # take into account multiple envs
            return dict()
            
        self.train_steps += 1
        self.disentangle_network.train()

        experiences = self.memory.uniform_sample()  # list of tuples of form (old obs, new obs, action, info)
        exp_batch = {
            "state": torch.as_tensor(self.agent.phi([elem[0] for elem in experiences]), device=self.agent.device),
            "next_state": torch.as_tensor(self.agent.phi([elem[1] for elem in experiences]), device=self.agent.device),
            "action": torch.as_tensor([elem[2] for elem in experiences], device=self.agent.device),
            "info": [elem[3] for elem in experiences],
        }
            
        observations = exp_batch["state"]
        next_observations = exp_batch["next_state"]
        actions = exp_batch["action"]
        assert observations.shape == next_observations.shape, f'mismatch in dims. observarions = {observations.shape}, next_observarions = {next_observations.shape}'
        total_effect = next_observations - observations

        controllable_effect, _ = self.disentangle_network(observations, actions)
        total_reconstruction_loss = F.mse_loss(controllable_effect, total_effect)
        loss = total_reconstruction_loss

        self.next_observations = next_observations # for mask visualizations
        self.controllable_effect = controllable_effect # for mask visualizations
        # self.normal_effect = normal_effect # for mask visualizations
        self.actions = actions
        self.infos = exp_batch['info']
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.accum_loss += loss.item()
        self.accum_recon += total_reconstruction_loss.item()
        self.accum_counter += 1

        return dict(loss=float(loss.item()), total_loss=float(self.accum_recon))

    def sample_imgs_clusters(self, size=5):
        next_observations = self.next_observations.detach().cpu().numpy()
        controllable_effect = self.controllable_effect.detach().cpu().numpy()
        actions = self.actions.detach().cpu().numpy()

        infos = []
        events = []
        for info in self.infos:
            a_move_box = False
            event_list = []
            for item in info['History']:
                a_move_box = (a_move_box or (('avatar' in item['SourceObjectName']) and ('move' in item['ActionName']) and ('box' in item['DestinationObjectName'])))
                event_list.append(f"{item['SourceObjectName']}-{item['ActionName']}-{item['DestinationObjectName']}")
            infos.append(a_move_box)
            events.append(event_list)
        events = np.array(events, dtype=object)
        infos = np.array(infos)

        indices = choose_idx(size, actions, infos)
        # indices = np.random.choice(range(len(next_observations)), size=size, replace=False)
        main_frames = next_observations[indices]
        masks = controllable_effect[indices]
        actions = actions[indices]
        events = events[indices]
        full_imgs = [combine_frames(main_frame, mask, action, event) for main_frame, mask, action, event in zip(main_frames, masks, actions, events)]
        wandb_imgs = [wandb.Image(img) for img in full_imgs]
        return wandb_imgs
    
    def sample_imgs_butterflies(self, size=5):
        next_observations = self.next_observations.detach().cpu().numpy()
        controllable_effect = self.controllable_effect.detach().cpu().numpy()
        actions = self.actions.detach().cpu().numpy()

        infos = []
        events = []
        for info in self.infos:
            a_move_box = False
            event_list = []
            for item in info['History']:
                a_move_box = (a_move_box or (('catcher' in item['SourceObjectName']) and ('move' in item['ActionName']) and ('butterfly' in item['DestinationObjectName'])))
                event_list.append(f"{item['SourceObjectName']}-{item['ActionName']}-{item['DestinationObjectName']}")
            infos.append(a_move_box)
            events.append(event_list)
        events = np.array(events, dtype=object)
        infos = np.array(infos)

        indices = choose_idx(size, actions, infos)
        main_frames = next_observations[indices]
        masks = controllable_effect[indices]
        actions = actions[indices]
        events = events[indices]
        full_imgs = [combine_frames(main_frame, mask, action, event) for main_frame, mask, action, event in zip(main_frames, masks, actions, events)]
        wandb_imgs = [wandb.Image(img) for img in full_imgs]
        return wandb_imgs

def choose_idx(size, actions, infos):
    indices = []
    # one from each action
    indices.extend(np.unique(actions, return_index=True)[1])
    # add some random samples
    indices.extend(np.random.choice(range(len(actions)), size=size//2, replace=False))
    # add some important events
    indices.extend(np.random.choice(np.where(infos)[0], size=min(size - size//2, np.sum(infos)), replace=False))
    return indices

action_name = {
    0: 'Noop',
    1: 'left',
    2: 'up',
    3: 'right',
    4: 'down'
}

def combine_frames(main_frame, mask, action, event, norm_mask=None):
    gray_mask = color.rgb2gray(mask.transpose(2,1,0))
    gray_norm_mask = color.rgb2gray(norm_mask.transpose(2,1,0))
    # bin_mask = gray_mask.round()
    if norm_mask is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    else:
        fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(main_frame.transpose(2,1,0))
    ax1.axis('off')
    # pdb.set_trace()
    ax1.set_title(f'action: {action_name[action]}\nmax_norm: {np.max(np.abs(gray_mask)):.3f}')
    ax2.imshow(gray_mask, cmap='gray')  #, vmin=0, vmax=1)
    ax2.axis('off')
    ax2.set_title('events: {}'.format("\n".join(event)))
    if norm_mask is not None:
        ax3.imshow(gray_norm_mask, cmap='gray') 
        ax3.axis('off')
        ax3.set_title(f'max_norm: {np.max(np.abs(gray_norm_mask)):.3f}')
    fig.canvas.draw()
    full_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.close(fig)
    return full_img