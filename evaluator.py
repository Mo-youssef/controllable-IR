import torch
import wandb
import pfrl
import datetime
import PIL
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import sklearn.metrics
from sklearn.model_selection import train_test_split

from skimage.transform import resize
from collections import defaultdict, deque
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from timing import Timing
from ppo_ir_modules import Eval_intrinsic_module
import pickle
import uuid

import pdb

class ProbingDataset():
    def __init__(self, max_size):
        self.max_size = max_size
        self.min_len = 0
        self.avatar_memory = deque(maxlen=max_size)
        self.avatar_only_memory = deque(maxlen=max_size)
        self.box_memory = deque(maxlen=max_size)
    def add(self, item, type):
        '''
        item: tuple of form (observation, location, action)
        '''
        if type == 'a':
            self.avatar_memory.append(item)
        elif type == 'b':
            self.box_memory.append(item)
        elif type == 'am':
            self.avatar_only_memory.append(item)
    def reset(self):
        self.avatar_memory.clear()
        self.avatar_only_memory.clear()
        self.box_memory.clear()
        self.min_len = 0
    def ready(self):
        self.min_len = min(len(self.avatar_memory), len(self.box_memory))
        return self.min_len >= self.max_size
    def get_splits(self, module=None, test_size=0.2):
        x, y_avatar, a = [], [], []
        for item in self.avatar_memory:
            x.append(item[0])
            y_avatar.append(item[1])
            a.append(item[2])
        y_box = []
        for item in self.box_memory:
            x.append(item[0])
            y_box.append(item[1])
            a.append(item[2])
        y_avatar_only = []
        for item in self.avatar_only_memory:
            x.append(item[0])
            y_avatar_only.append(item[1])
            a.append(item[2])
        if module is not None:
            embeddings = module(x) if module.name == 'NGU' else module(x, a)
            avatar_data = train_test_split(embeddings[:len(y_avatar)], y_avatar, test_size=test_size)
            box_data = train_test_split(embeddings[len(y_avatar):len(y_avatar)+len(y_box)], y_box, test_size=test_size)
            avatar_only_data = train_test_split(embeddings[-len(y_avatar_only):], y_avatar_only, test_size=test_size)
        else:
            avatar_data = train_test_split(x[:len(y_avatar)], y_avatar, a[:len(y_avatar)], test_size=test_size)
            box_data = train_test_split(x[len(y_avatar):len(y_avatar)+len(y_box)], y_box, a[len(y_avatar):len(y_avatar)+len(y_box)], test_size=test_size)
            avatar_only_data = train_test_split(x[-len(y_avatar_only):], y_avatar_only, a[-len(y_avatar_only):], test_size=test_size)
        return avatar_data, box_data, avatar_only_data
    def evaluate_module(self, module, unique_id):
        avatar_data, box_data, avatar_move = self.get_splits(module=module)
        avatar_train = avatar_data[0], avatar_data[2]
        avatar_test = avatar_data[1], avatar_data[3]
        box_train = box_data[0], box_data[2]
        box_test = box_data[1], box_data[3]
        avatar_model = SVC().fit(avatar_train[0], avatar_train[1]) # if changed to NN make sure to keep embeddings in GPU memory
        box_model = SVC().fit(box_train[0], box_train[1])
        avatar_f1 = sklearn.metrics.f1_score(avatar_test[1], avatar_model.predict(avatar_test[0]), average='weighted')
        box_f1 = sklearn.metrics.f1_score(box_test[1], box_model.predict(box_test[0]), average='weighted')

        # plot tsne
        min_len = min(len(avatar_move[0]), len(box_data[0]))
        avatar_embeddings = avatar_move[0][:min_len]
        avatar_labels = np.ones(len(avatar_embeddings))[:min_len]
        box_embeddings = box_data[0][:min_len]
        box_labels = np.ones(len(box_embeddings))[:min_len] * 2
        tsne_embeddings = np.concatenate([avatar_embeddings, box_embeddings], axis=0)
        tsne_labels = np.concatenate([avatar_labels, box_labels])
        with open('tsne_data/'+unique_id+'.pkl', 'wb') as f:
            pickle.dump((tsne_embeddings, tsne_labels), f)
        tsne_mappings = TSNE(n_components=2).fit_transform(tsne_embeddings)
        separation_score = calculate_separation(tsne_mappings, tsne_labels) 
        title = f'Separation Score: f1={separation_score[0]:.5f}, Acc={separation_score[1]:.5f}'
        wnb_tsne_plot = plot_scatter_plotly(tsne_mappings, tsne_labels, title)

        return dict(avatar_f1=avatar_f1, box_f1=box_f1, balanced_TSNE=wnb_tsne_plot)
    # def tsne_plot(self, module):
    #     _, box_data, avatar_data = self.get_splits(module=module)
    #     min_len = min(len(avatar_data[0]), len(box_data[0]))
    #     avatar_embeddings = avatar_data[0][:min_len]
    #     avatar_labels = np.zeros(len(avatar_embeddings))[:min_len]
    #     box_embeddings = box_data[0][:min_len]
    #     box_labels = np.ones(len(box_embeddings))[:min_len]
    #     tsne_embeddings = np.concatenate([avatar_embeddings, box_embeddings], axis=0)
    #     tsne_labels = np.concatenate([avatar_labels, box_labels])
    #     tsne_mappings = TSNE(n_components=2).fit_transform(tsne_embeddings)
    #     separation_score = calculate_separation(tsne_mappings, tsne_labels) 
    #     title = f'Separation Score: f1={separation_score[0]:.5f}, Acc={separation_score[1]:.5f}'
    #     wnb_tsne_plot = plot_scatter_plotly(tsne_mappings, tsne_labels, title)
    #     return dict(balanced_TSNE=wnb_tsne_plot)

class Evaluator:

    def __init__(self, env, agent, num_episodes, eval_frequency, video_frequency, intrinsic_model=None, unique_id=None):
        self.env = env
        self.vec_env = isinstance(env, pfrl.env.VectorEnv) # whether env is vectorized or normal
        self.agent = agent
        self.num_episodes = num_episodes
        self.device = agent.device
        self.eval_frequency = eval_frequency
        self.video_frequency = video_frequency
        self.last_eval = 0
        self.last_video = 0
        self.record_video = True
        self.intrinsic_model = None
        if intrinsic_model:
            eval_num_envs = env.num_envs if self.vec_env else 1
            self.intrinsic_model = Eval_intrinsic_module(encoder=intrinsic_model.test_embedding_fn, agent=agent,
                                            episodic_max_size=intrinsic_model.episodic_memory[0].max_size,
                                            embedding_size=intrinsic_model.episodic_memory[0].embedding_size,
                                            k_neighbors=intrinsic_model.episodic_memory[0].k_neighbors,
                                            num_envs=eval_num_envs)
            self.intrinsic_model.reset(np.ones(eval_num_envs))
            self.intrinsic_model.name = intrinsic_model.name
        self.probing_dataset = ProbingDataset(max_size=25000)
        self.unique_id = str(uuid.uuid4())[:8] if unique_id is None else unique_id
        # self.probing_dataset = None

    def force_evaluate(self):
        eval_stats = dict()
        with self.agent.eval_mode():
            if self.vec_env:
                self.num_envs = self.env.num_envs
                eval_stats = self.evaluate_vec_env()
            else:
                eval_stats = self.evaluate_env()
        self.record_video = False
        log_str = (f"[{datetime.datetime.now():%Y-%m-%d %T}] Evaluation:"
                    f" average_R:{eval_stats['eval_extrinsic_reward']:.4f}"
                    f" episode length:{eval_stats['eval_episode_length']:.4f}"
                    f" average_IR:{eval_stats['eval_intrinsic_reward']:.4f}")
        print(log_str, flush=True)
        if (self.intrinsic_model is not None) and (self.probing_dataset is not None) and (self.probing_dataset.ready()):
            scores = self.probing_dataset.evaluate_module(self.intrinsic_model, self.unique_id)
            log_str = (f"[{datetime.datetime.now():%Y-%m-%d %T}] Probing Scores:"
                        f" avatar f1:{scores['avatar_f1']:.4f}"
                        f" box f1:{scores['box_f1']:.4f}")
            print(log_str, flush=True)
            eval_stats.update(scores)

            # tsne_plot = self.probing_dataset.tsne_plot(self.intrinsic_model)
            # eval_stats.update(tsne_plot)

            self.probing_dataset.reset()
        return eval_stats        

    def evaluate_if_necessary(self, step):
        if step - self.last_video >= self.video_frequency:
            self.record_video = True
            self.last_video = step
        eval_stats = dict()
        if step - self.last_eval >= self.eval_frequency:
            self.last_eval = step
            with self.agent.eval_mode():
                if self.vec_env:
                    self.num_envs = self.env.num_envs
                    eval_stats = self.evaluate_vec_env()
                else:
                    eval_stats = self.evaluate_env()
            self.record_video = False
            log_str = (f"[{datetime.datetime.now():%Y-%m-%d %T}] Evaluation @ step:{step}"
                       f" average_R:{eval_stats['eval_extrinsic_reward']:.4f}"
                       f" episode length:{eval_stats['eval_episode_length']:.4f}"
                       f" average_IR:{eval_stats['eval_intrinsic_reward']:.4f}")
            print(log_str, flush=True)
        if (self.intrinsic_model is not None) and (self.probing_dataset is not None) and (self.probing_dataset.ready()):
            scores = self.probing_dataset.evaluate_module(self.intrinsic_model, self.unique_id)
            log_str = (f"[{datetime.datetime.now():%Y-%m-%d %T}] Probing Scores @ step:{step}"
                        f" avatar f1:{scores['avatar_f1']:.4f}"
                        f" box f1:{scores['box_f1']:.4f}")
            print(log_str, flush=True)
            eval_stats.update(scores)

            # tsne_plot = self.probing_dataset.tsne_plot(self.intrinsic_model)
            # eval_stats.update(tsne_plot)

            self.probing_dataset.reset()
        return eval_stats

    def evaluate_vec_env(self):
        stats = dict(
            extrinsic_reward=[],
            intrinsic_reward=[],
            values=[],
            episode_length=[],
            events_intrinsic=defaultdict(float),
            events_count=defaultdict(int),
        )

        timing = dict()
        videos = list()
        # tsne_embeddings = list()
        # tsne_labels = list()
        for _ in range(self.num_episodes):
            observations = self.env.reset()

            history_extrinsic_reward = list()
            history_intrinsic_reward = list()
            history_value = list()

            frames = list()
            if self.record_video:
                info_frame = self._to_info_frame(None, None, None)
                frames.append(self._merge_info_and_obs(info_frame, observations[-1].copy()))
            
            end = np.zeros(self.num_envs, dtype=bool)

            stats['extrinsic_reward'].append(0.)
            stats['intrinsic_reward'].append(0.)
            stats['episode_length'].append(0.)
            stats['values'].append(0.)

            while not np.all(end):
                with Timing(timing, 'time_evaluator_act'):
                    actions = self.agent.batch_act(observations)
                    value = self.agent.value_record[-1]
                old_observations = observations
                with Timing(timing, 'time_evaluator_step'):
                    observations, rewards, dones, infos = self.env.step(actions)

                if self.intrinsic_model is not None:
                    with Timing(timing, 'IR_model_step'):
                        # self.intrinsic_model.reset(end)     
                        if self.intrinsic_model.name == 'NGU':
                            intrinsic_rewards, ir_embeddings = self.intrinsic_model.compute_reward(observations)
                        elif self.intrinsic_model.name == 'CTRL':
                            intrinsic_rewards, ir_embeddings = self.intrinsic_model.compute_reward(old_observations, actions)
                else:
                    intrinsic_rewards = np.zeros(self.num_envs)

# events = list()
# for item in info['History']:
#     events.append('{}-{}-{}'.format(item['SourceObjectName'], item['ActionName'], item['DestinationObjectName']))
#     if (('box' in item['SourceObjectName']) and ('wall' in item['DestinationObjectName'])):
#         events.append('box-hit-wall')

# return events or ['nothing']  

                # tsne_label = []
                for intrinsic_reward, info, obs, action in zip(intrinsic_rewards, infos, observations, actions):
                    state =  info[1]
                    info = info[0]      # This is done because eval env returns: obs, r, done, (info, state)
                    event_items, events_details = self._get_griddly_events(info, return_details=True)
                    # move_event = 0
                    for event, details in zip(event_items, events_details):
                        stats['events_intrinsic'][event] += intrinsic_reward
                        stats['events_count'][event] += 1.0
                        if (event == 'avatar-move-_empty') or (event == 'avatar-move-near_wall'):
                            # move_event = 1
                            if (self.probing_dataset is not None):
                                avatar_loc = get_location(state, 'avatar', details)
                                self.probing_dataset.add((obs, avatar_loc, action), 'a')  # Action is added for the CTRL module 
                                self.probing_dataset.add((obs, avatar_loc, action), 'am')
                        elif (event == 'avatar-move-blue_box') or (event == 'avatar-move-red_box') or (event == 'avatar-move-green_box'):
                            # move_event = 2
                            if (self.probing_dataset is not None):
                                avatar_loc, box_loc = get_location(state, 'box', details)
                                self.probing_dataset.add((obs, avatar_loc, action), 'a')
                                self.probing_dataset.add((obs, box_loc, action), 'b')
                    # tsne_label.append(move_event)
                # tsne_label = np.array(tsne_label)
                # if (self.intrinsic_model is not None) and (np.sum(tsne_label)>0):
                #     tsne_labels.append(tsne_label[np.where(tsne_label)])
                #     tsne_embeddings.append(ir_embeddings[np.where(tsne_label)])

                with Timing(timing, 'gathering_stats_step'):
                    stats['intrinsic_reward'][-1] += np.sum(np.array(intrinsic_rewards)[~end]) / self.num_envs
                    stats['episode_length'][-1] += np.sum(~end) / self.num_envs
                    stats['extrinsic_reward'][-1] += np.sum(np.array(rewards)[~end]) / self.num_envs
                    history_extrinsic_reward.append(rewards[-1])
                    history_intrinsic_reward.append(intrinsic_rewards[-1])
                    history_value.append(value)

                    if ((not end[-1]) and self.record_video):
                        info_frame = self._to_info_frame(history_extrinsic_reward, history_intrinsic_reward, history_value)
                        frames.append(self._merge_info_and_obs(info_frame, observations[-1].copy()))

                # Reset environments and IR module
                end = np.logical_or.reduce((end, dones, [info[0].get("needs_reset", False) for info in infos])) # info[0] is done because eval env returns: obs, r, done, (info, state)
                _ = self.env.reset(~end)
                if self.intrinsic_model is not None:
                    self.intrinsic_model.reset(end)  

            if self.record_video:
                with Timing(timing, 'making_video_step'):
                    videos.append(wandb.Video(np.stack(frames).astype(np.uint8), fps=4, format="gif"))


        wnb_table_intrinsic = wandb.Table(
            data=[(k, v / stats['events_count'][k]) for k, v in stats['events_intrinsic'].items()],
            columns=["event", "intrinsic_reward"]
        )
        wnb_events_ir_time = plot_intrinsic(stats)
        wnb_events_counts_time = plot_counts(stats)
        wnb_table_events = wandb.Table(
            data=[(k, (stats['events_count'][k])/np.mean(stats['episode_length'])) for k in stats['events_intrinsic'].keys()],
            columns=["event", "count"]
        )
        # wnb_tsne_plot = None
        # tsne_labels = np.concatenate(tsne_labels, axis=0) if len(tsne_labels) > 0 else tsne_labels
        # if (self.intrinsic_model is not None) and (len(tsne_labels) > 1):
        #     with Timing(timing, 'TSNE_mapping'):
        #         tsne_embeddings = np.concatenate(tsne_embeddings, axis=0)
        #         tsne_mappings = TSNE(n_components=2).fit_transform(tsne_embeddings)
        #         mean_distance = np.linalg.norm(np.mean(tsne_embeddings[tsne_labels==1]) - np.mean(tsne_embeddings[tsne_labels==2]))
        #         separation_score = calculate_separation(tsne_mappings, tsne_labels) 
        #         title = f'Mean distance between events= {mean_distance:.8f}<br>Separation Score: f1={separation_score[0]:.5f}, Acc={separation_score[1]:.5f}'
        #         wnb_tsne_plot = plot_scatter_plotly(tsne_mappings, tsne_labels, title)

        ret_dict = dict(
            eval_extrinsic_reward=np.mean(stats['extrinsic_reward']),
            eval_intrinsic_reward=np.mean(stats['intrinsic_reward']),
            eval_episode_length=np.mean(stats['episode_length']),
            eval_events_intrinsic=wandb.plot.bar(wnb_table_intrinsic, "event", "intrinsic_reward", title="Intrinsic Rew. per event"),
            eval_events_count=wandb.plot.bar(wnb_table_events, "event", "count", title="Count per event"),
            # eval_tsne=wnb_tsne_plot,
            eval_events_intrinsic_time=wnb_events_ir_time,
            eval_events_counts_time=wnb_events_counts_time,
            **{k: v['time'] / v['count'] for k, v in timing.items()},
        )
        if self.record_video:
            ret_dict['video']=videos

        return ret_dict

    # def evaluate_env(self):
    #     stats = dict(
    #         extrinsic_reward=[0.],
    #         intrinsic_reward=[0.],
    #         values=[0.],
    #         episode_length=[0],
    #         action_cm=(list(), list()),
    #         action_accuracy=list(),
    #         events_intrinsic=defaultdict(float),
    #         events_count=defaultdict(int),
    #     )

    #     timing = dict()
    #     videos = list()
    #     for _ in range(self.num_episodes):
    #         observation = self.env.reset()

    #         done = False
    #         history_extrinsic_reward = list()
    #         history_intrinsic_reward = list()
    #         history_value = list()

    #         frames = list()
    #         info_frame = self._to_info_frame(None, None, None)
    #         frames.append(self._merge_info_and_obs(info_frame, observation.copy()))
    #         while not done:
    #             with Timing(timing, 'time_evaluator_act'):
    #                 action = self.agent.act(observation)
    #                 value = self.agent.value_record[-1]

    #             with Timing(timing, 'time_evaluator_step'):
    #                 observation, reward, done, info = self.env.step(action)
    #                 done = done or info.get("needs_reset", False)

    #             if self.intrinsic_model is not None:
    #                 self.intrinsic_model.reset([done])      

    #                 intrinsic_reward = self.intrinsic_model.compute_reward([observation])[0]

    #                 stats['intrinsic_reward'][-1] += intrinsic_reward
    #             else:
    #                 intrinsic_reward = 0.0

    #             for event in self._get_griddly_events(info):
    #                 stats['events_intrinsic'][event] += intrinsic_reward
    #                 stats['events_count'][event] += 1

    #             stats['episode_length'][-1] += 1
    #             stats['extrinsic_reward'][-1] += reward
    #             history_extrinsic_reward.append(reward)
    #             history_intrinsic_reward.append(intrinsic_reward)
    #             history_value.append(value)

    #             info_frame = self._to_info_frame(history_extrinsic_reward, history_intrinsic_reward, history_value)
    #             frames.append(self._merge_info_and_obs(info_frame, observation.copy()))

    #         # if self.intrinsic_model is not None and hasattr(self.intrinsic_model, 'reset'):
    #             # with Timing(timing, 'time_evaluator_intrinsic_reset'):
    #             #     self.intrinsic_model.reset(env_index=0)

    #         videos.append(wandb.Video(np.stack(frames).astype(np.uint8), fps=4, format="gif"))

    #         stats['extrinsic_reward'].append(0)
    #         stats['intrinsic_reward'].append(0)
    #         stats['episode_length'].append(0)
    #         stats['values'].append(0)

    #     wnb_table_intrinsic = wandb.Table(
    #         data=[(k, v / stats['events_count'][k]) for k, v in stats['events_intrinsic'].items()],
    #         columns=["event", "intrinsic_reward"]
    #     )

    #     return dict(
    #         video=videos,
    #         eval_extrinsic_reward=np.mean(stats['extrinsic_reward']),
    #         eval_intrinsic_reward=np.mean(stats['intrinsic_reward']),
    #         # eval_action_cm=wandb.plot.confusion_matrix(y_true=stats['action_cm'][1], preds=stats['action_cm'][0]),
    #         eval_episode_length=np.mean(stats['episode_length']),
    #         # eval_action_accuracy=np.mean(stats['action_accuracy']),
    #         eval_events_intrinsic=wandb.plot.bar(wnb_table_intrinsic, "event", "intrinsic_reward", title="Intrinsic Rew. per event"),
    #         **{k: v['time'] / v['count'] for k, v in timing.items()},
    #     )

    def _get_griddly_events(self, info, return_details=False):
        events = list()
        events_details = list()
        for item in info['History']:
            events.append('{}-{}-{}'.format(item['SourceObjectName'], item['ActionName'], item['DestinationObjectName']))
            events_details.append(item)
            if (('box' in item['SourceObjectName']) and ('wall' in item['DestinationObjectName'])):
                events.append('box-hit-wall')
                events_details.append(item)
        if return_details:
            return events or ['nothing'], events_details or ['nothing']
        return events or ['nothing']

    def _to_info_frame(self, extrinsic_rewards, intrinsic_rewards, values):
        fig = Figure(figsize=(8, 8), dpi=30)
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        if extrinsic_rewards:
            ax.plot(range(len(extrinsic_rewards)), extrinsic_rewards, color='green', label='extrinsic')
            ax.plot(range(len(intrinsic_rewards)), intrinsic_rewards, color='red', label='intrinsic')
            ax.plot(range(len(values)), values, color='blue', label='value')

        ax.patch.set_alpha(0)
        ax.legend()

        canvas.draw()
        w, h = canvas.get_width_height()
        info_image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8)
        info_image.shape = (w, h, 3)
        info_image = Image.frombytes("RGB", (w, h), info_image.tostring())
        plt.close(fig)
        return np.asarray(info_image).transpose(2, 0, 1)

    def _merge_info_and_obs(self, info_frame, obs_frame):
        obs_frame_scaled = resize(obs_frame.transpose(1, 2, 0), info_frame.shape[1:], preserve_range=True).transpose(2, 0, 1)
        return np.concatenate([obs_frame_scaled.astype(np.uint8), info_frame], axis=2)

def get_location(state, obj, details):
    '''
    state: internal state of the env
    obj: One of the object in the env, typically `avatar` or `box`
    '''
    avatar_src = np.array(details['SourceLocation'])
    avatar_dst = np.array(details['DestinationLocation'])
    move_dir = avatar_dst - avatar_src
    for o in state['Objects']:
        if o['Name'] == 'avatar':
            avatar_loc = (o['Location'][0] - 1) * 8 + o['Location'][1]
            target_loc = np.array(o['Location']) + move_dir
            dst_loc = (target_loc[0] - 1) * 8 + target_loc[1]
            if obj == 'avatar':
                return avatar_loc
            return avatar_loc, dst_loc

def plot_scatter_plotly(data, labels, title):
    assert len(data.T) == 2, 'data should be of shape (N,2)'
    labels_idxs = np.argsort(labels)
    labels = labels[labels_idxs]
    data = data[labels_idxs]
    label_dict = {1:'avatar-move-empty', 2:'avatar-move-box'}
    labels = np.array(list(map(lambda i: label_dict[i], labels)))
    fig = px.scatter(x=data[:,0], y=data[:,1], color=labels, opacity=0.7, title=title)
    return fig

def plot_counts(stats):
    data = [(k, (stats['events_count'][k])/np.mean(stats['episode_length'])) for k in stats['events_intrinsic'].keys()]
    columns=["event", "count"]
    df = pd.DataFrame(data, columns=columns)
    df = df.sort_values(by=['event'], ascending=False)
    fig = px.bar(df, x="count", y="event", orientation='h')
    return fig

def plot_intrinsic(stats):
    # pdb.set_trace()
    data=[(k, v / stats['events_count'][k]) for k, v in stats['events_intrinsic'].items()]
    columns=["event", "intrinsic_reward"]
    df = pd.DataFrame(data, columns=columns)
    df = df.sort_values(by=['event'], ascending=False)
    fig = px.bar(df, x="intrinsic_reward", y="event", orientation='h')
    return fig

def calculate_separation(data, labels):
    '''
    return F1, Accuracy
    '''
    if len(np.unique(labels)) < 2:
        return 0., 0.
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)
    model = SVC().fit(x_train, y_train)
    f1 = sklearn.metrics.f1_score(y_test, model.predict(x_test))
    acc = model.score(x_test, y_test)
    return f1, acc

def plot_scatter(data, labels, classes=None):
                # plt.close(wnb_tsne_plot)
                # wnb_tsne_plot = plot_scatter(tsne_mappings, tsne_labels, classes=['avatar-move-_empty', 'avatar-move-box'])
                # tsne_plot = plot_scatter(tsne_mappings, tsne_labels, classes=['avatar-move-_empty', 'avatar-move-box'])
                # wnb_tsne_plot = wandb.Image(tsne_plot, caption=None)
    assert len(data.T) == 2, 'data should be of shape (N,2)'
    fig, ax = plt.subplots()    #(figsize=(8,8))
    # scatter = ax.scatter(*data.T, c=labels, marker='.', alpha=0.4, edgecolors='None') 
    scatter = ax.scatter(*data.T, c=labels, marker='o', alpha=0.4, edgecolors='None', cmap='PiYG') 
    classes = classes if classes else np.unique(labels)
    assert len(classes) == len(np.unique(labels)), 'Number of classes must equal unique labels'
    legend1 = ax.legend(scatter.legend_elements()[0], classes) #, framealpha=0.3)
    ax.add_artist(legend1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True)
    # fig.canvas.draw()
    # full_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    # plt.close(fig)
    return fig