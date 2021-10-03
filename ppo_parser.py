import argparse

def create_parser():
    int_type = lambda x: int(float(x))
    parser = argparse.ArgumentParser(description="Arguments for PPO agent")
    # parser.add_argument('--tensorboard_dir_name', default='test', help='Name of tensorboard directory')
    parser.add_argument('--env_name', default='GDY-Clusters-Semi-Sparse-Wall-No-v0', help='Name of Env')
    parser.add_argument('--env_alias', default='clusters_semi_no_wall', help='Name of Env alias used in wandb')
    parser.add_argument('--successful_score', type=int_type, help='max env score')
    parser.add_argument('--max_frames', type=int_type, default=int_type(4000), help='max frames in an episode')
    parser.add_argument('--max_steps', type=int_type, default=int_type(7e6), help='max total env steps')
    parser.add_argument('--grad_norm', type=float, default=0.5, help='maximum gradient norm')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
    parser.add_argument('--batch_size', type=int_type, default=int_type(128), help='Batch Size')
    parser.add_argument('--warmup', type=int_type, default=int_type(10000), help='warmup steps before training')
    parser.add_argument('--discount', type=float, default=0.95, help='reward discount')
    parser.add_argument('--update_interval', type=int_type, default=int_type(30000), help='num of env steps per train step')
    parser.add_argument('--num_envs', type=int_type, default=int_type(16), help='number of PPO envs')
    parser.add_argument('--no_frame_stack', action='store_true', help='PPO frme stack flag')
    parser.add_argument('--epochs', type=int_type, default=int_type(10), help='ppo epochs')
    parser.add_argument('--outdir', default='ppo_results', help='output directory for saved agents and logs')
    parser.add_argument('--eval_n_runs', type=int_type, default=int_type(1), help='ppo evaluation number of runs')
    parser.add_argument('--eval_num_envs', type=int_type, default=int_type(4), help='ppo evaluation number of envs')
    parser.add_argument('--eval_max_frames', type=int_type, default=int_type(4000), help='max frames in an evaluation episode')
    parser.add_argument('--video_every', type=int_type, default=int_type(1000000), help='make a video every how many episodes')
    parser.add_argument('--log_freq', type=int_type, default=int_type(1000), help='log results every how many env steps')
    parser.add_argument('--checkpoint_frequency', type=int_type, default=int_type(500000), help='save the agent every how many env steps')
    parser.add_argument('--eval_interval', type=int_type, default=int_type(10000000), help='evaluate agent every how many steps')
    parser.add_argument('--evaluate', action='store_true', help='Calculate performance measure on evaluation env')
    parser.add_argument('--log_interval', type=int_type, default=int_type(10000), help='pfrl log frequency')
    parser.add_argument('--seed', type=int_type, default=int_type(42), help='random seed initializer')
    parser.add_argument('--wandb', action='store_true', help='use wandb for visualization')
    parser.add_argument('--wandb_project', default='clusters_ICLR', help='Name of wandb project')
    # NGU params
    parser.add_argument('--ir_beta', type=float, default=0.001, help='IR beta')
    parser.add_argument('--ir_warmup', type=int_type, default=int_type(10000), help='warmup steps before training IR')
    parser.add_argument('--ngu_embed_size', type=int_type, default=int_type(16), help='ngu module embedding size')
    parser.add_argument('--ngu_lr', type=float, default=0.0001, help='Learning Rate for ngu module')
    parser.add_argument('--ngu_update_schedule', type=eval, default='{150:1, 10000:1, 100000:4, 1000000:8}', help='ngu update schedule')
    parser.add_argument('--ngu_k_neighbors', type=int_type, default=int_type(10), help='ngu module K parameter')
    parser.add_argument('--ngu_L', type=int_type, default=int_type(5), help='ngu module L parameter')
    parser.add_argument('--ngu_mem', type=int_type, default=int_type(1e6), help='ngu memory size')
    # CTRL params
    parser.add_argument('--ctrl_hidden_size', type=int_type, default=int_type(32), help='ctrl module hidden size')
    parser.add_argument('--ctrl_channels', type=int_type, default=int_type(64), help='ctrl module channels')
    parser.add_argument('--ctrl_latent_size', type=int_type, default=int_type(16), help='ctrl module latent size')
    parser.add_argument('--ctrl_encoder_out', type=int_type, default=int_type(128), help='ctrl module latent size')
    parser.add_argument('--ctrl_weight_normal', type=float, default=0.01, help='ctrl module reconstruction loss weight')
    parser.add_argument('--ir_model_copy', type=int_type, default=int_type(10000), help='IR module embedding function copy')
    # IR selection: either ngu or ctrl
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ngu_reward', action='store_true', help='use ngu reward')
    group.add_argument('--ctrl_reward', action='store_true', help='use ctrl reward')

    parser.add_argument('--clip_rewards', action='store_true', help='Clip rewards to -1,0,1')
    parser.add_argument('--recurrent', action='store_true', help='Use an agent with recurrent memory')
    parser.add_argument('--punishment_scale', type=float, default=0.02, help='clusters env punishment')

    return parser