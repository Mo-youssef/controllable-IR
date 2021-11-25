#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export GRIDDLY_DEVICE_ORDER=$CUDA_DEVICE_ORDER
export GRIDDLY_VISIBLE_DEVICES=3
echo $GRIDDLY_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GRIDDLY_VISIBLE_DEVICES
# lower punishement 0.1 --> 0.05  // 0.0005 --> 0.0001, since increased BS from 512 -> 1024, LR should be 0.0005 * sqrt(2) = 0.0007
# discount 0.95 --> 0.9
# PPO rollout from 2048 --> 10000
/home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/iclr_thesis/main_ppo.py \
    --env_name=GDY-Butterflies-Spiders-Easy-v0 \
    --env_alias=butterflies \
    --wandb_project=butterflies_ICLR \
    --max_frames=200 \
    --max_steps=5e6 \
    --grad_norm=0.5 \
    --lr=0.0005 \
    --batch_size=128 \
    --warmup=10000 \
    --discount=0.95 \
    --update_interval=10000 \
    --num_envs=16 \
    --epochs=10 \
    --outdir=ICLR_BS \
    --eval_n_runs=1 \
    --eval_num_envs=4 \
    --eval_max_frames=4000 \
    --checkpoint_frequency=50000000 \
    --eval_interval=10000000 \
    --video_every=1000000 \
    --log_interval=10000 \
    --seed=200 \
    --ir_beta=0.5 \
    --ir_warmup=10000 \
    --ngu_embed_size=64 \
    --ngu_lr=1e-4 \
    --ngu_update_schedule='{150:1, 10000:1, 100000:4, 1000000:8}' \
    --ir_model_copy=10000 \
    --ngu_k_neighbors=10 \
    --ngu_L=5 \
    --ngu_mem=30000 \
    --ctrl_hidden_size=64 \
    --ctrl_channels=64 \
    --ctrl_latent_size=64 \
    --ctrl_encoder_out=128 \
    --ctrl_weight_normal=0.01 \
    --no_frame_stack \
    --wandb \
    --ngu_reward \
    --no_ext_reward \
    --recurrent \
    |& tee butterflies.out
    # --ctrl_reward \
    # --all_ctrl_reward \
    # --punishment_scale=0 \
    # --evaluate \
    # --clip_rewards \ 
    # --successful_score=1 \