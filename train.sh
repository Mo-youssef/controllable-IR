#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export GRIDDLY_DEVICE_ORDER=$CUDA_DEVICE_ORDER
export GRIDDLY_VISIBLE_DEVICES=3
echo $GRIDDLY_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GRIDDLY_VISIBLE_DEVICES
# lower punishement 0.1 --> 0.05  // 0.0005 --> 0.0001, since increased BS from 512 -> 1024, LR should be 0.0005 * sqrt(2) = 0.0007
# discount 0.95 --> 0.9
# PPO rollout from 2048 --> 10000
/home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py \
    --env_name=GDY-Clusters-Semi-Sparse-Wall-No-v0 \
    --env_alias=clusters_semi_no_wall \
    --wandb_project=clusters_ICLR \
    --max_frames=10000 \
    --max_steps=7e6 \
    --grad_norm=0.5 \
    --lr=0.0005 \
    --batch_size=512 \
    --warmup=10000 \
    --discount=0.95 \
    --update_interval=2048 \
    --num_envs=16 \
    --epochs=10 \
    --outdir=ICLR \
    --eval_n_runs=1 \
    --eval_num_envs=4 \
    --eval_max_frames=4000 \
    --checkpoint_frequency=500000 \
    --eval_interval=10000000 \
    --video_every=1000000 \
    --log_interval=10000 \
    --seed=300 \
    --ir_beta=0.001 \
    --ir_warmup=10000 \
    --ngu_embed_size=16 \
    --ngu_lr=1e-4 \
    --ngu_update_schedule='{150:1, 10000:1, 100000:4, 1000000:8}' \
    --ir_model_copy=10000 \
    --ngu_k_neighbors=10 \
    --ngu_L=5 \
    --ngu_mem=1e6 \
    --ctrl_hidden_size=32 \
    --ctrl_channels=64 \
    --ctrl_latent_size=16 \
    --ctrl_encoder_out=128 \
    --ctrl_weight_normal=0.01 \
    --no_frame_stack \
    --wandb \
    --punishment_scale=0.01 \
    --ctrl_reward \
    |& tee cluster_vanilla2.out
    # --ngu_reward \
    # --recurrent \
    # --evaluate \
    # --clip_rewards \ 
    # --successful_score=1 \