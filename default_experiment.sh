#!/bin/bash
env_name=GDY-Clusters-Semi-Sparse-Wall-No-v0
wandb_project=clusters_ICLR
outdir=ICLR
##################################################################################################################################
(
    export CUDA_VISIBLE_DEVICES=0
    echo $CUDA_VISIBLE_DEVICES
    export GRIDDLY_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    s=42
    env_alias=ctrl_$s
    echo starting experiment $env_alias
    /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --outdir=$outdir --no_frame_stack --evaluate --wandb \
        --ctrl_reward &> $env_alias.out 

    # s=200
    # env_alias=ctrl_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --ctrl_reward &> $env_alias.out

    # env_alias=ngu_lstm_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --recurrent --ngu_reward &> $env_alias.out 


#     # s=42
#     # env_alias=ctrl_$s
#     # echo starting experiment $env_alias
#     # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
#     #     --ctrl_reward &> $env_alias.out
) &
##################################################################################################################################
(
    sleep 1m
    export CUDA_VISIBLE_DEVICES=1
    echo $CUDA_VISIBLE_DEVICES
    export GRIDDLY_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    s=42
    env_alias=ngu_$s
    echo starting experiment $env_alias
    /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --outdir=$outdir --no_frame_stack --evaluate --wandb \
        --ngu_reward &> $env_alias.out 

    # env_alias=ctrl_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --ctrl_reward &> $env_alias.out 

    # s=100
    # env_alias=ngu_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --ngu_reward &> $env_alias.out

    # env_alias=ngu_lstm_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --recurrent --ngu_reward &> $env_alias.out

    # s=42
    # env_alias=ngu_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --ngu_reward &> $env_alias.out

    # env_alias=vanilla_lstm_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --recurrent &> $env_alias.out 
) &
# ##################################################################################################################################
(
    sleep 1m
    export CUDA_VISIBLE_DEVICES=2
    echo $CUDA_VISIBLE_DEVICES
    export GRIDDLY_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    s=100
    env_alias=ctrl_$s
    echo starting experiment $env_alias
    /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --outdir=$outdir --no_frame_stack --evaluate --wandb \
        --ctrl_reward &> $env_alias.out 

    # env_alias=ngu_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --ngu_reward &> $env_alias.out 

    # s=100
    # env_alias=vanilla_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     &> $env_alias.out

    # env_alias=vanilla_lstm_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --recurrent &> $env_alias.out

    # s=42
    # env_alias=ctrl_lstm_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --recurrent --ctrl_reward &> $env_alias.out 

    # env_alias=vanilla_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     &> $env_alias.out
) &
# ##################################################################################################################################
(
    sleep 1m
    export CUDA_VISIBLE_DEVICES=3
    echo $CUDA_VISIBLE_DEVICES
    export GRIDDLY_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    s=100
    env_alias=ngu_$s
    echo starting experiment $env_alias
    /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --outdir=$outdir --no_frame_stack --evaluate --wandb \
        --ngu_reward &> $env_alias.out 

    # env_alias=ctrl_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --ctrl_reward &> $env_alias.out 
    # s=1
    # env_alias=vanilla_lstm_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --recurrent &> $env_alias.out 

    # env_alias=ctrl_lstm_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --recurrent --ctrl_reward &> $env_alias.out 

    # s=100
    # env_alias=ctrl_lstm_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --recurrent --ctrl_reward &> $env_alias.out 

    # s=42
    # env_alias=ngu_lstm_$s
    # echo starting experiment $env_alias
    # /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=$s --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
    #     --recurrent --ngu_reward &> $env_alias.out
) &
wait
# ##################################################################################################################################
# export CUDA_VISIBLE_DEVICES=0
# echo $CUDA_VISIBLE_DEVICES
# export GRIDDLY_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
# env_alias=ngu_lstm
# /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=42 --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
#     --recurrent --ngu_reward &> $env_alias.out &
# sleep 2m
# # ##################################################################################################################################
# export CUDA_VISIBLE_DEVICES=1
# echo $CUDA_VISIBLE_DEVICES
# export GRIDDLY_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
# env_alias=ctrl_lstm
# /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=42 --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
#     --recurrent --ctrl_reward &> $env_alias.out &

# wait
# ##################################################################################################################################
# export CUDA_VISIBLE_DEVICES=2
# export GRIDDLY_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
# env_alias=cluster_vanilla_lstm 
# /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=42 --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
#     --recurrent --ctrl_reward --ngu_reward &> $env_alias.out
# ##################################################################################################################################
# export CUDA_VISIBLE_DEVICES=3
# export GRIDDLY_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
# env_alias=cluster_vanilla_lstm 
# /home/mohameys/miniconda3/envs/neural/bin/python  -u /home/mohameys/thesis/main_ppo.py --env_name=$env_name --env_alias=$env_alias --seed=42 --wandb_project=$wandb_project --no_frame_stack --evaluate --wandb \
#     --recurrent --ctrl_reward --ngu_reward &> $env_alias.out
# ##################################################################################################################################