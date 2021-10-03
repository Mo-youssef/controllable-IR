#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export GRIDDLY_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

wandb agent youssef101/Thesis_PPO_clusters2/$2 |& tee sweep_agent_$1_$2.out