#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/data/RL/RL_AUV_tracking/RL_AUV_tracking/

# --- Configuration ---
# Change these variables to easily switch between experiments

# 1. Choose the policy: "PPO" or "SAC"
ALG_CONFIG="configs/algorithm/ppo.yml"

# 2. Choose the environment config
ENV_CONFIG="configs/envs/v0_config.yml"

# 3. Choose the action: "0" (train), "1" (keep train), "2" (eval)
CHOICE="2"

# 4. Param
EVAL="1"


echo "================================================="
echo "Running experiment with the following settings:"
echo "Env Config:   $ENV_CONFIG"
echo "Alg Config:   $ALG_CONFIG"
echo "Action:       $CHOICE"
echo "================================================="

# Build the command
CMD="python SB3_trainer.py --env_config $ENV_CONFIG --alg_config $ALG_CONFIG --choice $CHOICE"

# Execute the command
eval $CMD