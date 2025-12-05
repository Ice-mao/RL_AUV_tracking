#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/data/RL/RL_AUV_tracking/RL_AUV_tracking/

# --- Configuration ---
# Change these variables to easily switch between experiments

# 1. Choose the policy: "PPO" or "SAC"
ALG_CONFIG="${ALG_CONFIG:-sac}"
# Supported: sac, sac_v1, sac_3d_v0, sac_3d_v1, ppo, ppo_v1, ppo_3d_v0

# 2. Choose the environment config
ENV_CONFIG="${ENV_CONFIG:-v0_config}"
# Supported: v0_config, v1_config, 3d_v0_config, 3d_v1_config, 3d_v0_config_test

# 3. Choose the action: "0" (train), "1" (keep train), "2" (eval)
CHOICE="${CHOICE:-0}"

# 4. Param
EVAL="${EVAL:-0}"
SHOW_VIEWPORT="${SHOW_VIEWPORT:-0}"

# Auto-complete paths
ALG_CONFIG="configs/algorithm/${ALG_CONFIG%.yml}.yml"
ENV_CONFIG="configs/envs/${ENV_CONFIG%.yml}.yml"

echo "================================================="
echo "Running experiment with the following settings:"
echo "Env Config:   $ENV_CONFIG"
echo "Alg Config:   $ALG_CONFIG"
echo "Action:       $CHOICE"
echo "================================================="

# Build the command
CMD="python SB3_trainer.py --env_config $ENV_CONFIG --alg_config $ALG_CONFIG
    --choice $CHOICE"

if [ "$EVAL" = '1' ]; then
    CMD="$CMD --eval"
fi

if [ "$SHOW_VIEWPORT" = '1' ]; then
    CMD="$CMD --show_viewport"
fi

# Execute the command
eval $CMD