#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/data/RL/RL_AUV_tracking/RL_AUV_tracking/

policy="PPO"

if [ "$policy" == "SAC" ]; then
    echo "Running training script"
    # --choice 0:train 1:keep training 2:eval (1、2 need resume-path of policy)
    python SB3_trainer.py "$@" \
    --device cuda \
    --choice 2 \
    --env v2-Teacher-render \
    --policy SAC \
    --render 1 \
    --nb_envs 1 \
    --max_episode_step 200 \
    \
    --seed 41 \
    --buffer-size 100000 \
    --lr 1e-4 \
    --alpha-lr 3e-4 \
    --noise_std 0.12 \
    --gamma 0.99 \
    --tau 0.005 \
    --auto_alpha 1 \
    --alpha 0.2 \
    \
    --start-timesteps 1000 \
    --timesteps 1000000 \
    --step-per-collect 5 \
    --update-per-step 0.2 \
    --n-step 2 \
    --batch-size 256 \
    --test_episode 10 \
    --log-dir ../../log/teacher \
    --resume-path /data/RL/RL_AUV_tracking/RL_AUV_tracking/log/teacher/SAC/04-23_12/rl_model_1500000_steps.zip \

elif [ "$policy" == "PPO" ]; then
    echo "Running testing script"
    # --choice 0:train 1:keep training 2:eval (1、2 need resume-path of policy)
    python SB3_trainer.py "$@" \
        --device cuda \
        --choice 2 \
        --env v2-Teacher-render \
        --policy PPO \
        --render 1 \
        --nb_envs 1 \
        --max_episode_step 200 \
        \
        --seed 46 \
        --buffer-size 100000 \
        --lr 3e-4 \
        --gamma 0.99 \
        \
        --n-steps 1024 \
        --vf-coef 0.25 \
        --ent-coef 0.0 \
        --gae-lambda 0.95 \
        --max-grad-norm 0.5 \
        --eps-clip 0.2 \
        --value-clip 0.1 \
        --norm-adv 0 \
        \
        --timesteps 1000000 \
        --batch-size 256 \
        --log-dir ../../log/teacher \
        --resume-path /home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/teacher/PPO/07-08_16/rl_model_670000_steps.zip \

else
    echo "Unknown policy"
fi
