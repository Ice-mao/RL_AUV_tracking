#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/data/RL/RL_AUV_tracking/RL_AUV_tracking/

policy="PPO"

if [ "$policy" == "SAC" ]; then
    echo "Running training script"
    # --choice 0:train 1:keep training 2:eval (1、2 need resume-path of policy)
    python SB3_trainer.py "$@" \
    --device cpu \
    --choice 0 \
    --env AUVTracking_rgb \
    --policy SAC \
    --render 0 \
    --nb_envs 5 \
    --max_episode_step 200 \
    \
    --seed 42 \
    --buffer-size 50000 \
    --lr 3e-4 \
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
    --batch-size 128 \
    --test_episode 10 \
    --log-dir ../../log
#    --resume-path-model \

elif [ "$policy" == "PPO" ]; then
    echo "Running testing script"
    # --choice 0:train 1:keep training 2:eval (1、2 need resume-path of policy)
    python SB3_trainer.py "$@" \
        --device cpu \
        --choice 0 \
        --env AUVTracking_rgb \
        --policy PPO \
        --render 0 \
        --nb_envs 5 \
        --max_episode_step 200 \
        \
        --seed 42 \
        --buffer-size 100000 \
        --lr 3e-4 \
        --gamma 0.99 \
        \
        --n-steps 512 \
        --vf-coef 0.25 \
        --ent-coef 0.0 \
        --gae-lambda 0.95 \
        --max-grad-norm 0.5 \
        --eps-clip 0.2 \
        --value-clip 0.1 \
        --norm-adv 0 \
        \
        --timesteps 1000000 \
        --batch-size 128 \
        --log-dir ../../log
    #    --resume-path \
else
    echo "Unknown policy"
fi