#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/data/RL/RL_AUV_tracking/RL_AUV_tracking/

policy="SAC"

if [ "$policy" == "SAC" ]; then
    echo "Running training script"
    # --choice 0:train 1:keep training 2:eval (1、2 need resume-path of policy)
    python teacher_trainer.py "$@" \
    --choice 2 \
    --env AUVTracking_rgb \
    --policy SAC \
    --render 1 \
    --nb_envs 3 \
    --max_episode_step 200 \
    \
    --seed 42 \
    --buffer-size 50000 \
    --actor-lr 3e-4 \
    --critic-lr 3e-4 \
    --alpha-lr 3e-4 \
    --noise_std 1.2 \
    --gamma 0.99 \
    --tau 0.005 \
    --auto_alpha 1 \
    --alpha 0.2 \
    \
    --start-timesteps 1000 \
    --epoch 100 \
    --step-per-epoch 12000 \
    --step-per-collect 5 \
    --update-per-step 0.2 \
    --n-step 2 \
    --batch-size 128 \
    --test_episode 10 \
    --logdir ../../log \
    --resume-path /home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/teacher/sac/12-13_10/policy_54.pth \

elif [ "$policy" == "PPO" ]; then
    echo "Running testing script"
    # --choice 0:train 1:keep training 2:eval (1、2 need resume-path of policy)
    python teacher_trainer.py "$@" \
        --choice 0 \
        --env AUVTracking_rgb \
        --policy PPO \
        --render 0 \
        --nb_envs 3 \
        --max_episode_step 200 \
        \
        --seed 42 \
        --buffer-size 50000 \
        --lr 3e-4 \
        --gamma 0.99 \
        \
        --rew-norm 1 \
        --vf-coef 0.25 \
        --ent-coef 0.0 \
        --gae-lambda 0.95 \
        --bound-action-method clip \
        --lr-decay 1 \
        --max-grad-norm 0.5 \
        --eps-clip 0.2 \
        --value-clip 0 \
        --norm-adv 0 \
        --recompute-adv 1 \
        \
        --start-timesteps 5000 \
        --epoch 100 \
        --step-per-epoch 12000 \
        --step-per-collect 5 \
        --update-per-step 0.2 \
        --n-step 2 \
        --batch-size 128 \
        --test_episode 10 \
        --logdir ../../log
    #    --resume-path \
else
    echo "Unknown policy"
fi