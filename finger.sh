# Script to run finger. Makes the following HP changes relative
# to defaults for walker/ball/reacher:
#
# - Changed action_repeat from 8 down to 2. This is a **BREAKING CHANGE**. It's
#   also a really small value; I suspect "finger spin" needs it to actually
#   work.
# - Removed all domains other than finger spin.
WORK_DIR=./rad-out/only-finger/
N_STEPS=1000000
ARGS="--encoder_type pixel --work_dir $WORK_DIR "\
"--action_repeat 2 --num_eval_episodes 10 "\
"--pre_transform_image_size 100 --image_size 84 "\
"--agent rad_sac --frame_stack 3 --data_augs crop "\
"--critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 "\
"--batch_size 128 --num_train_steps $N_STEPS "\
"--save_tb --save_model "

mkdir "$WORK_DIR"

for seed in 99 23 42; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --domain_name finger \
        --task_name spin \
        --seed "$seed" \
        $ARGS &
    # CUDA_VISIBLE_DEVICES=1 python train.py \
    #     --domain_name cheetah \
    #     --task_name run \
    #     --seed "$seed" \
    #     $ARGS &
    # CUDA_VISIBLE_DEVICES=0 python train.py \
    #     --domain_name cartpole \
    #     --task_name swingup \
    #     --seed "$seed" \
    #     $ARGS &
    # CUDA_VISIBLE_DEVICES=0 python train.py \
    #     --domain_name walker \
    #     --task_name walk \
    #     --seed "$seed" \
    #     $ARGS
    # CUDA_VISIBLE_DEVICES=0 python train.py \
    #     --domain_name ball_in_cup \
    #     --task_name catch \
    #     --seed "$seed" \
    #     $ARGS &
    # CUDA_VISIBLE_DEVICES=3 python train.py \
    #     --domain_name reacher \
    #     --task_name easy \
    #     --seed "$seed" \
    #     $ARGS &
    wait
done
