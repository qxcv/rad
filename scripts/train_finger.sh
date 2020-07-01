# Script to run finger. Makes the following HP changes relative
# to defaults for walker/ball/reacher:
#
# - Changed action_repeat from 8 down to 2. This is a **BREAKING CHANGE**. It's
#   also a really small value; I suspect "finger spin" needs it to actually
#   work.
# - Removed all domains other than finger spin.
WORK_DIR=./rad-out/
N_STEPS=1000000
TRAIN_ARGS="--work_dir $WORK_DIR --num_eval_episodes 10 "\
"--critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 "\
"--batch_size 128 --num_train_steps $N_STEPS "\
"--save_tb --save_model "
SEED=99
mkdir -p "$WORK_DIR"
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name finger \
    --task_name spin \
    --seed "$SEED" \
    $(cat scripts/test_args_finger.sh) \
    $TRAIN_ARGS
