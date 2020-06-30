# Script to run cheetah. Makes the following HP changes relative to defaults for
# walker/ball/reacher/cheetah:
#
# - Changed action_repeat from 8 down to 4. This is a **BREAKING CHANGE**.
# - Changed critic_lr and actor_lr from 1e-3 to 2e-4.
# - Removed cartpole/walker/etc., since they seem to have capped out or need
#   different HPs.
#
# It would make sense to use the "translate" augmentation, but I don't have
# access to that.
WORK_DIR=./rad-out/
# cheetah needs some extra time to cap out performance
N_STEPS=2000000
TRAIN_ARGS="--work_dir $WORK_DIR --num_eval_episodes 10 "\
"--critic_lr 2e-4 --actor_lr 2e-4 --eval_freq 10000 "\
"--batch_size 128 --num_train_steps $N_STEPS "\
"--save_tb --save_model "
SEED=99

mkdir "$WORK_DIR"

CUDA_VISIBLE_DEVICES=1 python train.py \
    --domain_name cheetah \
    --task_name run \
    --seed "$SEED" \
    $(cat test_args_cheetah.sh)
    $TRAIN_ARGS
