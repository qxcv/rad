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
WORK_DIR=./rad-out/finger-cheetah/
N_STEPS=1000000
ARGS="--encoder_type pixel --work_dir $WORK_DIR "\
"--action_repeat 4 --num_eval_episodes 10 "\
"--pre_transform_image_size 100 --image_size 84 "\
"--agent rad_sac --frame_stack 3 --data_augs crop "\
"--critic_lr 2e-4 --actor_lr 2e-4 --eval_freq 10000 "\
"--batch_size 128 --num_train_steps $N_STEPS "\
"--save_tb --save_model "

mkdir "$WORK_DIR"

for seed in 99 23 42; do
    CUDA_VISIBLE_DEVICES=1 python train.py \
        --domain_name cheetah \
        --task_name run \
        --seed "$seed" \
        $ARGS &
    # CUDA_VISIBLE_DEVICES=0 python train.py \
    #     --domain_name finger \
    #     --task_name spin \
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
