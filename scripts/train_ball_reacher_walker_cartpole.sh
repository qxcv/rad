# trains ball/walker/cartpole/reacher
WORK_DIR=./rad-out/
N_STEPS=1000000
TRAIN_ARGS="--work_dir $WORK_DIR --num_eval_episodes 10 "\
"--critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 "\
"--batch_size 128 --num_train_steps $N_STEPS "\
"--save_tb --save_model "
SEED=23
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name ball_in_cup \
    --task_name catch \
    --seed "$SEED" \
    $(cat test_args_ball_reacher_walker_cartpole.sh) \
    $TRAIN_ARGS &
CUDA_VISIBLE_DEVICES=1 python train.py \
    --domain_name walker \
    --task_name walk \
    --seed "$SEED" \
    $(cat test_args_ball_reacher_walker_cartpole.sh) \
    $TRAIN_ARGS &
CUDA_VISIBLE_DEVICES=2 python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --seed "$SEED" \
    $(cat test_args_ball_reacher_walker_cartpole.sh) \
    $TRAIN_ARGS &
CUDA_VISIBLE_DEVICES=3 python train.py \
    --domain_name reacher \
    --task_name easy \
    --seed "$SEED" \
    $(cat test_args_ball_reacher_walker_cartpole.sh) \
    $TRAIN_ARGS &
wait
