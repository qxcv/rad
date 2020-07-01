# trains ball/walker/cartpole/reacher
WORK_DIR=./rad-out/
N_STEPS=1500000
TRAIN_ARGS="--work_dir $WORK_DIR --num_eval_episodes 10 "\
"--critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 "\
"--batch_size 128 --num_train_steps $N_STEPS "\
"--save_tb --save_model "
SEED=23
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name walker \
    --task_name walk \
    --seed "$SEED" \
    $(cat scripts/test_args_ball_reacher_walker_cartpole.sh) \
    $TRAIN_ARGS
