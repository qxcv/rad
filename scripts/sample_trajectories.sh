#!/usr/bin/env python3

set -e

get_latest_policy() {
    if [ -z "$1" ]; then
        snap_dir="$(readlink -f .)"
    else
        snap_dir="$(readlink -f "$1")"
    fi
    last_pol="$(find "$1" -name 'actor_*.pt' -type f -printf "%f\n" | sort -V | tail -n 1)"
    echo "$(readlink -f "$1/$last_pol")"
}

# this script takes trained policies produced by train_*.sh and uses them to
# generate demonstrations

echo -e "\n\nWorking on cartpole/swingup"
python collect_demos.py \
    $(cat ./scripts/test_args_ball_reacher_walker_cartpole.sh) \
    "$(get_latest_policy ./rad-out/cartpole-swingup-*-pixel/model/)" \
    --domain_name cartpole --task_name swingup \
    --save-path ./rad-out/trajectories/cartpole-swingup.pkl.gz

echo -e "\n\nWorking on ball_in_cup/catch"
python collect_demos.py \
    $(cat ./scripts/test_args_ball_reacher_walker_cartpole.sh) \
    "$(get_latest_policy ./rad-out/ball_in_cup-catch-*-pixel/model/)" \
    --domain_name ball_in_cup --task_name catch \
    --save-path ./rad-out/trajectories/ball-in-cup-catch.pkl.gz

echo -e "\n\nWorking on reacher/easy"
python collect_demos.py \
    $(cat ./scripts/test_args_ball_reacher_walker_cartpole.sh) \
    "$(get_latest_policy ./rad-out/reacher-easy-*-pixel/model/)" \
    --domain_name reacher --task_name easy \
    --save-path ./rad-out/trajectories/reacher-easy.pkl.gz

echo -e "\n\nWorking on walker/walk"
python collect_demos.py \
    $(cat ./scripts/test_args_ball_reacher_walker_cartpole.sh) \
    "$(get_latest_policy ./rad-out/walker-walk-*-pixel/model/)" \
    --domain_name walker --task_name walk \
    --save-path ./rad-out/trajectories/walker-walk.pkl.gz

echo -e "\n\nWorking on finger/spin"
python collect_demos.py \
    $(cat ./scripts/test_args_finger.sh) \
    "$(get_latest_policy ./rad-out/finger-spin-*-pixel/model/)" \
    --domain_name finger --task_name spin \
    --save-path ./rad-out/trajectories/finger-spin.pkl.gz

echo -e "\n\nWorking on cheetah/run"
python collect_demos.py \
    $(cat ./scripts/test_args_cheetah.sh) \
    "$(get_latest_policy ./rad-out/cheetah-run-*-pixel/model/)" \
    --domain_name cheetah --task_name run \
    --save-path ./rad-out/trajectories/cheetah-run.pkl.gz

echo "Done, check rad-out/trajectories/"
