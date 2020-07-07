#!/usr/bin/env bash

set -e

NTRAJ=50
DATE="$(date '+%Y-%m-%d')"

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
    --domain_name cartpole --task_name swingup --ntraj $NTRAJ \
    --save-path "./rad-out/trajectories/cartpole-swingup-${DATE}.pkl.gz" &

echo -e "\n\nWorking on ball_in_cup/catch"
python collect_demos.py \
    $(cat ./scripts/test_args_ball_reacher_walker_cartpole.sh) \
    "$(get_latest_policy ./rad-out/ball_in_cup-catch-*-pixel/model/)" \
    --domain_name ball_in_cup --task_name catch --ntraj $NTRAJ \
    --save-path "./rad-out/trajectories/ball-in-cup-catch-${DATE}.pkl.gz" &

echo -e "\n\nWorking on reacher/easy"
python collect_demos.py \
    $(cat ./scripts/test_args_ball_reacher_walker_cartpole.sh) \
    "$(get_latest_policy ./rad-out/reacher-easy-*-pixel/model/)" \
    --domain_name reacher --task_name easy --ntraj $NTRAJ \
    --save-path "./rad-out/trajectories/reacher-easy-${DATE}.pkl.gz" &

echo -e "\n\nWorking on walker/walk"
python collect_demos.py \
    $(cat ./scripts/test_args_ball_reacher_walker_cartpole.sh) \
    "$(get_latest_policy ./rad-out/walker-walk-*-pixel/model/)" \
    --domain_name walker --task_name walk --ntraj $NTRAJ \
    --save-path "./rad-out/trajectories/walker-walk-${DATE}.pkl.gz" &

echo -e "\n\nWorking on finger/spin"
python collect_demos.py \
    $(cat ./scripts/test_args_finger.sh) \
    "$(get_latest_policy ./rad-out/finger-spin-*-pixel/model/)" \
    --domain_name finger --task_name spin --ntraj $NTRAJ \
    --save-path "./rad-out/trajectories/finger-spin-${DATE}.pkl.gz" &

echo -e "\n\nWorking on cheetah/run"
python collect_demos.py \
    $(cat ./scripts/test_args_cheetah.sh) \
    "$(get_latest_policy ./rad-out/cheetah-run-*-pixel/model/)" \
    --domain_name cheetah --task_name run --ntraj $NTRAJ \
    --save-path "./rad-out/trajectories/cheetah-run-${DATE}.pkl.gz" &

wait

echo "Done, check rad-out/trajectories/"
