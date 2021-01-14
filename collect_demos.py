#!/usr/bin/env python3
import argparse
import collections
import gzip
import os

import cloudpickle
import dmc2gym
from dm_control import viewer
from imitation.data.types import TrajectoryWithRew
import numpy as np
import torch

from curl_sac import RadSacAgent
from train import add_common_args


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("actor_path", help="path to actor model")
    parser.add_argument(
        "--viewer",
        default=False,
        action='store_true',
        help='launch a viewer to interact with the policy')
    parser.add_argument(
        "--save-path", default=None, help="path to save demonstrations to")
    parser.add_argument(
        "--ntraj", default=10, type=int, help="number of trajectories to save")
    parser.add_argument(
        "--threads", default=None, type=int, help="number of threads for Torch")
    return parser.parse_args()


def save_compressed_pickle(obj, out_path):
    # TODO: move this utility function into `imitation`, along with a symmetric
    # function that loads compressed pickles.
    dir_path = os.path.dirname(out_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with gzip.GzipFile(out_path, 'w') as fp:
        cloudpickle.dump(obj, fp)


def unwrap(env):
    attrs = ['env', '_env']
    while any(hasattr(env, attr) for attr in attrs):
        for attr in attrs:
            new_env = getattr(env, attr, None)
            env = new_env if new_env is not None else env
    return env


def sample_traj_stacked(gym_env, agent, frame_stack):
    obs = gym_env.reset()

    frames = collections.deque(maxlen=frame_stack or 1)
    while len(frames) < frames.maxlen:
        frames.append(obs)

    all_obs = [obs]
    all_acts = []
    all_infos = []
    all_rews = []

    done = False
    while not done:
        stacked_frames = np.concatenate(frames, axis=0)
        act = agent.sample_action(stacked_frames / 255.)
        obs, rew, done, info = gym_env.step(act)

        # update frame stack
        frames.append(obs)

        # record rest of trajectory
        all_obs.append(obs)
        all_acts.append(act)
        all_infos.append(info)
        all_rews.append(rew)

    # obs/acts/rews are ndarrays; infos can be a list
    all_obs = np.stack(all_obs, axis=0)
    all_acts = np.stack(all_acts, axis=0)
    all_rews = np.stack(all_rews, axis=0)

    return TrajectoryWithRew(
        obs=all_obs, acts=all_acts, infos=all_infos, rews=all_rews)


def main():
    args = parse_args()

    if not (bool(args.viewer) ^ bool(args.save_path)):
        raise Exception("you need to provide --viewer xor --save-dir "
                        "arguments for this to do anything useful :)")

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    # TODO: The next few calls are copy-pasted out of train.py. Consider
    # refactoring so that you don't have to copy-paste (otoh not very important
    # since this code only needs to be run once)
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    pre_transform_image_size = args.pre_transform_image_size if 'crop' \
        in args.data_augs else args.image_size
    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=pre_transform_image_size,
        width=pre_transform_image_size,
        frame_skip=args.action_repeat)
    env.seed(args.seed)
    action_shape = env.action_space.shape
    obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
    agent = RadSacAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=dev,
        hidden_dim=args.hidden_dim,
        encoder_type=args.encoder_type,
        encoder_feature_dim=args.encoder_feature_dim,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        latent_dim=args.latent_dim,
        data_augs=args.data_augs, )
    agent.load_ac(actor_path=args.actor_path)

    if args.viewer:
        dmc_env = unwrap(env)
        frames = collections.deque(maxlen=args.frame_stack or 1)

        def loaded_policy(time_step):
            # time_step just contains joint angles; we want image observation
            obs = env.env._get_obs(time_step)
            frames.append(obs)
            while len(frames) < frames.maxlen:
                # for init
                frames.append(obs)
            stacked_obs = np.concatenate(frames, axis=0) / 255.
            return agent.sample_action(stacked_obs)

        viewer.launch(dmc_env, policy=loaded_policy)
        return  # done

    # otherwise, we need to save a bunch of imitation.data.TrajectoryWithRew
    # instance to some directory somewhere…
    all_traj = []
    for t in range(args.ntraj):
        traj = sample_traj_stacked(env, agent,
                                   frame_stack=args.frame_stack or 1)
        all_traj.append(traj)
    # for now I'm just going to save all trajectories in one file
    print(f"Saving to '{args.save_path}'")
    save_compressed_pickle(all_traj, args.save_path)

    env.close()


if __name__ == '__main__':
    main()
