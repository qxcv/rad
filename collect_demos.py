#!/usr/bin/env python3
import argparse
import gzip
import os
import time

import cloudpickle
import dmc2gym
from imitation.data.rollout import TrajectoryAccumulator
import torch

from curl_sac import RadSacAgent
from train import add_common_args
import utils

FPS = 25
SPF = 1. / FPS


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("actor_path", help="path to actor model")
    parser.add_argument(
        "--show-video",
        default=False,
        action='store_true',
        help='render video while rolling out trajectories')
    parser.add_argument(
        "--save-dir", default=None, help="directory to save demonstrations to")
    return parser.parse_args()


def save_compressed_pickle(obj, out_path):
    # TODO: move this utility function into `imitation`, along with a symmetric
    # function that loads compressed pickles
    dir_path = os.path.dirname(out_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with gzip.GzipFile(out_path, 'w') as fp:
        cloudpickle.dump(obj, fp)


def main():
    args = parse_args()

    if not (args.show_video or args.save_dir):
        raise Exception("you need to provide --show-video and/or --save-dir "
                        "arguments for this to do anything useful :)")

    # TODO: The next few calls are copy-pasted out of train.py. Consider
    # refactoring so that you don't have to copy-paste.
    dev = torch.device('cuda')
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

    if args.save_dir:
        pass

    done = False
    while not done:
        obs = env.reset()
        if args.show_video:
            env.render(mode="human")
            time.sleep(SPF / args.action_repeat)
        with utils.eval_mode(agent):
            action = agent.sample_action(obs / 255.)
        obs, reward, done, infos = env.step(action)
        if args.show_video:
            env.render(mode="human")
            time.sleep(SPF / args.action_repeat)

    env.close()


if __name__ == '__main__':
    main()
