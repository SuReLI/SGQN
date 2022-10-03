import torch
import os

import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
import os
import matplotlib.pyplot as plt
import numpy as np

torch.backends.cudnn.benchmark = True


def main(args):
    home = os.environ["HOME"]
    os.environ["MUJOCO_MJKEY_PATH"] = f"{home}/.mujoco/mujoco210_linux/bin/mjkey.txt"
    # os.environ["MUJOCO_GL"] = "egl"

    # Set seed
    utils.set_seed_everywhere(args.seed)
    if args.cameras == 0:
        cameras = ["third_person"]
    elif args.cameras == 1:
        cameras = ["first_person"]
    elif args.cameras == 2:
        cameras = ["third_person", "first_person"]
    else:
        raise Exception("Current Camera Pose Not Supported.")

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        n_substeps=args.n_substeps,
        frame_stack=args.frame_stack,
        image_size=args.image_size,
        mode="train",
        cameras=cameras,  # ['third_person', 'first_person']
        observation_type=args.observation_type,
        action_space=args.action_space,
        test=4,
    )

    env.reset()
    for i in range(100):
        action = env.action_space.sample()

        next_obs, next_state, reward, done, info = env.step(action)
    obs = env.render("rgb_array")
    # act until done
    # i = 0
    # while not done:
    #     action = env.action_space.sample()
    #     observation, reward, done, info = env.step(action)
    #
    #     print(obs_to_input(observation).shape)
    #     # print(reward)
    #     # cv2.imwrite(f"result/d{i}.png", obs)
    #     i += 1
    # next_obs = random_color_jitter(np.array(next_obs))
    print(np.array(next_obs).shape)
    plt.imshow(np.array(next_obs).transpose(1, 2, 0))
    plt.savefig(f"test2.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
