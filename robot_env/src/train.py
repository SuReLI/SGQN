import torch
import os
import numpy as np
import gym
from algorithms.rl_utils import make_obs_grad_grid
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder


def evaluate(
    env, agent, algorithm, video, num_episodes, L, step, test_env=False, eval_mode=None
):
    episode_rewards = []
    for i in range(num_episodes):
        obs, _ = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        episode_step = 0
        torch_obs = []
        torch_action = []
        while not done:
            with torch.no_grad():
                with utils.eval_mode(agent):
                    action = agent.select_action(obs)

                obs, _, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward
                # log in tensorboard 15th step
                if algorithm == "sgsac":
                    if i == 0 and episode_step in [15, 20, 25, 30, 35, 40] and step > 0:
                        _obs = agent._obs_to_input(obs)
                        torch_obs.append(_obs)
                        torch_action.append(
                            torch.tensor(action).to(_obs.device).unsqueeze(0)
                        )
                        prefix = "eval" if eval_mode is None else f"test_{eval_mode}"
                    if i == 0 and episode_step == 40 and step > 0:
                        agent.log_tensorboard(
                            torch.cat(torch_obs, 0),
                            torch.cat(torch_action, 0),
                            step,
                            prefix=prefix,
                        )
                    # attrib_grid = make_obs_grad_grid(torch.sigmoid(mask))
                    # agent.writer.add_image(
                    #     prefix + "/smooth_attrib", attrib_grid, global_step=step
                    # )

                episode_step += 1

        if L is not None:
            _test_env = f"_test_env_{eval_mode}" if test_env else ""
            video.save(f"{step}{_test_env}.mp4")
            L.log(f"eval/episode_reward{_test_env}", episode_reward, step)
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


def main(args):
    home = os.environ["HOME"]
    os.environ["MJKEY_PATH"] = f"{home}/.mujoco/mujoco210_linux/bin/mjkey.txt"
    os.environ["MUJOCO_GL"] = "egl"
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
    )

    test_envs = []
    test_envs_mode = []
    for eval_mode in range(1, 6):
        test_env = make_env(
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
            test=eval_mode,
        )
        test_envs.append(test_env)
        test_envs_mode.append(eval_mode)

    # Create working directory
    work_dir = os.path.join(
        args.log_dir,
        args.domain_name + "_" + args.task_name,
        args.algorithm,
        str(args.seed),
    )
    print("Working directory:", work_dir)
    assert not os.path.exists(
        os.path.join(work_dir, "train.log")
    ), "specified working directory already exists"
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    utils.write_info(args, os.path.join(work_dir, "info.log"))

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size,
    )
    cropped_obs_shape = (
        3 * args.frame_stack,
        args.image_crop_size,
        args.image_crop_size,
    )
    print("Observations:", env.observation_space.shape)
    print("Cropped observations:", cropped_obs_shape)
    agent = make_agent(
        obs_shape=cropped_obs_shape, action_shape=env.action_space.shape, args=args
    )

    start_step, episode, episode_reward, done = 0, 0, 0, True
    L = Logger(work_dir)
    start_time = time.time()
    for step in range(start_step, args.train_steps + 1):
        if done:
            if step > start_step:
                L.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # Evaluate agent periodically
            if step % args.eval_freq == 0:
                print("Evaluating:", work_dir)
                L.log("eval/episode", episode, step)
                evaluate(env, agent, args.algorithm, video, args.eval_episodes, L, step)
                if test_envs is not None:
                    for test_env, test_env_mode in zip(test_envs, test_envs_mode):
                        evaluate(
                            test_env,
                            agent,
                            args.algorithm,
                            video,
                            args.eval_episodes,
                            L,
                            step,
                            test_env=True,
                            eval_mode=test_env_mode,
                        )
                L.dump(step)

            # Save agent periodically
            if step > start_step and step % args.save_freq == 0:
                torch.save(
                    agent.actor.state_dict(),
                    os.path.join(model_dir, f"actor_{step}.pt"),
                )
                torch.save(
                    agent.critic.state_dict(),
                    os.path.join(model_dir, f"critic_{step}.pt"),
                )
                if args.algorithm == "sgsac":
                    torch.save(
                        agent.attribution_predictor.state_dict(),
                        os.path.join(model_dir, f"attrib_predictor_{step}.pt"),
                    )

            L.log("train/episode_reward", episode_reward, step)

            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log("train/episode", episode, step)

        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # Take step
        next_obs, _, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs

        episode_step += 1

    print("Completed training for", work_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
