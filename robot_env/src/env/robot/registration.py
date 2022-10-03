from gym.envs.registration import register

REGISTERED_ROBOT_ENVS = False


def register_robot_envs(
    n_substeps=20,
    observation_type="image",
    reward_type="dense",
    image_size=84,
    use_xyz=False,
):
    global REGISTERED_ROBOT_ENVS
    if REGISTERED_ROBOT_ENVS:
        return

    register(
        id="RobotPegbox-v0",
        entry_point="env.robot.peg_in_box:PegBoxEnv",
        kwargs=dict(
            xml_path="robot/peg_in_box.xml",
            n_substeps=n_substeps,
            observation_type=observation_type,
            reward_type=reward_type,
            image_size=image_size,
            use_xyz=use_xyz,
        ),
    )

    for i in range(1, 6):
        register(
            id=f"RobotPegbox_test_{i}-v0",
            entry_point="env.robot.peg_in_box:PegBoxEnv",
            kwargs=dict(
                xml_path=f"robot/peg_in_box_test{i}.xml",
                n_substeps=n_substeps,
                observation_type=observation_type,
                reward_type=reward_type,
                image_size=image_size,
                use_xyz=use_xyz,
            ),
        )

    register(
        id="RobotHammerall-v0",
        entry_point="env.robot.hammer_all:HammerAllEnv",
        kwargs=dict(
            xml_path="robot/hammer_all.xml",
            n_substeps=n_substeps,
            observation_type=observation_type,
            reward_type=reward_type,
            image_size=image_size,
            use_xyz=use_xyz,
        ),
    )

    for i in range(1, 6):
        register(
            id=f"RobotHammerall_test_{i}-v0",
            entry_point="env.robot.hammer_all:HammerAllEnv",
            kwargs=dict(
                xml_path=f"robot/hammer_all_test{i}.xml",
                n_substeps=n_substeps,
                observation_type=observation_type,
                reward_type=reward_type,
                image_size=image_size,
                use_xyz=use_xyz,
            ),
        )

    register(
        id="RobotReach-v0",
        entry_point="env.robot.reach:ReachEnv",
        kwargs=dict(
            xml_path="robot/reach.xml",
            n_substeps=n_substeps,
            observation_type=observation_type,
            reward_type=reward_type,
            image_size=image_size,
            use_xyz=use_xyz,
        ),
    )

    for i in range(1, 6):
        register(
            id=f"RobotReach_test_{i}-v0",
            entry_point="env.robot.reach:ReachEnv",
            kwargs=dict(
                xml_path=f"robot/reach_test{i}.xml",
                n_substeps=n_substeps,
                observation_type=observation_type,
                reward_type=reward_type,
                image_size=image_size,
                use_xyz=use_xyz,
            ),
        )

    register(
        id="RobotPush-v0",
        entry_point="env.robot.push:PushEnv",
        kwargs=dict(
            xml_path="robot/push.xml",
            n_substeps=n_substeps,
            observation_type=observation_type,
            reward_type=reward_type,
            image_size=image_size,
            use_xyz=use_xyz,
        ),
    )

    for i in range(1, 6):
        register(
            id=f"RobotPush_test_{i}-v0",
            entry_point="env.robot.push:PushEnv",
            kwargs=dict(
                xml_path=f"robot/push_test{i}.xml",
                n_substeps=n_substeps,
                observation_type=observation_type,
                reward_type=reward_type,
                image_size=image_size,
                use_xyz=use_xyz,
            ),
        )

    REGISTERED_ROBOT_ENVS = True
