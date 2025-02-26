import isaacgymenvs

env = isaacgymenvs.make(seed=0,
                        task="Cartpole",
                        num_envs=2000,
                        sim_device="cuda:0",
                        rl_device="cuda:0",
                        graphics_device_id=0,
                        headless=False)