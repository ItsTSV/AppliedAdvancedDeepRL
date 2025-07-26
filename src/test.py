from wandb_wrapper import WandbWrapper
import numpy as np
from environment_manager import EnvironmentManager


wdb = WandbWrapper("../config/test.yaml")

env = EnvironmentManager("HalfCheetah-v5", "human")
action_space, observation_space = env.get_dimensions()
print(f"Action space size: {action_space}, observation space size: {observation_space}")

for i in range(0, 5):
    finished = False
    _, _, state = env.reset()
    while not finished:
        state, reward, finished, info, = env.step(np.random.random(6,))
        env.render()
    total_steps, total_reward, _ = env.reset()
    wdb.log({
        "episode_steps": total_steps,
        "episode_reward": total_reward
    })

env.close()
wdb.finish()
