from wandb_wrapper import WandbWrapper
import numpy as np
from environment_manager import EnvironmentManager
from ppo_agent import PPOAgent
from ppo_models import ActorCriticNet

# Initialize WandbWrapper and EnvironmentManager
wdb = WandbWrapper("../config/ppo.yaml")
env = EnvironmentManager("CartPole-v1", "rgb_array")

# Initialize network
action_space, observation_space = env.get_dimensions()
model = ActorCriticNet(action_space, observation_space)

# Initialize PPO agent
agent = PPOAgent(env, wdb, model)

# Start training
agent.train()

env.close()
wdb.finish()
