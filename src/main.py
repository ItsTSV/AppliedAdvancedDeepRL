from wandb_wrapper import WandbWrapper
import numpy as np
from environment_manager import EnvironmentManager
from ppo_agent_discrete import PPOAgentDiscrete
from ppo_models import ActorCriticNet

# Initialize WandbWrapper
wdb = WandbWrapper("../config/ppo.yaml")

# Initialize environment
name = wdb.get_hyperparameter("environment")
env = EnvironmentManager(name, "rgb_array")

# Initialize network
action_space, observation_space = env.get_dimensions()
model = ActorCriticNet(action_space, observation_space)

# Initialize PPO agent
agent = PPOAgentDiscrete(env, wdb, model)

# Start training
agent.train()

env.close()
wdb.finish()
