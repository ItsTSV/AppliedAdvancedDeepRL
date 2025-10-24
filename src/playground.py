"""
This serves as a playground for testing trained models, environments and algorithms.
So far, the code here is only a temporary placeholder.
Later, it will probably become a terminal application ;)
"""
from wandb_wrapper import WandbWrapper
from environment_manager import EnvironmentManager
from ppo_agent_continuous import PPOAgentContinuous
from ppo_models import ContinuousActorCriticNet

# Initialize WandbWrapper
wdb = WandbWrapper("../config/ppo_continuous_testing.yaml")

# Initialize environment
name = wdb.get_hyperparameter("environment")
env = EnvironmentManager(name, "human")
env.build_continuous()

# Initialize network
action_space, observation_space = env.get_dimensions()
model = ContinuousActorCriticNet(action_space, observation_space)

# Initialize PPO agent
agent = PPOAgentContinuous(env, wdb, model)

# Load model
agent.load_model("../models/ppo_swimmer.pth")

# Play
agent.play()

# Finish stuff
env.close()
wdb.finish()
