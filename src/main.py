from wandb_wrapper import WandbWrapper
import numpy as np
from environment_manager import EnvironmentManager
from ppo_agent_discrete import PPOAgentDiscrete
from ppo_agent_continuous import PPOAgentContinuous
from ppo_models import DiscreteActorCriticNet, ContinuousActorCriticNet

# Initialize WandbWrapper
#wdb = WandbWrapper("../config/ppo_discrete.yaml")
wdb = WandbWrapper("../config/ppo_continuous.yaml")

# Initialize environment
name = wdb.get_hyperparameter("environment")
env = EnvironmentManager(name, "rgb_array")
env.build_continuous()

# Initialize network
action_space, observation_space = env.get_dimensions()
#model = DiscreteActorCriticNet(action_space, observation_space)
model = ContinuousActorCriticNet(action_space, observation_space)

# Initialize PPO agent
#agent = PPOAgentDiscrete(env, wdb, model)
agent = PPOAgentContinuous(env, wdb, model)

# Start training
agent.train()

env.close()
wdb.finish()
