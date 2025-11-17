from wandb_wrapper import WandbWrapper
import numpy as np
from environment_manager import EnvironmentManager
from ppo_agent_discrete import PPOAgentDiscrete
from ppo_agent_continuous import PPOAgentContinuous
from ppo_models import DiscreteActorCriticNet, ContinuousActorCriticNet
from sac_agent import SACAgent
from rainbow_agent import RainbowAgent

# Initialize WandbWrapper
#wdb = WandbWrapper("../config/rainbow.yaml")
#wdb = WandbWrapper("../config/sac.yaml")
#wdb = WandbWrapper("../config/ppo_discrete.yaml")
wdb = WandbWrapper("../config/ppo_ant.yaml")

# Initialize environment
name = wdb.get_hyperparameter("environment")
env = EnvironmentManager(name, "rgb_array")
env.build_continuous()

# Initialize network
network_size = wdb.get_hyperparameter("network_size")
action_space, observation_space = env.get_dimensions()
#model = DiscreteActorCriticNet(action_space, observation_space, network_size)
model = ContinuousActorCriticNet(action_space, observation_space, network_size)

# Initialize PPO agent
#agent = PPOAgentDiscrete(env, wdb, model)
agent = PPOAgentContinuous(env, wdb, model)
#agent = SACAgent(env, wdb)
#agent = RainbowAgent(env, wdb)

# Start training
agent.train()

env.close()
wdb.finish()
