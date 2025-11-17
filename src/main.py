from src.utils.wandb_wrapper import WandbWrapper
from src.utils.environment_manager import EnvironmentManager
from src.ppo.agent_continuous import PPOAgentContinuous
from src.ppo.models import ContinuousActorCriticNet

# Initialize WandbWrapper
#wdb = WandbWrapper("config/rainbow.yaml")
#wdb = WandbWrapper("config/sac.yaml")
#wdb = WandbWrapper("config/ppo_discrete.yaml")
wdb = WandbWrapper("config/ppo_ant.yaml")

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
