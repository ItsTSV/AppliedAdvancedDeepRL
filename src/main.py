from src.utils.wandb_wrapper import WandbWrapper
from src.utils.environment_manager import EnvironmentManager
from src.ppo.agent_continuous import PPOAgentContinuous
from src.ppo.models import ContinuousActorCriticNet
from src.sac.agent import SACAgent

# Initialize WandbWrapper
#wdb = WandbWrapper("config/ppo_walker2d.yaml")
wdb = WandbWrapper("config/sac_half_cheetah.yaml")

# Initialize environment
name = wdb.get_hyperparameter("environment")
env = EnvironmentManager(name, "rgb_array")
env.build_continuous()

# Initialize network
#network_size = wdb.get_hyperparameter("network_size")
#action_space, observation_space = env.get_dimensions()
#model = ContinuousActorCriticNet(action_space, observation_space, network_size)

# Initialize PPO agent
#agent = PPOAgentContinuous(env, wdb, model)
agent = SACAgent(env, wdb)

# Start training
agent.train()

env.close()
wdb.finish()
