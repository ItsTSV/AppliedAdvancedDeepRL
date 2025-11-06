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
wdb = WandbWrapper("../config/ppo_hopper.yaml", mode="disabled")

# Initialize environment
name = wdb.get_hyperparameter("environment")
env = EnvironmentManager(name, "rgb_array")
env.build_continuous()
env.build_video_recorder()

# Initialize network
network_size = wdb.get_hyperparameter("network_size")
action_space, observation_space = env.get_dimensions()
model = ContinuousActorCriticNet(action_space, observation_space, network_size)

# Initialize PPO agent
agent = PPOAgentContinuous(env, wdb, model)

# Load model
agent.load_model("../models/ppo_hopper.pth")

# Play
'''
total_reward = 0
for i in range(10):
    reward = agent.play()
    total_reward += reward
    print(f"Episode {i+1}: Reward = {reward}")

print(f"Average Reward over 10 episodes: {total_reward / 10}")
'''
reward = agent.play()
print(f"Reward: {reward}")

# Finish stuff
env.close()
wdb.finish()
