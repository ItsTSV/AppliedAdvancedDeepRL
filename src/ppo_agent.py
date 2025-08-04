import numpy
import torch
from environment_manager import EnvironmentManager
from wandb_wrapper import WandbWrapper
from ppo_models import ActorCriticNet
from ppo_memory import RolloutBuffer


class PPOAgent:
    """Agent that implements Proximal Policy Optimization (PPO) algorithm."""

    def __init__(self, environment: EnvironmentManager, wandb: WandbWrapper, model: ActorCriticNet):
        """Initializes the PPO agent with the environment, wandb logger, model, and device.

        Args:
            environment (EnvironmentManager): The environment in which the agent operates.
            wandb (WandbWrapper): Weights and Biases logger for tracking experiments and hyperparameter management.
            model (ActorCriticNet): The neural network model used by the agent.
        """
        self.env = environment
        self.wdb = wandb
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.memory = RolloutBuffer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.wdb.get_hyperparameter("learning_rate"))

    def get_action(self, state: numpy.ndarray) -> tuple:
        """Selects an action based on the current state using the model.

        Args:
            state (numpy.ndarray): The current state of the environment.

        Returns:
            tuple: A tuple containing the selected action, its log probability and value estimate.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, value = self.model(state_tensor)
            action_probs = torch.softmax(logits, dim=-1)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            log_prob = action_distribution.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def train(self):
        """Training loop for the PPO agent."""
        for episode in range(self.wdb.get_hyperparameter("episodes")):
            state = self.env.reset()
            self.memory.clear()

            for step in range(self.wdb.get_hyperparameter("max_steps")):
                # Select an action based on the current state
                action, log_prob, value = self.get_action(state)

                # Take a step in the environment
                new_state, reward, finished, info = self.env.step(action)

                # Store the transition in memory, update state
                self.memory.add(state, action, log_prob, reward, value, finished)
                state = new_state

                if finished:
                    break

            # Process the collected data and update the model


            # Log episode information
            episode_steps, episode_reward = self.env.get_episode_info()
            self.wdb.log({
                "episode": episode,
                "steps": episode_steps,
                "reward": episode_reward,
                "mean_reward": episode_reward / episode_steps if episode_steps > 0 else 0
            })
