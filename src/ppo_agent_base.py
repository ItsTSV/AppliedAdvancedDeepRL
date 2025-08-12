import numpy as np
import torch
from abc import ABC, abstractmethod
from collections import deque
from environment_manager import EnvironmentManager
from wandb_wrapper import WandbWrapper
from ppo_memory import RolloutBuffer


class PPOAgentBase(ABC):
    """Agent that serves as a base for PPO algorithm. Contains common methods and properties."""

    def __init__(self, environment: EnvironmentManager, wandb: WandbWrapper, model):
        """Initializes the PPO agent with the environment, wandb logger, model, and device.

        Args:
            environment (EnvironmentManager): The environment in which the agent operates.
            wandb (WandbWrapper): Wandb wrapper for tracking and hyperparameter management.
            model: The neural network model used by the agent.
        """
        self.env = environment
        self.wdb = wandb
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.memory = RolloutBuffer()

    @abstractmethod
    def get_action(self, state: np.ndarray) -> tuple:
        """Selects an action based on the current state using the model."""
        pass

    @abstractmethod
    def evaluate_actions(
        self, batch_states: torch.tensor, batch_actions: torch.tensor
    ) -> tuple:
        """Evaluates actions for a batch of states."""
        pass

    @abstractmethod
    def optimize_model(self, final_state: np.ndarray) -> tuple:
        """Optimizes the model using the collected rollout data."""
        pass

    def compute_advantages(
        self,
        rewards: torch.tensor,
        values: torch.tensor,
        last_value: torch.tensor,
        dones: torch.tensor,
    ) -> torch.tensor:
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(rewards, dtype=torch.float32).to(self.device)
        gae = 0.0
        gamma = self.wdb.get_hyperparameter("gamma")
        lmbda = self.wdb.get_hyperparameter("lambda")
        values = torch.cat([values, last_value], dim=0)

        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + gamma * (1 - dones[step]) * values[step + 1]
                - values[step]
            )
            gae = delta + gamma * lmbda * (1 - dones[step]) * gae
            advantages[step] = gae

        # Normalize advantages to stabilize training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def train(self):
        """Main training loop for the PPO agent."""
        for episode in range(self.wdb.get_hyperparameter("episodes")):
            state = self.env.reset()
            self.memory.clear()
            reward_buffer = deque([0] * 10, maxlen=10)

            for _ in range(self.wdb.get_hyperparameter("max_steps")):
                # Get action from the model
                action, log_prob, value = self.get_action(state)

                # Advance the environment
                new_state, reward, done, _ = self.env.step(action)

                # Add step to memory, change state
                self.memory.add(state, action, log_prob, reward, value, done)
                state = new_state

                # If the episode is done, break the loop and optionally print for sanity check
                if done:
                    _, episode_reward = self.env.get_episode_info()
                    reward_buffer.append(episode_reward)

                    if episode % 10 == 0:
                        mean = np.sum(reward_buffer) / 10
                        print(
                            f"Episode {episode} done: mean reward over last ten episodes: {mean}"
                        )
                    break

            # Perform an optimisation step
            value_loss, policy_loss = self.optimize_model(state)

            # Logging episode results
            episode_steps, episode_reward = self.env.get_episode_info()
            self.wdb.log(
                {
                    "episode": episode,
                    "steps": episode_steps,
                    "reward": episode_reward,
                    "mean_reward": (
                        episode_reward / episode_steps if episode_steps > 0 else 0
                    ),
                    "value_loss": value_loss,
                    "policy_loss": policy_loss,
                }
            )
