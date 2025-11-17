import numpy as np
import torch
import os
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
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> tuple:
        """Selects an action based on the current state using the model."""
        pass

    @abstractmethod
    def evaluate_actions(
        self, batch_states: torch.Tensor, batch_actions: torch.Tensor
    ) -> tuple:
        """Evaluates actions for a batch of states."""
        pass

    @abstractmethod
    def optimize_model(self, final_state: np.ndarray) -> tuple:
        """Optimizes the model using the collected rollout data."""
        pass

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        last_value: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
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
        episode = 0
        total_steps = 0
        max_steps = self.wdb.get_hyperparameter("total_steps")
        best_mean = float("-inf")
        save_interval = self.wdb.get_hyperparameter("save_interval")
        reward_buffer = deque(maxlen=save_interval)
        policy_loss_buffer = deque(maxlen=5)
        value_loss_buffer = deque(maxlen=5)

        while True:
            state = self.env.reset()

            for step in range(self.wdb.get_hyperparameter("episode_steps")):
                # Update steps
                total_steps += 1

                # Get action from the model
                action, log_prob, value = self.get_action(state)

                # Advance the environment
                new_state, reward, done, _ = self.env.step(action)

                # Add step to memory, change state
                rollout_size = self.memory.add(
                    state, action, log_prob, reward, value, done
                )
                state = new_state

                # If the rollout length is achieved, optimise
                if rollout_size == self.wdb.get_hyperparameter("rollout_length"):
                    value_loss, policy_loss = self.optimize_model(state)
                    value_loss_buffer.append(value_loss)
                    policy_loss_buffer.append(policy_loss)
                    self.memory.clear()
                    self.wdb.log(
                        {
                            "Total Steps": total_steps,
                            "Value Loss": np.mean(value_loss_buffer),
                            "Policy Loss": np.mean(policy_loss_buffer),
                        }
                    )

                # If the episode is done, break the loop and optionally print for sanity check
                if done or step == self.wdb.get_hyperparameter("episode_steps") - 1:
                    # Update episode & steps info
                    episode_steps, episode_reward = self.env.get_episode_info()
                    episode += 1

                    # Calculate mean rewards in last episodes
                    reward_buffer.append(episode_reward)
                    mean = np.sum(reward_buffer) / save_interval

                    # Save model callback
                    if mean > best_mean:
                        best_mean = mean
                        self.save_model()
                        print(
                            f"Episode {episode} -- saving model with new best mean reward: {mean}"
                        )

                    # Sanity check report
                    if episode % 10 == 0:
                        print(
                            f"Episode {episode} finished in {episode_steps} steps with reward {episode_reward}. "
                            f"Total steps: {total_steps}/{max_steps}."
                        )
                    break

            # Logging episode results
            episode_steps, episode_reward = self.env.get_episode_info()
            self.wdb.log(
                {
                    "Total Steps": total_steps,
                    "Episode Length": episode_steps,
                    "Episode Reward": episode_reward,
                    "Rolling Return": np.mean(reward_buffer),
                }
            )

            # Terminate?
            if total_steps > max_steps:
                print("The training has successfully finished!")
                break

        # When done, save the best model to wandb and close
        self.save_artifact()
        self.wdb.finish()

    def play(self) -> tuple:
        """See the agent perform in selected environment."""
        state = self.env.reset()
        done = False
        while not done:
            action, _, _ = self.get_action(state, deterministic=True)
            state, reward, done, _ = self.env.step(action)
            self.env.render()

        steps, reward = self.env.get_episode_info()
        return reward, steps

    def save_model(self):
        """INFERENCE ONLY -- Saves state dict of the model"""
        path = self.wdb.get_hyperparameter("save_dir")
        name = self.wdb.get_hyperparameter("save_name")
        if not os.path.exists(path):
            raise FileNotFoundError("Save dir does not exist!")

        save_path = path + name + ".pth"
        torch.save(self.model.state_dict(), save_path)
        self.env.save_normalization_parameters(path + name + "_rms.npz")

    def save_artifact(self):
        """INFERENCE ONLY -- Saves state dict to wandb"""
        path = self.wdb.get_hyperparameter("save_dir")
        name = self.wdb.get_hyperparameter("save_name")
        if not os.path.exists(path):
            raise FileNotFoundError("Save dir does not exist!")

        save_path = path + name + ".pth"
        self.wdb.log_model(name, save_path)
        self.wdb.log_model(name + "_rms", path + name + "_rms.npz")

    def load_model(self, path):
        """INFERENCE ONLY -- Loads state dict of the model"""
        if not os.path.exists(path):
            raise FileNotFoundError("Path does not exist!")

        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        self.env.load_normalization_parameters(path.replace(".pth", "_rms.npz"))
