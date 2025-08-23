import numpy as np
import torch
from collections import deque
import os
import torch.nn.functional as F
from environment_manager import EnvironmentManager
from wandb_wrapper import WandbWrapper
from rainbow_models import DuelingDQN
from rainbow_memory import PrioritizedExperienceReplay


class RainbowAgent:
    """Agent that implements the Rainbow DQN algorithm."""

    def __init__(self, environment: EnvironmentManager, wandb: WandbWrapper):
        """Initializes the Rainbow Agent with the environment and Wandb logger.

        Args:
            environment (EnvironmentManager): The environment in which the agent operates.
            wandb (WandbWrapper): Wandb wrapper for tracking and hyperparameter management.
        """
        # Environment, wandb
        self.env = environment
        self.wdb = wandb
        self.action_count, state_count = self.env.get_dimensions()

        # Models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_network = DuelingDQN(self.action_count, state_count).to(self.device)
        self.target_network = DuelingDQN(self.action_count, state_count).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # Create memory
        self.memory = PrioritizedExperienceReplay(
            self.wdb.get_hyperparameter("memory_size"),
            self.wdb.get_hyperparameter("alpha"),
            self.wdb.get_hyperparameter("beta"),
            self.wdb.get_hyperparameter("batch_size"),
            self.device
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.wdb.get_hyperparameter("learning_rate_policy"),
        )

        # Epsilon -- Temporary solution, will be replaced by noisy nets
        self.epsilon = 1
        self.epsilon_decay = self.wdb.get_hyperparameter("epsilon_decay")

    def get_action(self, state: np.ndarray) -> int:
        """Performs epsilon greedy selection, will later be replaced in favour of noisy nets!"""
        random = np.random.random()
        if random < self.epsilon:
            return np.random.randint(0, self.action_count)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_network(state).argmax(1).item()

    def optimize(self) -> float:
        """Performs one optimization step for Double DQN with PER (TorchRL)."""
        # Temporary epsilon decay
        self.epsilon *= self.epsilon_decay

        # Sample batch from memory
        batch_size = self.wdb.get_hyperparameter("batch_size")
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(batch_size)

        # Get current Q-values
        current_q_values = self.policy_network(states).gather(1, actions)

        # Double DQN target
        gamma = self.wdb.get_hyperparameter("gamma")
        with torch.no_grad():
            next_actions = self.policy_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * gamma * next_q_values

        # TD-errors
        td_errors = target_q_values - current_q_values

        # Per-sample Smooth L1 loss
        per_sample_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none").squeeze(1)

        # Weighted PER loss
        loss = (weights.squeeze(1) * per_sample_loss).mean()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in memory
        td_errors_to_update = td_errors.detach().abs().squeeze(1) + 1e-6
        self.memory.update_priorities(indices, td_errors_to_update)

        return loss.item()

    def soft_tau_update(self):
        """Updates target networks by soft tau update (polyak averaging)."""
        tau = self.wdb.get_hyperparameter("tau")
        for src_param, target_param in zip(
            self.policy_network.parameters(), self.target_network.parameters()
        ):
            target_param.data.copy_(
                tau * src_param.data + (1 - tau) * target_param.data
            )

    def train(self):
        """Main training loop for Rainbow agent."""
        episode = 0
        total_steps = 0
        max_steps = self.wdb.get_hyperparameter("total_steps")
        warmup_steps = self.wdb.get_hyperparameter("warmup_steps")
        best_mean = float("-inf")
        save_interval = self.wdb.get_hyperparameter("save_interval")
        reward_buffer = deque(maxlen=save_interval)

        while True:
            state = self.env.reset()

            for _ in range(self.wdb.get_hyperparameter("episode_steps")):
                # Update steps
                total_steps += 1

                # Get action from the model, advance the environment
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # Add to memory, adjust new state
                self.memory.add(state, action, reward, next_state, done)
                state = next_state

                # If the warmup period is over, run optimisation
                if total_steps > warmup_steps:
                    policy_loss = self.optimize()
                    self.soft_tau_update()

                    # Log policy to Wandb
                    if total_steps % 100 == 0:
                        self.wdb.log(
                            {"Total Steps": total_steps, "Policy Loss": policy_loss}
                        )

                # If the episode is done, finish logging and model saving
                if done:
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
                            f"Total steps: {total_steps}/{max_steps}. Epsilon: {self.epsilon}"
                        )
                    break

            # Log episode parameters
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

        # When done, save the best model and close wandb
        self.save_artifact()
        self.wdb.finish()

    def play(self):
        """See the agent perform in selected environment."""
        state = self.env.reset()
        done = False
        self.epsilon = 0
        while not done:
            action = self.get_action(state)
            state, reward, done, _ = self.env.step(action)
            self.env.render()

        steps, reward = self.env.get_episode_info()
        print(f"Test run finished in {steps} steps with {reward} reward!")
        self.env.close()

    def save_model(self):
        """INFERENCE ONLY -- Saves state dict of the model"""
        path = self.wdb.get_hyperparameter("save_dir")
        name = self.wdb.get_hyperparameter("save_name")
        if not os.path.exists(path):
            raise FileNotFoundError("Save dir does not exist!")

        save_path = path + name + ".pth"
        torch.save(self.policy_network.state_dict(), save_path)

    def save_artifact(self):
        """INFERENCE ONLY -- Saves state dict to wandb"""
        path = self.wdb.get_hyperparameter("save_dir")
        name = self.wdb.get_hyperparameter("save_name")
        if not os.path.exists(path):
            raise FileNotFoundError("Save dir does not exist!")

        save_path = path + name + ".pth"
        self.wdb.log_model(name, save_path)

    def load_model(self, path):
        """INFERENCE ONLY -- Loads model state dict"""
        if not os.path.exists(path):
            raise FileNotFoundError("Path does not exist!")

        self.policy_network.load_state_dict(torch.load(path, weights_only=True))
        self.policy_network.eval()
