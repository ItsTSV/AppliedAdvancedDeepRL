import numpy as np
import torch
from collections import deque
import os
from src.utils.environment_manager import EnvironmentManager
from src.utils.wandb_wrapper import WandbWrapper
from .models import DuelingQRDQN
from .memory import PrioritizedExperienceReplay


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

        # Epsilon
        self.epsilon = 1
        self.epsilon_decay = self.wdb.get_hyperparameter("epsilon_decay")

        # Models
        self.quantile_count = self.wdb.get_hyperparameter("quantile_count")
        self.use_noisy = self.wdb.get_hyperparameter("use_noisy")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_network = DuelingQRDQN(
            self.action_count, state_count, self.use_noisy, self.quantile_count
        ).to(self.device)
        self.target_network = DuelingQRDQN(
            self.action_count, state_count, self.use_noisy, self.quantile_count
        ).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # Create memory
        self.memory = PrioritizedExperienceReplay(
            self.wdb.get_hyperparameter("memory_size"),
            self.wdb.get_hyperparameter("alpha"),
            self.wdb.get_hyperparameter("beta"),
            self.wdb.get_hyperparameter("batch_size"),
            self.device,
            self.wdb.get_hyperparameter("gamma"),
            self.wdb.get_hyperparameter("n_step"),
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.wdb.get_hyperparameter("learning_rate_policy"),
        )

    def get_noisy_action(self, state: np.ndarray) -> int:
        """Selects action using NoisyNets"""
        self.policy_network.reset_noise()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_network(state).mean(2).argmax(1).item()

    def epsilon_greedy_action(self, state: np.ndarray) -> int:
        """Selects action using Epsilon greedy method"""
        rnd = np.random.random()
        if rnd < self.epsilon:
            return np.random.randint(0, self.action_count)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_network(state).mean(2).argmax(1).item()

    def decay_epsilon(self):
        """Slowly anneal epsilon to encourage more exploitation"""
        self.epsilon = max(0.02, self.epsilon * self.epsilon_decay)

    def optimize(self) -> float:
        """Performs one optimization step for DQN + Double + Dueling + PER + NoisyNets, will later be adjusted"""
        # Perform exploration vs. exploitation step
        if self.use_noisy:
            self.policy_network.reset_noise()
        else:
            self.decay_epsilon()

        # Sample batch from memory
        batch_size = self.wdb.get_hyperparameter("batch_size")
        states, actions, rewards, next_states, dones, weights, indices = (
            self.memory.sample(batch_size)
        )

        # Get current Q-values
        current_quantiles = self.policy_network(states).gather(
            1, actions.unsqueeze(-1).expand(-1, -1, self.quantile_count)
        )

        # Double DQN target with n-step gamma
        # This is an expand-unsqueeze-something hell; basically, all the tensors that are used for calculations should
        # be in shape [BATCH, 1, QUANTILES], so all the reshaping is done to ensure that.
        gamma = self.wdb.get_hyperparameter("gamma")
        n_step = self.wdb.get_hyperparameter("n_step")
        with torch.no_grad():
            next_actions = (
                self.policy_network(next_states).mean(2).argmax(1).unsqueeze(-1)
            )
            next_quantiles = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(-1).expand(-1, -1, self.quantile_count)
            )
            rewards_expanded = rewards.unsqueeze(-1).expand(-1, -1, self.quantile_count)
            dones_expanded = dones.unsqueeze(-1).expand(-1, -1, self.quantile_count)
            target_quantiles = (
                rewards_expanded
                + (1 - dones_expanded) * (gamma**n_step) * next_quantiles
            )

        # TD errors
        td_errors = target_quantiles - current_quantiles.transpose(1, 2)

        # Quantile Huber Loss
        kappa = 1.0
        tau_hat = (
            (torch.arange(0, self.quantile_count, device=self.device).float() + 0.5)
            / self.quantile_count
        ).view(1, -1, 1)

        huber_loss = torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa),
        )

        # Quantile weights
        quantile_weights = torch.abs(tau_hat - (td_errors.detach() < 0).float())

        # Final loss
        quantile_loss = quantile_weights * huber_loss
        per_sample_loss = quantile_loss.sum(2).mean(1)

        # Weighted PER loss
        loss = (weights * per_sample_loss).mean()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities
        td_errors_abs = td_errors.detach().abs().mean(dim=(1, 2))
        priorities = td_errors_abs.cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities)

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
                action = (
                    self.get_noisy_action(state)
                    if self.use_noisy
                    else self.epsilon_greedy_action(state)
                )
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
                            f"Total steps: {total_steps}/{max_steps}."
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
            action = (
                self.get_noisy_action(state)
                if self.use_noisy
                else self.epsilon_greedy_action(state)
            )
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
