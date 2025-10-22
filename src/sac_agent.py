import numpy as np
import torch
import itertools
import torch.nn.functional as F
import os
from collections import deque
from environment_manager import EnvironmentManager
from wandb_wrapper import WandbWrapper
from sac_models import QNet, ActorNet
from sac_memory import ReplayBuffer


class SACAgent:
    """Agent that implements Soft Actor Critic algorithm."""

    def __init__(self, environment: EnvironmentManager, wandb: WandbWrapper):
        """Initializes the SAC agent with environment, wandb, models, and memory.

        Args:
            environment (EnvironmentManager): The environment in which the agent operates.
            wandb (WandbWrapper): Wandb wrapper for tracking and hyperparameter management.
        """
        # Environment, wandb
        self.env = environment
        self.wdb = wandb
        action_count, state_count = self.env.get_dimensions()

        # Create models and memory based on env parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor = ActorNet(action_count, state_count).to(self.device)
        self.qnet1 = QNet(action_count, state_count).to(self.device)
        self.qnet2 = QNet(action_count, state_count).to(self.device)
        self.qnet1_target = QNet(action_count, state_count).to(self.device)
        self.qnet2_target = QNet(action_count, state_count).to(self.device)

        # Target networks start with same weights as policy ones
        self.qnet1_target.load_state_dict(self.qnet1.state_dict())
        self.qnet2_target.load_state_dict(self.qnet2.state_dict())

        # Create memory
        memory_size = self.wdb.get_hyperparameter("memory_size")
        self.memory = ReplayBuffer(memory_size, action_count, state_count)

        # Optimizers
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.wdb.get_hyperparameter("learning_rate_actor"),
        )
        self.optimizer_q = torch.optim.Adam(
            itertools.chain(self.qnet1.parameters(), self.qnet2.parameters()),
            lr=self.wdb.get_hyperparameter("learning_rate_q"),
        )

    def get_action(self, state) -> tuple:
        """Selects an action based on the current state using the actor.

        Args:
            state: The current state of the environment.

        Returns:
            action: Action tensor with [-1, 1 bounds]
            log_prob: Log probability of selected action
        """
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        log_std_min = self.wdb.get_hyperparameter("log_std_min")
        log_std_max = self.wdb.get_hyperparameter("log_std_max")

        with torch.no_grad():
            mean, log_std = self.actor(state)

            # Normalise (OpenAI version)
            log_std = torch.tanh(log_std)
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * log_std + 1

            # Distribution and re-parametrisation trick
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
            action = torch.tanh(z)

            # Because tanh was used, logprobs need to be adjusted
            log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            return action, log_prob

    def optimize_q_networks(self) -> tuple:
        """Optimizes Q-policy networks using data from memory

        Returns:
            tuple (float, float. float): Total Q-Loss, Q1 loss, Q2 loss
        """
        # Sample data from memory
        batch_size = self.wdb.get_hyperparameter("batch_size")
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        dones = dones.float()

        # Compute next Q-values using Q-target networks
        gamma = self.wdb.get_hyperparameter("gamma")
        alpha = self.wdb.get_hyperparameter("entropy_temperature")
        with torch.no_grad():
            next_actions, next_log_probs = self.get_action(next_states)
            next_targets_qnet1 = self.qnet1_target(next_states, next_actions)
            next_targets_qnet2 = self.qnet2_target(next_states, next_actions)
            min_next_targets = torch.min(next_targets_qnet1, next_targets_qnet2)
            next_q_values = rewards + gamma * (1 - dones) * (
                min_next_targets - alpha * next_log_probs
            )

        # Get current Q-values using Q-policy networks
        q1_values = self.qnet1(states, actions)
        q2_values = self.qnet2(states, actions)

        # Compute losses
        q1_loss = F.mse_loss(q1_values, next_q_values)
        q2_loss = F.mse_loss(q2_values, next_q_values)
        q_loss = q1_loss + q2_loss

        # Optimise Q-policy networks via gradient descent
        self.optimizer_q.zero_grad()
        q_loss.backward()
        self.optimizer_q.step()

        return q_loss.item(), q1_loss.item(), q2_loss.item()

    def optimize_actor_network(self) -> float:
        """Optimizes actor network using data from memory"""
        # Sample data from memory
        batch_size = self.wdb.get_hyperparameter("batch_size")
        states, _, _, _, _ = self.memory.sample(batch_size)

        # Get next_actions and their Q-values
        next_actions, log_probs = self.get_action(states)
        q1_values = self.qnet1(states, next_actions)
        q2_values = self.qnet2(states, next_actions)
        min_q_values = torch.min(q1_values, q2_values)

        # Compute actor loss
        alpha = self.wdb.get_hyperparameter("entropy_temperature")
        actor_loss = log_probs * alpha - min_q_values
        actor_loss = actor_loss.mean()

        # Run gradient descent
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        return actor_loss.item()

    def polyak_update(self, source: torch.nn.Module, target: torch.nn.Module):
        """Updates target networks by polyak averaging."""
        tau = self.wdb.get_hyperparameter("tau")
        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                tau * src_param.data + (1 - tau) * target_param.data
            )

    def train(self):
        """Main training loop for the SAC agent."""
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

                # Get action from the model
                action, _ = self.get_action(state)
                action = action.item()

                # Advance the environment
                next_state, reward, done, _ = self.env.step(action)

                # Add to memory, adjust new state
                self.memory.add(state, action, reward, next_state, done)
                state = next_state

                # If a warmup period is over, run optimisation
                if total_steps > warmup_steps:
                    q_loss, q1_loss, q2_loss = self.optimize_q_networks()
                    actor_loss = self.optimize_actor_network()
                    self.polyak_update(self.qnet1, self.qnet1_target)
                    self.polyak_update(self.qnet2, self.qnet2_target)

                    # Prevent wandb spam, log only occasionally
                    if total_steps % 100 == 0:
                        self.wdb.log(
                            {
                                "Total Steps": total_steps,
                                "Q Loss": q_loss,
                                "Q1 Loss": q1_loss,
                                "Q2 Loss": q2_loss,
                                "Actor Loss": actor_loss,
                            }
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

        # When done, save the best model to wandb and close
        self.save_artifact()
        self.wdb.finish()

    def play(self):
        """See the agent perform in selected environment."""
        state = self.env.reset()
        done = False
        while not done:
            action, _ = self.get_action(state)
            state, reward, done, _ = self.env.step(action.item())
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
        torch.save(self.actor.state_dict(), save_path)

    def save_artifact(self):
        """INFERENCE ONLY -- Saves state dict to wandb"""
        path = self.wdb.get_hyperparameter("save_dir")
        name = self.wdb.get_hyperparameter("save_name")
        if not os.path.exists(path):
            raise FileNotFoundError("Save dir does not exist!")

        save_path = path + name + ".pth"
        self.wdb.log_model(name, save_path)

    def load_model(self, path: str):
        """INFERENCE ONLY -- Loads state dict of the model"""
        if not os.path.exists(path):
            raise FileNotFoundError("Path does not exist!")

        self.actor.load_state_dict(torch.load(path, weights_only=True))
        self.actor.eval()
