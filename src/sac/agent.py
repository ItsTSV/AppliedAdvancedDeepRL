import numpy as np
import torch
import itertools
import torch.nn.functional as F
from pathlib import Path
from collections import deque
from src.utils.environment_manager import EnvironmentManager
from src.utils.wandb_wrapper import WandbWrapper
from .models import QNet, ActorNet
from .memory import ReplayBuffer


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
        network_size = self.wdb.get_hyperparameter("network_size")

        # Actor
        self.actor = ActorNet(action_count, state_count, network_size).to(self.device)

        # Q-Networks
        self.qnet1 = QNet(action_count, state_count, network_size).to(self.device)
        self.qnet2 = QNet(action_count, state_count, network_size).to(self.device)

        # Target networks
        self.qnet1_target = QNet(action_count, state_count, network_size).to(self.device)
        self.qnet2_target = QNet(action_count, state_count, network_size).to(self.device)

        # Adaptive entropy temperature
        self.target_entropy = -float(action_count)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        # Target networks start with same weights as policy ones
        self.qnet1_target.load_state_dict(self.qnet1.state_dict())
        self.qnet2_target.load_state_dict(self.qnet2.state_dict())

        # Create memory
        memory_size = self.wdb.get_hyperparameter("memory_size")
        self.memory = ReplayBuffer(memory_size, action_count, state_count)

        # Lock target networks
        for param in self.qnet1_target.parameters():
            param.requires_grad = False
        for param in self.qnet2_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.wdb.get_hyperparameter("learning_rate_actor"),
        )
        self.optimizer_q = torch.optim.Adam(
            itertools.chain(self.qnet1.parameters(), self.qnet2.parameters()),
            lr=self.wdb.get_hyperparameter("learning_rate_q"),
        )
        self.optimizer_alpha = torch.optim.Adam(
            [self.log_alpha],
            lr=self.wdb.get_hyperparameter("learning_rate_actor")
        )

        # File handling
        current_path = Path(__file__).resolve()
        self.project_root = current_path.parent.parent.parent

    def get_action(self, state) -> tuple:
        """Selects an action based on the current state using the actor.

        Args:
            state: The current state of the environment.

        Returns:
            action: Action tensor with [-1, 1 bounds]
            log_prob: Log probability of selected action
        """
        # Get log std range hyperparameters
        log_std_min = self.wdb.get_hyperparameter("log_std_min")
        log_std_max = self.wdb.get_hyperparameter("log_std_max")

        # Get mean and log standard deviation of action
        mean, log_std = self.actor(state)

        # Normalise (OpenAI version)
        log_std = torch.tanh(log_std)
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

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
        alpha = self.log_alpha.exp().item()
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

    def optimize_actor_network(self) -> tuple:
        """Optimizes actor network and entropy temperature using data from memory"""
        # Sample data from memory
        batch_size = self.wdb.get_hyperparameter("batch_size")
        states, _, _, _, _ = self.memory.sample(batch_size)

        # Get current_actions and their Q-values
        current_actions, log_probs = self.get_action(states)
        q1_values = self.qnet1(states, current_actions)
        q2_values = self.qnet2(states, current_actions)
        min_q_values = torch.min(q1_values, q2_values)

        # Compute actor loss
        alpha = self.log_alpha.exp().item()
        actor_loss = log_probs * alpha - min_q_values
        actor_loss = actor_loss.mean()

        # Optimize actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        # Optimize
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()

        return actor_loss.item(), alpha_loss.item(), alpha

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

                # Get action (from network -- needs to be detached and have its batch removed)
                if total_steps < warmup_steps:
                    action = self.env.get_random_action()
                else:
                    state_tensor = torch.tensor(state).to(self.device).unsqueeze(0)
                    action, log_probs = self.get_action(state_tensor)
                    action = action.detach().cpu().numpy()[0]

                # Advance the environment
                next_state, reward, terminated, done, _ = self.env.step(action)

                # Add to memory, adjust new state
                self.memory.add(state, action, reward, next_state, terminated)
                state = next_state

                # If a warmup period is over, run optimisation
                if total_steps > warmup_steps:
                    q_loss, q1_loss, q2_loss = self.optimize_q_networks()
                    actor_loss, alpha_loss, alpha = self.optimize_actor_network()
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
                                "Alpha Loss": alpha_loss,
                                "Alpha": alpha,
                            }
                        )

                # If the episode is done, finish logging and model saving
                if done:
                    # Update episode & steps info
                    episode_steps, episode_reward = self.env.get_episode_info()
                    episode += 1

                    # Calculate mean rewards in last episodes
                    reward_buffer.append(episode_reward)
                    mean = np.mean(reward_buffer)

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
            state = torch.tensor(state).to(self.device).unsqueeze(0)
            action, _ = self.get_action(state)
            action = action.detach().cpu().numpy()[0]
            state, reward, _, done, _ = self.env.step(action)
            self.env.render()

        steps, reward = self.env.get_episode_info()
        print(f"Test run finished in {steps} steps with {reward} reward!")
        self.env.close()

    def save_model(self):
        """INFERENCE ONLY -- Saves state dict of the model"""
        dir_parameter = self.wdb.get_hyperparameter("save_dir")
        name = self.wdb.get_hyperparameter("save_name")

        # Get save directory path
        abs_save_dir = self.project_root / dir_parameter

        # Create directory
        abs_save_dir.mkdir(parents=True, exist_ok=True)

        # Get paths
        model_path = (abs_save_dir / name).with_suffix(".pth")
        rms_name = name + "_rms"
        rms_path = (abs_save_dir / rms_name).with_suffix(".npz")

        # Save
        torch.save(self.actor.state_dict(), str(model_path))
        self.env.save_normalization_parameters(str(rms_path))

    def save_artifact(self):
        """INFERENCE ONLY -- Saves state dict to wandb"""
        dir_parameter = self.wdb.get_hyperparameter("save_dir")
        name = self.wdb.get_hyperparameter("save_name")

        # Get save directory path
        abs_save_dir = self.project_root / dir_parameter

        # Create directory
        abs_save_dir.mkdir(parents=True, exist_ok=True)

        # Get paths
        model_path = (abs_save_dir / name).with_suffix(".pth")
        rms_name = name + "_rms"
        rms_path = (abs_save_dir / rms_name).with_suffix(".npz")

        # Save to WDB (online)
        self.wdb.log_model(name, str(model_path))
        self.wdb.log_model(name + "_rms", str(rms_path))

    def load_model(self, path):
        """INFERENCE ONLY -- Loads state dict of the model"""
        # Model
        model_path = self.project_root / path

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Load and eval
        self.actor.load_state_dict(torch.load(model_path, weights_only=True))
        self.actor.eval()

        # Rms
        rms_path = model_path.parent / (model_path.stem + "_rms.npz")

        if not rms_path.exists():
            raise FileNotFoundError(f"Normalization file not found: {rms_path}")

        self.env.load_normalization_parameters(str(rms_path))
