import numpy
import torch
from environment_manager import EnvironmentManager
from wandb_wrapper import WandbWrapper
from ppo_models import ActorCriticNet
from ppo_memory import RolloutBuffer


class PPOAgent:
    """Agent that implements Proximal Policy Optimization (PPO) algorithm."""

    def __init__(
        self,
        environment: EnvironmentManager,
        wandb: WandbWrapper,
        model: ActorCriticNet,
    ):
        """Initializes the PPO agent with the environment, wandb logger, model, and device.

        Args:
            environment (EnvironmentManager): The environment in which the agent operates.
            wandb (WandbWrapper): Wandb wrapper for tracking and hyperparameter management.
            model (ActorCriticNet): The neural network model used by the agent.
        """
        self.env = environment
        self.wdb = wandb
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.memory = RolloutBuffer()
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.model.network.parameters(),
                    "lr": self.wdb.get_hyperparameter("learning_rate_shared"),
                },
                {
                    "params": self.model.actor_head.parameters(),
                    "lr": self.wdb.get_hyperparameter("learning_rate_actor"),
                },
                {
                    "params": self.model.critic_head.parameters(),
                    "lr": self.wdb.get_hyperparameter("learning_rate_critic"),
                },
            ],
            eps=1e-5,  # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        )

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

    def optimize_model(self, final_state: numpy.ndarray) -> tuple:
        """Optimizes the model using the collected rollout data.

        Returns:
            tuple: A tuple containing the total policy loss and value loss.
        """
        # Prepare rollout data
        states, actions, log_probs, rewards, values, dones = self.memory.get_tensors()

        # Compute next value from final state
        # Make sure to estimate 0 if the episode is terminated
        with torch.no_grad():
            if dones[-1]:
                next_value = torch.tensor([0.0], dtype=torch.float32).to(self.device)
            else:
                _, next_value = self.model(
                    torch.tensor(final_state, dtype=torch.float32).to(self.device)
                )

        # Compute advantages and returns using GAE and value estimates
        advantages = self.compute_advantages(rewards, values, next_value, dones)
        returns = values + advantages

        # Optimize policy for K epochs
        epoch_count = self.wdb.get_hyperparameter("ppo_epochs")
        batch_size = self.wdb.get_hyperparameter("batch_size")
        clip_eps = self.wdb.get_hyperparameter("clip_epsilon")

        # Stats
        total_policy_loss = 0.0
        total_value_loss = 0.0
        update_count = 0

        for _ in range(epoch_count):
            # Shuffle the data for each epoch to ensure better generalisation
            # Needs to be done before batching so the data does not repeat
            indices = torch.randperm(len(states))

            for i in range(0, len(indices), batch_size):
                # Create batches
                batch_indices = indices[i:i + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Input states to the model, get logits and values according to current policy
                logits, values_pred = self.model(batch_states)
                values_pred = values_pred.squeeze(-1)
                action_probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)

                # Get the entropy for the current policy
                entropy = dist.entropy().mean()

                # Compute a ratio r_t between new and old log probabilities
                # Uses torch.exp to avoid unstable values
                ratio = torch.exp(new_log_probs - batch_log_probs)

                # First part of the surrogate loss r_t * A_t
                surr1 = ratio * batch_advantages

                # The second part of the surrogate loss r_t * A_t
                # Clipping avoids large updates which cause unstability
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                    * batch_advantages
                )

                # Compute the policy loss (negative of the surrogate loss)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute the value loss (mean squared error)
                value_loss = torch.nn.functional.mse_loss(values_pred, batch_returns)

                # Total loss = policy loss + value loss - entropy (to encourage exploration)
                loss = (
                    policy_loss
                    + self.wdb.get_hyperparameter("value_loss_coef") * value_loss
                    - self.wdb.get_hyperparameter("entropy_coef") * entropy
                )

                # Update total losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                update_count += 1

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return total_policy_loss / update_count, total_value_loss / update_count

    def train(self):
        """Main training loop for the PPO agent."""
        for episode in range(self.wdb.get_hyperparameter("episodes")):
            state = self.env.reset()
            self.memory.clear()

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
                    if episode % 10 == 0:
                        print(f"Episode {episode} done.")
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
