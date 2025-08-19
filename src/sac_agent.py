import numpy as np
import torch
import os
import itertools
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

        # Polyak averaging
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

    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass

    def optimize_models(self) -> tuple[float, float]:
        pass

    def update_targets(self):
        pass

    def train(self):
        """Main training loop for the SAC agent."""
        episode = 0
        total_steps = 0
        max_steps = self.wdb.get_hyperparameter("total_steps")
        warmup_steps = self.wdb.get_hyperparameter("warmup_steps")
        best_mean = float("-inf")
        save_interval = self.wdb.get_hyperparameter("save_interval")
        reward_buffer = deque([float("-inf")] * save_interval, maxlen=save_interval)

        while True:
            state = self.env.reset()

            for _ in range(self.wdb.get_hyperparameter("episode_steps")):
                # Update steps
                total_steps += 1

                # Get action from the model
                action = self.get_action(state)

                # Advance the environment
                next_state, reward, done, _ = self.env.step(action)

                # Add to memory, adjust new state
                self.memory.add(state, action, reward, next_state, done)
                state = next_state

                # If a warmup period is over, run optimisation
                if total_steps > warmup_steps:
                    q_loss, actor_loss = self.optimize_models()
                    self.update_targets()

                    # Prevent wandb spam, log only occasionally
                    if total_steps % 100 == 0:
                        self.wdb.log(
                            {
                                "Total Steps": total_steps,
                                "Q Loss": q_loss,
                                "Actor Loss": actor_loss
                            }
                        )

                # If the episode is done, finish logging etc. and optimise
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
                        self.save_models()
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

    def save_models(self):
        pass

    def save_artifact(self):
        pass
