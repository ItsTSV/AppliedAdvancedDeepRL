import numpy
import torch
from torch.distributions import Normal
from environment_manager import EnvironmentManager
from wandb_wrapper import WandbWrapper
from ppo_models import ContinuousActorCriticNet
from ppo_agent_base import PPOAgentBase


class PPOAgentContinuous(PPOAgentBase):
    """Agent that implements Proximal Policy Optimization (PPO) algorithm
      for continuous action spaces, such as MuJoCo.

    This agent derives most of its functionality from PPOAgent class, which
    implements the PPO algorithm for discrete action spaces. The main difference
    is in network architecture, optimiser settings and action selection and evaluation.
    """

    def __init__(self, environment: EnvironmentManager, wandb: WandbWrapper, model: ContinuousActorCriticNet):
        super().__init__(environment, wandb, model)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.model.actor.parameters(),
                    "lr": self.wdb.get_hyperparameter("learning_rate_actor"),
                },
                {
                    "params": self.model.critic.parameters(),
                    "lr": self.wdb.get_hyperparameter("learning_rate_critic"),
                },
            ],
            eps=1e-5,
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
            mean, log_std, value = self.model(state_tensor)
            action_std = torch.exp(log_std)
            distribution = Normal(mean, action_std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)

        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(self, batch_states: torch.tensor, batch_actions: torch.tensor) -> tuple:
        mean, log_std, values_pred = self.model(batch_states)
        values_pred = values_pred.squeeze(-1)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        return dist, new_log_probs, entropy, values_pred

    def optimize_model(self, final_state: numpy.ndarray) -> tuple:
        """Optimizes the model using the collected rollout data.

        Returns:
            tuple: A tuple containing the total policy loss and value loss.
        """
        pass
