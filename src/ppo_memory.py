import numpy as np


class RolloutBuffer:
    """Buffer that serves as a storage for PPO episode data."""

    def __init__(self):
        """Initializes the buffer for storing agent's actions, states, log probabilities,
        rewards, values, and episode finishers.
        """
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.finishers = []

    def add(self, state, action, logprob, reward, value, finished):
        """Adds a new step to the buffer.

        Args:
            state (np.ndarray): State of the environment
            action (np.ndarray): Action taken by the agent
            logprob (float): Log probability of the action
            reward (float): Reward received from the environment
            value (float): Value estimate of the state
            finished (bool): Whether the episode has finished
        """
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.finishers.append(finished)

    def clear(self):
        """Clears the buffer, removing all stored data"""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.finishers[:]
