import gymnasium as gym


class EnvironmentManager:
    """Provides a manager that builds gymnasium environment and handles
    interaction and data processing
    """

    def __init__(self, name: str, render_mode: str):
        """Initializes Gymnasium environment and info about it

        Attributes:
            name (str): Selected environment, see https://gymnasium.farama.org/
            render_mode (str): "rgb_array" for computations, "human" for showcases
        """
        self.env = gym.make(name, render_mode=render_mode)
        self.episode_steps = 0
        self.episode_reward = 0

    def get_dimensions(self) -> tuple:
        """Returns state and observation space dimension"""
        return (
            (
                len(self.env.action_space.sample())
                if isinstance(self.env.action_space, gym.spaces.Box)
                else self.env.action_space.n
            ),
            len(self.env.observation_space.sample()),
        )

    def step(self, action) -> tuple:
        """Advances the environment, processes the output

        Returns:
            New observation, reward acquired by performing action,
            termination info, additional env data
        """
        state, episode_reward, terminated, truncated, info = self.env.step(action)
        finished = terminated or truncated

        self.episode_steps += 1
        self.episode_reward += episode_reward
        return state, episode_reward, finished, info

    def reset(self) -> tuple:
        """Resets the environment, returns a new state"""
        new_state = self.env.reset()[0]
        self.episode_steps = 0
        self.episode_reward = 0
        return new_state

    def get_episode_info(self) -> tuple:
        """Returns the number of steps and total reward in the current episode"""
        return self.episode_steps, self.episode_reward

    def close(self) -> None:
        self.env.close()

    def render(self):
        """Wrapper for gymnasium render function. The return is based on
        selected render mode.
        """
        return self.env.render()
