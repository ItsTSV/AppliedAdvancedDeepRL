import gymnasium as gym
from gymnasium import wrappers
import numpy as np


class EnvironmentManager:
    """Provides a manager that builds gymnasium environment and handles
    interaction and data processing."""

    def __init__(self, name: str, render_mode: str):
        """Initializes Gymnasium environment and info about it

        Attributes:
            name (str): Selected environment, see https://gymnasium.farama.org/
            render_mode (str): "rgb_array" for computations, "human" for showcases
        """
        self.env = gym.make(name, render_mode=render_mode)
        self.episode_steps = 0
        self.episode_reward = 0
        self.observation_norm_wrapper = None

    def build_continuous(self):
        """Wraps itself in wrappers that are used for environments with continuous action space.

        Clip actions -- normalises the input action to [-1, 1] range.
        Normalize Observation -- normalises observations to have mean 0 and variance 1.
        Record Episode Statistics -- records episode reward and length in info dict.
        Normalize Reward -- normalises rewards to have mean 0 and variance 1.
        """
        self.env = wrappers.ClipAction(self.env)
        self.env = wrappers.NormalizeObservation(self.env)
        self.observation_norm_wrapper = self.env
        self.env = wrappers.RecordEpisodeStatistics(self.env)
        self.env = wrappers.NormalizeReward(self.env)

    def build_video_recorder(self, video_folder: str = "outputs/", fps: int = 120):
        """Wraps itself in a video recorder wrapper.

        Args:
            video_folder (str): Folder where videos will be saved.
            fps (int): Frames per second for the recorded video.
        """
        self.env.metadata["render_fps"] = fps
        self.env = wrappers.RecordVideo(
            self.env,
            video_folder,
            episode_trigger=lambda episode_id: True,
        )

    def save_normalization_parameters(self, path):
        """Saves the observation normalization mean and variance to a file.

        Args:
            path (str): The file path where the normalization parameters will be saved.
        """
        if self.observation_norm_wrapper is None:
            raise ValueError(
                "Normalization wrapper not found. Ensure build_continuous() has been called."
            )
        rms = self.observation_norm_wrapper.obs_rms
        np.savez(path, mean=rms.mean, var=rms.var, count=rms.count)

    def load_normalization_parameters(self, path):
        """Loads the observation normalization mean and variance from a file.

        Args:
            path (str): The file path from which the normalization parameters will be loaded.
        """
        if self.observation_norm_wrapper is None:
            raise ValueError(
                "Normalization wrapper not found. Ensure build_continuous() has been called."
            )
        print("Loaded!")
        data = np.load(path)
        rms = self.observation_norm_wrapper.obs_rms
        rms.mean = data["mean"]
        rms.var = data["var"]
        rms.count = data["count"]

    def get_dimensions(self) -> tuple:
        return (
            (
                len(self.env.action_space.sample())
                if isinstance(self.env.action_space, gym.spaces.Box)
                else self.env.action_space.n
            ),
            len(self.env.observation_space.sample()),
        )

    def get_state_shape(self) -> tuple:
        return self.env.observation_space.shape

    def get_random_action(self) -> np.ndarray:
        return self.env.action_space.sample()

    def step(self, action) -> tuple:
        """Advances the environment, processes the output

        Returns:
            New observation, reward acquired by performing action,
            termination info, additional env data
        """
        state, reward, terminated, truncated, info = self.env.step(action)
        finished = terminated or truncated

        self.episode_steps += 1
        if "episode" in info:
            self.episode_reward = info["episode"]["r"]

        return state, reward, terminated, finished, info

    def reset(self) -> np.ndarray:
        """Resets the environment, returns a new state"""
        new_state = self.env.reset()[0]
        self.episode_steps = 0
        self.episode_reward = 0
        return new_state

    def get_episode_info(self) -> tuple:
        return self.episode_steps, self.episode_reward

    def close(self) -> None:
        self.env.close()

    def render(self):
        return self.env.render()
