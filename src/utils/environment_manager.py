import gymnasium as gym
from gymnasium import wrappers
import numpy as np


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
        self.norm_wrapper = None

    def build_continuous(self):
        """Wraps itself in wrappers that are used for environments with continuous action space.

        Clip actions -- normalises the input action to [-1, 1] range.
        Normalize Observation -- normalises observations to have mean 0 and variance 1.
        """
        self.env = wrappers.ClipAction(self.env)
        self.env = wrappers.NormalizeObservation(self.env)
        self.norm_wrapper = self.env

    def build_convolutional(self):
        """Wraps itself in wrappers that are used for environments with continuous state space.

        When the episode starts, randomly skips up to 10 steps to ensure stochasticity.
        The environment images are scaled to width and height of 84x84; converted from RGB to gray scale and
        pixel value is scaled from [0,255] to [0,1].
        The frames are stacked, so the agent does know dynamic context.
        """
        self.env = wrappers.AtariPreprocessing(
            self.env, noop_max=10, frame_skip=1, scale_obs=True
        )
        self.env = wrappers.FrameStackObservation(self.env, 4)

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
        if self.norm_wrapper is None:
            raise ValueError(
                "Normalization wrapper not found. Ensure build_continuous() has been called."
            )
        rms = self.norm_wrapper.obs_rms
        np.savez(path, mean=rms.mean, var=rms.var, count=rms.count)

    def load_normalization_parameters(self, path):
        """Loads the observation normalization mean and variance from a file.

        Args:
            path (str): The file path from which the normalization parameters will be loaded.
        """
        if self.norm_wrapper is None:
            raise ValueError(
                "Normalization wrapper not found. Ensure build_continuous() has been called."
            )
        print("Loaded!")
        data = np.load(path)
        rms = self.norm_wrapper.obs_rms
        rms.mean = data["mean"]
        rms.var = data["var"]
        rms.count = data["count"]

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

    def get_state_shape(self) -> tuple:
        """Returns state shape"""
        return self.env.observation_space.shape

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

    def reset(self) -> np.ndarray:
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
