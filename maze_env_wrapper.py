import gymnasium as gym
from typing import Optional
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space


class EnvCompatibility(gym.Env):
    """A wrapper converting gym.Env from old gym API to the new one.

    "Old API" refers to step() method returning (observation, reward, done, info),
    and reset() only retuning the observation.
    "New API" refers to step() method returning (observation, reward, terminated,
    truncated, info) and reset() returning (observation, info).

    Known limitations:
    - Environments that use `self.np_random` might not work as expected.
    """

    def __init__(self, old_env, render_mode: Optional[str] = None):
        """A wrapper which converts old-style envs to valid modern envs."""
        super().__init__()

        self.metadata = getattr(old_env, "metadata", {"render_modes": []})
        self.render_mode = render_mode
        self.reward_range = getattr(old_env, "reward_range", None)
        self.spec = getattr(old_env, "spec", None)
        self.env = old_env

        self.observation_space = convert_old_gym_space_to_gymnasium_space(
            old_env.observation_space
        )
        self.action_space = convert_old_gym_space_to_gymnasium_space(
            old_env.action_space
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> list:
        # Use old `seed()` method.
        if seed is not None:
            self.env.seed(seed)
        # Options are ignored

        if self.render_mode == "human":
            self.render()

        obs = self.env.reset()
        infos = {}
        return obs, infos

    def step(self, action) -> list:
        obs, reward, terminated, info = self.env.step(action)

        # Truncated should always be False by default.
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self):
        # Use the old `render()` API, where we have to pass in the mode to each call.
        return self.env.render(mode=self.render_mode)

    def close(self):
        self.env.close()
