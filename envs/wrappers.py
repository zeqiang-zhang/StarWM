import datetime
import gymnasium as gym
import numpy as np
import uuid


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, truncated, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, truncated, info
    def reset(self,seed=None, options=None):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class ActionDelay(gym.Wrapper):
    """Delay the agent's action by k steps (Appendix F.5): the wrapped env
    applies a_{t-k}, so a_t only takes visual effect k steps later. Priming
    actions a_{<0} are one-hot action 0. The replay still stores the issued a_t;
    StarWM's ``inv_delay=k`` shifts the inverse-dynamics window accordingly."""

    def __init__(self, env, delay):
        super().__init__(env)
        self._delay = int(delay)
        adim = self.action_space.shape[0]
        self._noop = np.zeros(adim, dtype=np.float32)
        self._noop[0] = 1.0
        self._buffer = []

    def reset(self, *args, **kwargs):
        self._buffer = [self._noop.copy() for _ in range(self._delay)]
        try:
            return self.env.reset(*args, **kwargs)
        except TypeError:
            return self.env.reset()

    def step(self, action):
        self._buffer.append(action)
        applied = self._buffer.pop(0)      # a_{t-k}
        return self.env.step(applied)


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()
