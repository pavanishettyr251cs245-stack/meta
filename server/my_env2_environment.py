from uuid import uuid4
from environment.core import InsuranceClaimEnvironment

class MyEnv2Environment:
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._env = InsuranceClaimEnvironment({"max_steps": 6})
        self._episode_id = str(uuid4())
        self._step_count = 0

    def reset(self):
        self._episode_id = str(uuid4())
        self._step_count = 0
        return self._env.reset()

    def step(self, action):
        self._step_count += 1
        obs, reward, done, info = self._env.step(action)
        return obs

    @property
    def state(self):
        return {"episode_id": self._episode_id, "step_count": self._step_count}