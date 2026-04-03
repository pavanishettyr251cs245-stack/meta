from uuid import uuid4
from openenv.core.env_server.types import State
from environment.core import InsuranceClaimEnvironment
from environment.schemas import ClaimAction, ClaimObservation

try:
    from ..models import MyEnv2Action, MyEnv2Observation
except ImportError:
    from models import MyEnv2Action, MyEnv2Observation

class MyEnv2Environment:
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._env = InsuranceClaimEnvironment({"max_steps": 6})
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        obs = self._env.reset()
        return obs

    def step(self, action):
        self._state.step_count += 1
        obs, reward, done, info = self._env.step(action)
        return obs

    @property
    def state(self):
        return self._state