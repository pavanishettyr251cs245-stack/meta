from environment.core import InsuranceClaimEnvironment
from environment.schemas import ClaimAction, ClaimObservation
from openenv.core.env_server import create_app

env = InsuranceClaimEnvironment({"max_steps": 6})
app = create_app(env, ClaimAction, ClaimObservation)