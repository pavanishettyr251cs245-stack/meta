from environment.core import InsuranceClaimEnvironment
from environment.schemas import ClaimAction, ClaimObservation
from openenv.core.env_server import create_app

env = InsuranceClaimEnvironment({"max_steps": 6})
app = create_app(env, ClaimAction, ClaimObservation)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()