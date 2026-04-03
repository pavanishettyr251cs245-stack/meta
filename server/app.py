from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
from environment.core import InsuranceClaimEnvironment
from environment.schemas import ClaimAction, ReasoningOutput

app = FastAPI(title="Insurance Claim Validation Environment")
env = InsuranceClaimEnvironment({"max_steps": 6})

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: ClaimAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
@app.get("/")
def root():
    return {
        "name": "Insurance Claim Validation Environment",
        "description": "OpenEnv-compatible RL environment for insurance claim validation",
        "endpoints": {
            "POST /reset": "Reset environment and get initial observation",
            "POST /step": "Take a step with a ClaimAction",
            "GET /state": "Get current environment state"
        }
    }