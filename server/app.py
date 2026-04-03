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