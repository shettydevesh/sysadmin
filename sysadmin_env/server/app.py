"""FastAPI server for the Sysadmin Game environment."""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..environment import SysadminEnv
from ..scenarios import list_scenarios, TRAIN_SCENARIO_IDS, VAL_SCENARIO_IDS


app = FastAPI(
    title="Sysadmin Game Environment",
    description="OpenEnv-compliant RL environment for Linux troubleshooting",
    version="0.1.0",
)

# Global environment instance
env: Optional[SysadminEnv] = None


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    scenario_id: Optional[str] = None
    scenario_ids: Optional[list[str]] = None


class ResetResponse(BaseModel):
    output: str
    done: bool
    reward: float
    metadata: dict


class StepRequest(BaseModel):
    command: str


class StepResponse(BaseModel):
    output: str
    done: bool
    reward: float
    metadata: dict


class StateResponse(BaseModel):
    scenario_id: str
    command_count: int
    diagnostics_used: list[str]
    elapsed_time: float
    total_reward: float
    container_id: str


class HealthResponse(BaseModel):
    status: str
    env_initialized: bool


class ScenariosResponse(BaseModel):
    all: list[str]
    train: list[str]
    val: list[str]


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = None):
    """Reset the environment to a new episode."""
    global env

    request = request or ResetRequest()

    # Create or reconfigure environment
    scenario_ids = request.scenario_ids or TRAIN_SCENARIO_IDS
    env = SysadminEnv(scenario_ids=scenario_ids)

    # Reset environment
    obs = env.reset(seed=request.seed, scenario_id=request.scenario_id)

    return ResetResponse(
        output=obs.output,
        done=obs.done,
        reward=obs.reward,
        metadata=obs.metadata,
    )


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """Execute a command in the environment."""
    global env

    if env is None or env.state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )

    from ..models import Action
    action = Action(command=request.command)
    obs = env.step(action)

    return StepResponse(
        output=obs.output,
        done=obs.done,
        reward=obs.reward,
        metadata=obs.metadata,
    )


@app.get("/state", response_model=StateResponse)
async def get_state():
    """Get the current environment state."""
    global env

    if env is None or env.state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )

    import time
    state = env.state
    elapsed = time.time() - state.start_time

    return StateResponse(
        scenario_id=state.scenario_id,
        command_count=state.command_count,
        diagnostics_used=list(state.diagnostics_used),
        elapsed_time=elapsed,
        total_reward=state.total_reward,
        container_id=state.container_id,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    global env
    return HealthResponse(
        status="ok",
        env_initialized=env is not None and env.state is not None,
    )


@app.get("/scenarios", response_model=ScenariosResponse)
async def scenarios():
    """List available scenarios."""
    return ScenariosResponse(
        all=list_scenarios(),
        train=TRAIN_SCENARIO_IDS,
        val=VAL_SCENARIO_IDS,
    )


@app.on_event("shutdown")
async def shutdown():
    """Clean up on shutdown."""
    global env
    if env:
        env.close()


def main():
    """Entry point for the server."""
    uvicorn.run(
        "sysadmin_env.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
