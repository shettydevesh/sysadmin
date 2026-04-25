"""FastAPI server for the Sysadmin Game environment.

This server keeps a clean client/server boundary: clients talk only to the HTTP
API and never import scenario or sandbox internals. Multiple episode IDs are
supported so evaluation and RL rollouts do not stomp on one global environment.
"""

import uvicorn
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from ..environment import SysadminEnv
from ..scenarios import list_scenarios, TRAIN_SCENARIO_IDS, VAL_SCENARIO_IDS


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Clean up active sandboxes when the server exits."""
    try:
        yield
    finally:
        for env in list(envs.values()):
            env.close()
        envs.clear()


app = FastAPI(
    title="Sysadmin Game Environment",
    description="OpenEnv-compliant RL environment for Linux troubleshooting",
    version="0.1.0",
    lifespan=lifespan,
)

# Active episode registry. Each reset creates a fresh sandboxed environment.
envs: dict[str, SysadminEnv] = {}
latest_episode_id: Optional[str] = None


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
    episode_id: Optional[str] = None


class StepResponse(BaseModel):
    output: str
    done: bool
    reward: float
    metadata: dict


class StateResponse(BaseModel):
    episode_id: str
    scenario_id: str
    step_count: int
    command_count: int
    diagnostics_used: list[str]
    elapsed_time: float
    total_reward: float
    container_id: str


class HealthResponse(BaseModel):
    status: str
    env_initialized: bool
    active_episodes: int


class ScenariosResponse(BaseModel):
    all: list[str]
    train: list[str]
    val: list[str]


@app.get("/")
async def root():
    """Basic landing response for Spaces and smoke tests."""
    return {
        "name": "sysadmin-game",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "scenarios": "/scenarios",
    }


def _resolve_episode_id(episode_id: Optional[str] = None) -> str:
    """Resolve explicit or latest episode ID, raising an HTTP error if missing."""
    resolved = episode_id or latest_episode_id
    if not resolved or resolved not in envs:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first or pass a valid episode_id.",
        )
    return resolved


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = None):
    """Reset the environment to a new episode."""
    global latest_episode_id

    request = request or ResetRequest()

    # Create or reconfigure environment
    scenario_ids = request.scenario_ids or TRAIN_SCENARIO_IDS
    env = SysadminEnv(scenario_ids=scenario_ids)

    # Reset environment
    obs = env.reset(seed=request.seed, scenario_id=request.scenario_id)
    episode_id = obs.metadata["episode_id"]
    envs[episode_id] = env
    latest_episode_id = episode_id

    return ResetResponse(
        output=obs.output,
        done=obs.done,
        reward=obs.reward,
        metadata=obs.metadata,
    )


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """Execute a command in the environment."""
    episode_id = _resolve_episode_id(request.episode_id)
    env = envs[episode_id]

    if env.state and env.state.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode.",
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
async def get_state(episode_id: Optional[str] = Query(default=None)):
    """Get the current environment state."""
    resolved_episode_id = _resolve_episode_id(episode_id)
    env = envs[resolved_episode_id]

    import time
    state = env.state
    elapsed = time.time() - state.start_time

    return StateResponse(
        episode_id=state.episode_id,
        scenario_id=state.scenario_id,
        step_count=state.step_count,
        command_count=state.command_count,
        diagnostics_used=list(state.diagnostics_used),
        elapsed_time=elapsed,
        total_reward=state.total_reward,
        container_id=state.container_id,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        env_initialized=bool(envs),
        active_episodes=len(envs),
    )


@app.get("/scenarios", response_model=ScenariosResponse)
async def scenarios():
    """List available scenarios."""
    return ScenariosResponse(
        all=list_scenarios(),
        train=TRAIN_SCENARIO_IDS,
        val=VAL_SCENARIO_IDS,
    )


@app.delete("/episodes/{episode_id}")
async def close_episode(episode_id: str):
    """Close and remove a single episode sandbox."""
    env = envs.pop(episode_id, None)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown episode_id: {episode_id}")
    env.close()
    return {"closed": episode_id}


def main():
    """Entry point for the server."""
    uvicorn.run(
        "sysadmin_env.server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
