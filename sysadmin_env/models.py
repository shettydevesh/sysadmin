"""Typed models for the Sysadmin Game environment.

The classes intentionally use Pydantic instead of dataclasses because OpenEnv
manifests and clients discover action/observation schemas from model fields.
When ``openenv-core`` is installed these models also inherit the OpenEnv base
types; locally they remain ordinary Pydantic models so development does not
require the full OpenEnv toolchain.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:  # pragma: no cover - exercised only when openenv-core is installed.
    from openenv.core.env_server.types import (
        Action as _OpenEnvAction,
        Observation as _OpenEnvObservation,
        State as _OpenEnvState,
    )
except Exception:  # pragma: no cover - fallback is covered by local tests/imports.
    class _OpenEnvAction(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

    class _OpenEnvObservation(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

    class _OpenEnvState(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)


class Action(_OpenEnvAction):
    """A shell command issued by the agent."""

    command: str = Field(..., description="Single bash command to execute in the sandbox")


class Observation(_OpenEnvObservation):
    """Observation returned after reset or step."""

    output: str = Field(..., description="Command output or initial user complaint")
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: float = Field(default=0.0, description="Reward assigned to the last action")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Episode metadata")


class State(_OpenEnvState):
    """Internal and client-visible episode state."""

    episode_id: str = Field(default="", description="Unique episode/session identifier")
    step_count: int = Field(default=0, description="OpenEnv-compatible step count")
    scenario_id: str = Field(default="", description="Scenario identifier")
    command_count: int = Field(default=0, description="Number of shell commands executed")
    diagnostics_used: set[str] = Field(default_factory=set, description="Diagnostic tools seen")
    start_time: float = Field(default=0.0, description="Unix timestamp for episode start")
    container_id: str = Field(default="", description="Sandbox container identifier")
    total_reward: float = Field(default=0.0, description="Cumulative episode reward")
    elapsed_time: float = Field(default=0.0, description="Seconds since reset")
    done: bool = Field(default=False, description="Whether this episode has terminated")
