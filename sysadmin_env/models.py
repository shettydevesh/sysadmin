"""Data models for the Sysadmin Game environment."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """An action in the environment (a shell command to execute)."""
    command: str


@dataclass
class Observation:
    """Observation returned after taking an action."""
    output: str
    done: bool = False
    reward: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class State:
    """Internal state of the environment."""
    scenario_id: str
    command_count: int
    diagnostics_used: set = field(default_factory=set)
    start_time: float = 0.0
    container_id: str = ""
    total_reward: float = 0.0
