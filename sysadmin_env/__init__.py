"""Sysadmin Game Environment.

An OpenEnv-compliant RL environment for training LLMs on real Linux troubleshooting.
"""

from .models import Action, Observation, State
from .environment import SysadminEnv
from .sandbox import DockerSandbox
from .client import OpenEnvSysadminClient, SysadminEnvClient
from .scenarios import (
    list_scenarios,
    get_scenario,
    get_random_scenario,
    TRAIN_SCENARIO_IDS,
    VAL_SCENARIO_IDS,
    HELDOUT_SCENARIO_IDS,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Observation",
    "State",
    "SysadminEnv",
    "SysadminEnvClient",
    "OpenEnvSysadminClient",
    "DockerSandbox",
    "list_scenarios",
    "get_scenario",
    "get_random_scenario",
    "TRAIN_SCENARIO_IDS",
    "VAL_SCENARIO_IDS",
    "HELDOUT_SCENARIO_IDS",
]
