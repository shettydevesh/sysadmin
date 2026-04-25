"""Scenario registry for the Sysadmin Game environment."""

import random
from typing import Type

from .base import BaseScenario
from .disk_full import DiskFullScenario
from .disk_full_alt import DiskFullAltScenario
from .nginx_syntax import NginxSyntaxScenario
from .nginx_unknown import NginxUnknownScenario
from .ownership import OwnershipScenario
from .port_bound import PortBoundScenario
from .runaway_cpu import RunawayCpuScenario
from .expired_cert import ExpiredCertScenario
from .stale_pid import StalePidScenario
from .venv_broken import VenvBrokenScenario

# All available scenarios
ALL_SCENARIOS: dict[str, Type[BaseScenario]] = {
    "disk_full": DiskFullScenario,
    "disk_full_alt": DiskFullAltScenario,
    "nginx_syntax": NginxSyntaxScenario,
    "nginx_unknown": NginxUnknownScenario,
    "ownership": OwnershipScenario,
    "port_bound": PortBoundScenario,
    "runaway_cpu": RunawayCpuScenario,
    "expired_cert": ExpiredCertScenario,
    "stale_pid": StalePidScenario,
    "venv_broken": VenvBrokenScenario,
}

# Scenarios for training (all of them)
TRAIN_SCENARIO_IDS = list(ALL_SCENARIOS.keys())

# Scenarios for validation (one representative from each)
VAL_SCENARIO_IDS = [
    "ownership",  # Permissions
    "nginx_syntax",  # Service config
    "disk_full",  # Disk
    "port_bound",  # Network
    "runaway_cpu",  # Process
]


def get_scenario(scenario_id: str) -> BaseScenario:
    """Get a scenario instance by ID.

    Args:
        scenario_id: The scenario identifier

    Returns:
        An instance of the scenario

    Raises:
        KeyError: If scenario_id is not found
    """
    if scenario_id not in ALL_SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_id}")
    return ALL_SCENARIOS[scenario_id]()


def get_random_scenario(scenario_ids: list[str] = None) -> BaseScenario:
    """Get a random scenario instance.

    Args:
        scenario_ids: Optional list of scenario IDs to choose from

    Returns:
        An instance of a randomly selected scenario
    """
    ids = scenario_ids or TRAIN_SCENARIO_IDS
    scenario_id = random.choice(ids)
    return get_scenario(scenario_id)


def list_scenarios() -> list[str]:
    """List all available scenario IDs."""
    return list(ALL_SCENARIOS.keys())
