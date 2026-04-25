"""Optional OpenEnv-native ASGI app.

The default ``server.app`` keeps a simple HTTP API for demos and curl smoke
tests. This module is for OpenEnv tooling that expects ``create_app`` from the
latest OpenEnv package.
"""

import os

from ..environment import SysadminEnv
from ..models import Action, Observation

try:
    from openenv.core.env_server import create_app
except ImportError as exc:  # pragma: no cover - depends on optional package.
    raise RuntimeError("Install openenv-core to use the OpenEnv-native app") from exc


def create_sysadmin_environment() -> SysadminEnv:
    """Factory used so each OpenEnv session receives an isolated sandbox."""
    return SysadminEnv()


app = create_app(
    create_sysadmin_environment,
    Action,
    Observation,
    env_name="sysadmin-game",
    max_concurrent_envs=int(os.getenv("SYSADMIN_MAX_CONCURRENT_ENVS", "8")),
)
