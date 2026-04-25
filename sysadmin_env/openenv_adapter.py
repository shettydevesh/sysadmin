"""TRL/OpenEnv adapter for live sysadmin rollouts.

TRL's modern ``GRPOTrainer`` can manage multi-turn environment interaction when
given an ``environment_factory``. This adapter exposes a single meaningful tool,
``run_shell``, while keeping reward and done state on the environment instance
for the reward function.
"""

from __future__ import annotations

import os
from typing import Optional

from .client import SysadminEnvClient
from .environment import SysadminEnv
from .models import Action


SYSADMIN_RL_PROMPT = """You are an expert SRE agent inside a Linux troubleshooting environment.

Use the run_shell tool to diagnose and fix the incident. Work like a careful
operator: inspect status and logs first, make the smallest safe fix, then verify
the service or system is healthy. Stop once the issue is fixed.
"""


class SysadminToolEnv:
    """Tool-style wrapper used by TRL ``GRPOTrainer(environment_factory=...)``."""

    def __init__(self):
        self.env_url = os.getenv("SYSADMIN_ENV_URL")
        self.reward = 0.0
        self.total_reward = 0.0
        self.done = False
        self.last_metadata: dict = {}
        self._local_env: Optional[SysadminEnv] = None
        self._client: Optional[SysadminEnvClient] = None

    def reset(self, **kwargs) -> str | None:
        """Start a new incident episode.

        Dataset columns such as ``scenario_id`` and ``seed`` are passed through
        by TRL as keyword arguments.
        """
        self.reward = 0.0
        self.total_reward = 0.0
        self.done = False
        scenario_id = kwargs.get("scenario_id")
        seed = kwargs.get("seed")

        if self.env_url:
            self._client = SysadminEnvClient(base_url=self.env_url)
            obs = self._client.reset(seed=seed, scenario_id=scenario_id)
        else:
            self._local_env = SysadminEnv()
            obs = self._local_env.reset(seed=seed, scenario_id=scenario_id)

        self.last_metadata = obs.metadata
        return obs.output

    def run_shell(self, command: str) -> str:
        """
        Execute one safe bash command in the broken Linux sandbox.

        Args:
            command: A single bash command to run. Use diagnostics before fixes
                and avoid destructive commands such as disk wipes or recursive
                deletes outside temporary directories.

        Returns:
            The command output plus compact reward and episode status metadata.
        """
        if self.done:
            raise ValueError("Episode is already done.")

        action = Action(command=command)
        if self._client is not None:
            obs = self._client.step(action)
        elif self._local_env is not None:
            obs = self._local_env.step(action)
        else:
            raise ValueError("Environment has not been reset.")

        self.reward = obs.reward
        self.total_reward = obs.metadata.get("total_reward", self.total_reward + obs.reward)
        self.done = obs.done
        self.last_metadata = obs.metadata

        status = (
            f"\n[reward={obs.reward:+.2f} total={self.total_reward:+.2f} "
            f"done={obs.done} fixed={obs.metadata.get('fixed', False)}]"
        )
        return obs.output + status

    def _close(self) -> None:
        """Clean up local or remote episode resources."""
        if self._client is not None:
            self._client.close()
        if self._local_env is not None:
            self._local_env.close()


def sysadmin_reward(environments, **kwargs) -> list[float]:
    """Reward function for TRL GRPOTrainer."""
    return [env.total_reward for env in environments]
