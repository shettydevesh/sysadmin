"""Core environment implementation."""

import time
import random
from uuid import uuid4
from typing import Optional

from .models import Action, Observation, State
from .sandbox import DockerSandbox
from .reward import calculate_reward
from .blocklist import is_destructive
from .scenarios import get_scenario, get_random_scenario, TRAIN_SCENARIO_IDS


# Episode constraints
MAX_COMMANDS = 25
MAX_TIME_SECONDS = 60


class SysadminEnv:
    """OpenEnv-compliant environment for sysadmin training.

    This environment simulates broken Linux systems that an agent must diagnose
    and fix using shell commands.
    """

    def __init__(self, scenario_ids: Optional[list[str]] = None):
        """Initialize the environment.

        Args:
            scenario_ids: List of scenario IDs to use. Defaults to all training scenarios.
        """
        self.scenario_ids = scenario_ids or TRAIN_SCENARIO_IDS
        self.sandbox: Optional[DockerSandbox] = None
        self.scenario = None
        self._state: Optional[State] = None

    def reset(self, seed: Optional[int] = None, scenario_id: Optional[str] = None) -> Observation:
        """Reset the environment to a new episode.

        Args:
            seed: Random seed for reproducibility
            scenario_id: Specific scenario to use (optional)

        Returns:
            Initial observation with the user complaint
        """
        if seed is not None:
            random.seed(seed)

        # Destroy old container
        if self.sandbox:
            self.sandbox.destroy()

        # Pick scenario
        if scenario_id:
            self.scenario = get_scenario(scenario_id)
        else:
            self.scenario = get_random_scenario(self.scenario_ids)

        episode_id = str(uuid4())

        # Create new container
        self.sandbox = DockerSandbox()
        container_id = self.sandbox.create(name_suffix=f"{self.scenario.id}_{episode_id[:8]}")

        # Initialize state
        self._state = State(
            episode_id=episode_id,
            step_count=0,
            scenario_id=self.scenario.id,
            command_count=0,
            diagnostics_used=set(),
            start_time=time.time(),
            container_id=container_id,
            total_reward=0.0,
        )

        # Break the system
        self.scenario.break_system(self.sandbox)

        # Start timer after container is ready — excludes creation/break overhead
        self._state.start_time = time.time()

        # Return initial observation with user complaint
        return Observation(
            output=self.scenario.get_complaint(),
            done=False,
            reward=0.0,
            metadata={
                "episode_id": self._state.episode_id,
                "scenario_id": self.scenario.id,
                "category": self.scenario.category,
                "command_count": 0,
                "step_count": 0,
                "is_initial": True,
            },
        )

    def step(self, action: Action) -> Observation:
        """Execute an action (shell command) in the environment.

        Args:
            action: The action to take (command to execute)

        Returns:
            Observation with command output, reward, and done status
        """
        if not self.sandbox or not self._state:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new episode.")

        command = action.command if isinstance(action, Action) else action.get("command", "")

        # Check for destructive command
        destructive = is_destructive(command)

        if destructive:
            # Terminate episode immediately
            self._state.command_count += 1
            self._state.step_count += 1
            reward, _ = calculate_reward(
                command=command,
                fixed=False,
                diagnostics_used=self._state.diagnostics_used,
                is_destructive=True,
            )
            self._state.total_reward += reward
            self._state.done = True

            return Observation(
                output="BLOCKED: Destructive command detected. Episode terminated.",
                done=True,
                reward=reward,
                metadata={
                    "episode_id": self._state.episode_id,
                    "scenario_id": self._state.scenario_id,
                    "command_count": self._state.command_count,
                    "step_count": self._state.step_count,
                    "total_reward": self._state.total_reward,
                    "termination_reason": "destructive_command",
                    "fixed": False,
                },
            )

        # Execute command
        output = self.sandbox.exec(command)
        self._state.command_count += 1
        self._state.step_count += 1

        # Check if fixed
        try:
            fixed = self.scenario.check_fixed(self.sandbox)
        except Exception:
            fixed = False

        # Calculate reward
        reward, new_diagnostics = calculate_reward(
            command=command,
            fixed=fixed,
            diagnostics_used=self._state.diagnostics_used,
            is_destructive=False,
        )
        self._state.diagnostics_used = new_diagnostics
        self._state.total_reward += reward

        # Check termination conditions
        elapsed = time.time() - self._state.start_time
        self._state.elapsed_time = elapsed
        done = (
            fixed
            or self._state.command_count >= MAX_COMMANDS
            or elapsed >= MAX_TIME_SECONDS
        )

        termination_reason = None
        if done:
            self._state.done = True
            if fixed:
                termination_reason = "fixed"
            elif self._state.command_count >= MAX_COMMANDS:
                termination_reason = "max_commands"
            elif elapsed >= MAX_TIME_SECONDS:
                termination_reason = "timeout"

        return Observation(
            output=output,
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "scenario_id": self._state.scenario_id,
                "command_count": self._state.command_count,
                "step_count": self._state.step_count,
                "total_reward": self._state.total_reward,
                "fixed": fixed,
                "elapsed_time": elapsed,
                "termination_reason": termination_reason,
            },
        )

    @property
    def state(self) -> Optional[State]:
        """Get the current environment state."""
        return self._state

    def close(self):
        """Clean up resources."""
        if self.sandbox:
            self.sandbox.destroy()
            self.sandbox = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
