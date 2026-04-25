"""Clients for the Sysadmin Game environment server."""

from __future__ import annotations

import json
from typing import Any, Optional
from urllib import parse, request

from .models import Action, Observation, State


def _json_request(
    method: str,
    url: str,
    payload: Optional[dict[str, Any]] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Send a JSON request and return a decoded JSON object."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


class SysadminEnvClient:
    """Small HTTP client for local servers and Hugging Face Space APIs."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.episode_id: Optional[str] = None

    def reset(
        self,
        seed: Optional[int] = None,
        scenario_id: Optional[str] = None,
        scenario_ids: Optional[list[str]] = None,
    ) -> Observation:
        """Start a fresh episode and remember its episode ID."""
        payload = {
            "seed": seed,
            "scenario_id": scenario_id,
            "scenario_ids": scenario_ids,
        }
        data = _json_request("POST", f"{self.base_url}/reset", payload, self.timeout)
        obs = Observation(**data)
        self.episode_id = obs.metadata.get("episode_id")
        return obs

    def step(self, action: Action | str) -> Observation:
        """Execute one shell command in the current episode."""
        command = action.command if isinstance(action, Action) else str(action)
        payload = {"command": command, "episode_id": self.episode_id}
        data = _json_request("POST", f"{self.base_url}/step", payload, self.timeout)
        return Observation(**data)

    def state(self) -> State:
        """Fetch server-side state for the current episode."""
        query = ""
        if self.episode_id:
            query = "?" + parse.urlencode({"episode_id": self.episode_id})
        data = _json_request("GET", f"{self.base_url}/state{query}", timeout=self.timeout)
        return State(**data)

    def health(self) -> dict[str, Any]:
        """Fetch server health."""
        return _json_request("GET", f"{self.base_url}/health", timeout=self.timeout)

    def scenarios(self) -> dict[str, list[str]]:
        """Fetch available scenario IDs."""
        return _json_request("GET", f"{self.base_url}/scenarios", timeout=self.timeout)

    def close(self) -> None:
        """Close the current episode on the server."""
        if not self.episode_id:
            return
        try:
            _json_request(
                "DELETE",
                f"{self.base_url}/episodes/{self.episode_id}",
                timeout=self.timeout,
            )
        finally:
            self.episode_id = None


try:  # pragma: no cover - depends on optional OpenEnv installation.
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient as _OpenEnvClient
except Exception:  # pragma: no cover - local fallback path.
    StepResult = None
    _OpenEnvClient = None


if _OpenEnvClient is not None:  # pragma: no cover - requires openenv-core.

    class OpenEnvSysadminClient(_OpenEnvClient[Action, Observation, State]):
        """OpenEnv WebSocket client used by AutoEnv/openenv tooling."""

        def _step_payload(self, action: Action) -> dict[str, Any]:
            return {"command": action.command}

        def _parse_result(self, payload: dict[str, Any]) -> StepResult[Observation]:
            obs_data = payload.get("observation", payload)
            observation = Observation(**obs_data)
            return StepResult(
                observation=observation,
                reward=payload.get("reward", observation.reward),
                done=payload.get("done", observation.done),
            )

        def _parse_state(self, payload: dict[str, Any]) -> State:
            return State(**payload)

else:
    OpenEnvSysadminClient = SysadminEnvClient
