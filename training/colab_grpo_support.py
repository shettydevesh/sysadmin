"""Helpers for the Colab GRPO notebook.

These utilities keep the notebook lighter and more robust:
- resilient HTTP client with episode tracking and retries
- smaller, Colab-friendly rollout helpers
- judge-friendly logs and summary generation
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:  # pragma: no cover - only used inside notebooks
    from IPython.display import HTML, display
except Exception:  # pragma: no cover - safe fallback outside notebooks
    HTML = None
    display = None


DEFAULT_SYSTEM_PROMPT = """You are an expert SRE agent that diagnoses and fixes Linux system issues.

When given a problem:
1. Think step-by-step about the diagnosis in <think> tags
2. Run ONE command at a time in <bash> tags
3. Analyze the output before running the next command
4. Prefer diagnostics before making changes
5. Stop once the issue is fixed

Example response:
<think>
The user reports nginx won't start. First, let me check the service status.
</think>
<bash>systemctl status nginx --no-pager -l</bash>"""

DEFAULT_TRAIN_SCENARIOS = [
    "disk_full",
    "nginx_syntax",
    "ownership",
    "port_bound",
    "runaway_cpu",
]

FALLBACK_COMMAND_RE = re.compile(
    r"^(?:\$+\s*)?(?:"
    r"systemctl|journalctl|nginx|apache2ctl|ss|netstat|lsof|ps|top|df|du|find|grep|awk|sed|perl|python3?|"
    r"cat|ls|stat|file|id|whoami|chmod|chown|chgrp|rm|truncate|kill|pkill|killall|service|curl|nohup"
    r")\b"
)
INTERACTIVE_COMMAND_PATTERNS = [
    re.compile(r"(^|[;&|]\s*)(nano|vi|vim|view)\b", re.IGNORECASE),
    re.compile(r"(^|[;&|]\s*)(less|more|man|watch|tail\s+-f)\b", re.IGNORECASE),
    re.compile(r"(^|[;&|]\s*)(top|htop)\b(?![^\n|;&]*-b)", re.IGNORECASE),
]
IRRELEVANT_COMMAND_PATTERNS = [
    re.compile(r"(^|[;&|]\s*)apt(-get)?\s+update\b", re.IGNORECASE),
    re.compile(r"(^|[;&|]\s*)apt(-get)?\s+install\b", re.IGNORECASE),
    re.compile(r"(^|[;&|]\s*)pip3?\s+install\b", re.IGNORECASE),
    re.compile(r"(^|[;&|]\s*)clear\b", re.IGNORECASE),
]
PLACEHOLDER_PATTERN = re.compile(r"<[A-Z][A-Z0-9_ -]*>")


ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


@dataclass
class Observation:
    """Environment observation payload."""

    output: str
    done: bool
    reward: float
    metadata: dict[str, Any]


class OpenEnvClient:
    """Resilient HTTP client for the remote sysadmin environment."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 45.0,
        max_retries: int = 5,
        backoff_factor: float = 1.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.episode_id: Optional[str] = None

        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            allowed_methods=frozenset(["GET", "POST", "DELETE"]),
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=backoff_factor,
            raise_on_status=False,
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def scenarios(self) -> dict[str, Any]:
        return self._request("GET", "/scenarios")

    def reset(
        self,
        scenario_id: Optional[str] = None,
        seed: Optional[int] = None,
        scenario_ids: Optional[list[str]] = None,
    ) -> Observation:
        payload: dict[str, Any] = {}
        if scenario_id:
            payload["scenario_id"] = scenario_id
        if seed is not None:
            payload["seed"] = seed
        if scenario_ids is not None:
            payload["scenario_ids"] = scenario_ids

        data = self._request("POST", "/reset", json=payload)
        obs = Observation(**data)
        self.episode_id = obs.metadata.get("episode_id")
        return obs

    def step(self, command: str) -> Observation:
        payload = {
            "command": command,
            "episode_id": self.episode_id,
        }
        data = self._request("POST", "/step", json=payload)
        return Observation(**data)

    def close(self) -> None:
        if not self.episode_id:
            return
        try:
            self._request("DELETE", f"/episodes/{self.episode_id}")
        except Exception:
            pass
        finally:
            self.episode_id = None


def parse_response(response: str) -> tuple[str | None, str | None]:
    """Extract thinking and a bash command from model output."""
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    bash_match = re.search(r"<bash>(.*?)</bash>", response, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else None
    command = bash_match.group(1).strip() if bash_match else extract_fallback_command(response)
    return thinking, command


def extract_fallback_command(response: str) -> str | None:
    """Recover a plain shell command when the model forgets <bash> tags."""
    fenced_match = re.search(r"```(?:bash|sh)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip().splitlines()
        for line in candidate:
            line = line.strip()
            if line and FALLBACK_COMMAND_RE.match(line):
                return line.lstrip("$").strip()

    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if FALLBACK_COMMAND_RE.match(line):
            return line.lstrip("$").strip()
    return None


def truncate_text(text: str, limit: int) -> str:
    """Trim long environment outputs to keep context under control."""
    text = text.strip()
    if len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    return text[:half] + "\n... [truncated for context] ...\n" + text[-half:]


def normalize_command(command: str | None) -> str | None:
    """Normalize commands to reduce wasted turns on notebook-only quirks."""
    if not command:
        return command
    command = command.strip().strip("`")
    command = re.sub(r"^\$+\s*", "", command)
    if command.startswith("sudo "):
        command = command[5:].strip()
    command = re.sub(r"^\s*export\s+TERM=[^;]+;\s*", "", command)
    command = re.sub(r"\s+", " ", command).strip()
    return command


def canonicalize_command(command: str | None) -> str | None:
    """Collapse whitespace and notebook wrappers for loop detection."""
    if not command:
        return None
    command = normalize_command(command)
    if not command:
        return None
    return command.lower()


def classify_blocked_command(command: str | None) -> tuple[str, str, str] | None:
    """Return (reason, output, penalty_key) for notebook-side blocked commands."""
    if not command:
        return None
    if PLACEHOLDER_PATTERN.search(command):
        return (
            "placeholder_command",
            "Blocked locally because the command still contains a placeholder like <PID> instead of a real value.",
            "placeholder_command_penalty",
        )
    for pattern in INTERACTIVE_COMMAND_PATTERNS:
        if pattern.search(command):
            return (
                "interactive_command",
                "Blocked locally because interactive editors or live TUI commands waste turns in this environment. "
                "Use a non-interactive command instead.",
                "interactive_command_penalty",
            )
    for pattern in IRRELEVANT_COMMAND_PATTERNS:
        if pattern.search(command):
            return (
                "irrelevant_command",
                "Blocked locally because package installation or screen-control commands are not part of the repair path here.",
                "irrelevant_command_penalty",
            )
    return None


def scenario_hint(complaint: str, scenario_id: str | None = None) -> str:
    """Return a concise tactic hint based on the complaint/scenario."""
    text = (complaint or "").lower()
    sid = (scenario_id or "").lower()

    if "disk" in text or "space left" in text or "disk_full" in sid:
        return (
            "Disk incidents: after confirming the filesystem is full, move quickly to a large file under "
            "/var/log using du or find, truncate or remove the specific culprit, then verify free space. "
            "Target the bad file directly instead of deleting broad directories."
        )
    if "address already in use" in text or "port" in text or "port_bound" in sid:
        return (
            "Port conflicts: find the current listener with ss or lsof, stop the conflicting service, "
            "then start nginx and verify it is listening on port 80. Do not install packages."
        )
    if "config" in text or "syntax" in text or "directive" in text or "nginx" in text:
        return (
            "Config failures: inspect service status, run the service's config test, fix the exact bad line, "
            "then validate again before restarting. Use sed, perl, python, cat, or tee for non-interactive edits."
        )
    if "cpu" in text or "slow" in text or "load" in text or "runaway_cpu" in sid:
        return (
            "CPU incidents: use non-interactive inspection like ps aux --sort=-%cpu or top -bn1, identify the runaway job, "
            "stop it, and verify the bad processes are gone."
        )
    if "certificate" in text or "https" in text or "cert" in text or "expired_cert" in sid:
        return (
            "TLS incidents: inspect cert dates, replace or renew the cert, reload the service, "
            "and verify the endpoint responds."
        )
    if "pid" in text or "crashed" in text or "stale_pid" in sid:
        return (
            "PID-file incidents: confirm the process is actually gone before deleting the stale pid file, "
            "then start the service and verify it."
        )
    if "venv" in text or "python app" in text or "no such file or directory" in text or "venv_broken" in sid:
        return (
            "Broken virtualenv incidents: inspect the venv python symlink, compare it to the system python, "
            "repair or recreate the venv, reinstall deps if needed, and verify imports."
        )
    if "permission" in text or "chmod" in text or "ownership" in sid:
        return (
            "Permission incidents: check ownership and mode on the exact file named in the error, "
            "restore expected ownership like root:root plus the expected mode, then restart nginx and verify."
        )
    return (
        "Do not repeat the same diagnostic unless new information appeared. After 1-2 diagnostics, "
        "choose the most likely fix, apply it, and verify."
    )


def build_runtime_system_prompt(base_prompt: str, complaint: str, scenario_id: str | None = None) -> str:
    """Specialize the system prompt for the current complaint."""
    hint = scenario_hint(complaint, scenario_id)
    return (
        f"{base_prompt}\n\n"
        "Runtime guidance for this episode:\n"
        f"- {hint}\n"
        "- Avoid repeating the same command pattern unless it answers a new question.\n"
        "- Avoid using sudo in this sandbox.\n"
        "- Never use interactive editors or pagers such as nano, vim, vi, less, more, top, or htop.\n"
        "- Do not install packages during the episode.\n"
        "- Return exactly one shell command inside <bash>...</bash> on every step.\n"
        "- Aim to diagnose, fix, and verify within the available turn budget.\n"
    )


def style(text: str, *effects: str) -> str:
    prefix = "".join(ANSI.get(effect, "") for effect in effects)
    return f"{prefix}{text}{ANSI['reset']}"


def print_banner(title: str, subtitle: str | None = None, tone: str = "cyan") -> None:
    line = "═" * 76
    print(style(line, tone))
    print(style(f" {title}", "bold", tone))
    if subtitle:
        print(style(f" {subtitle}", "dim"))
    print(style(line, tone))


def print_kv_table(pairs: list[tuple[str, Any]]) -> None:
    if not pairs:
        return
    width = max(len(label) for label, _ in pairs)
    for label, value in pairs:
        print(f"{style(label.ljust(width), 'bold')}  {value}")


def print_metric_strip(metrics: list[tuple[str, str, str]]) -> None:
    parts = []
    for label, value, tone in metrics:
        parts.append(f"{style(label, 'bold')} {style(value, tone)}")
    print("  |  ".join(parts))


def print_episode_log(episode_idx: int, trajectory: dict[str, Any]) -> None:
    tone = "green" if trajectory["fixed"] else "yellow"
    print(
        f"    {style(f'Episode {episode_idx}', 'bold')}  "
        f"scenario={style(trajectory['scenario_id'], tone)}  "
        f"reward={style(f'{trajectory['total_reward']:+.2f}', tone)}  "
        f"cmds={trajectory['num_commands']}  "
        f"end={trajectory['termination_reason']}"
    )
    trail = [cmd for cmd in trajectory["commands"] if cmd]
    if trail:
        preview = " -> ".join(trail[:4])
        print(style(f"      trail: {preview}", "dim"))


def print_failure_log(episode_idx: int, error: str) -> None:
    print(
        f"    {style(f'Episode {episode_idx}', 'bold')}  "
        f"{style('FAILED', 'red', 'bold')}  {error}"
    )


def render_notebook_status(title: str, metrics: list[tuple[str, Any]], tone: str = "#0f766e") -> None:
    if display is None or HTML is None:
        return

    rows = "".join(
        f"""
        <div style="padding:10px 14px;border-radius:12px;background:#fff;border:1px solid #d6d3d1;min-width:150px;">
          <div style="font-size:12px;color:#57534e;text-transform:uppercase;letter-spacing:0.05em;">{label}</div>
          <div style="font-size:20px;font-weight:700;color:#1c1917;margin-top:4px;">{value}</div>
        </div>
        """
        for label, value in metrics
    )

    display(
        HTML(
            f"""
            <div style="margin:12px 0 16px 0;padding:16px 18px;border-radius:18px;background:#f8fafc;border:1px solid #e2e8f0;">
              <div style="font-size:20px;font-weight:800;color:{tone};margin-bottom:12px;">{title}</div>
              <div style="display:flex;gap:12px;flex-wrap:wrap;">{rows}</div>
            </div>
            """
        )
    )


def print_config_summary(config: dict[str, Any]) -> None:
    print_banner("Run Configuration", "Colab-friendly GRPO-style setup", tone="blue")
    print_kv_table(
        [
            ("Model", config["model_name"]),
            ("SFT checkpoint", config["sft_checkpoint"]),
            ("Environment URL", config["env_url"]),
            ("Training steps", config["num_steps"]),
            ("Episodes / step", config["episodes_per_step"]),
            ("Group size", config["group_size"]),
            ("Max turns", config["max_turns"]),
            ("Max new tokens", config["max_new_tokens"]),
            ("Max seq length", config["max_seq_length"]),
            ("Context trim", f"{config['max_observation_chars']} chars"),
            ("Learning rate", config["learning_rate"]),
        ]
    )
    render_notebook_status(
        "GRPO Notebook Configuration",
        [
            ("Model", config["model_name"]),
            ("Steps", config["num_steps"]),
            ("Episodes / step", config["episodes_per_step"]),
            ("Turns", config["max_turns"]),
        ],
        tone="#1d4ed8",
    )


def run_episode_live(
    client: OpenEnvClient,
    model,
    tokenizer,
    config: dict[str, Any],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    verbose: bool = False,
    scenario_id: str | None = None,
    episode_seed: int | None = None,
) -> dict[str, Any]:
    """Run one environment episode and capture detailed logs."""
    obs = client.reset(seed=episode_seed, scenario_id=scenario_id)
    started_at = time.time()
    prompt_text = build_runtime_system_prompt(system_prompt, obs.output, obs.metadata.get("scenario_id"))

    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": truncate_text(obs.output, config["max_observation_chars"])},
    ]

    trajectory: dict[str, Any] = {
        "scenario_id": obs.metadata["scenario_id"],
        "prompts": [],
        "responses": [],
        "rewards": [],
        "commands": [],
        "events": [],
    }

    total_reward = 0.0
    final_obs = obs
    seen_commands: set[str] = set()
    repeated_command_count = 0

    try:
        for turn in range(config["max_turns"]):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            trajectory["prompts"].append(prompt)

            try:
                from unsloth import FastLanguageModel

                FastLanguageModel.for_inference(model)
            except Exception:
                pass

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=config["max_seq_length"],
            ).to(model.device)

            with config["torch"].no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            thinking, command = parse_response(response)
            trajectory["responses"].append(response)

            event = {
                "turn": turn + 1,
                "thinking_preview": (thinking or "")[:220],
                "response_preview": response[:260],
            }

            if not command:
                penalty = config.get("no_command_penalty", -0.1)
                total_reward += penalty
                trajectory["commands"].append(None)
                trajectory["rewards"].append(penalty)
                event["command"] = None
                event["reward"] = penalty
                event["termination_reason"] = "no_command"
                event["output_preview"] = "Model did not emit a <bash> command."
                trajectory["events"].append(event)
                final_obs = Observation(
                    output=event["output_preview"],
                    done=True,
                    reward=penalty,
                    metadata={"fixed": False, "termination_reason": "no_command"},
                )
                break

            command = normalize_command(command)
            trajectory["commands"].append(command)
            event["command"] = command

            blocked = classify_blocked_command(command)
            if blocked:
                reason, message, penalty_key = blocked
                penalty = config.get(penalty_key, -0.15)
                total_reward += penalty
                trajectory["rewards"].append(penalty)
                event["reward"] = penalty
                event["termination_reason"] = reason
                event["output_preview"] = message
                trajectory["events"].append(event)
                final_obs = Observation(
                    output=message,
                    done=True,
                    reward=penalty,
                    metadata={"fixed": False, "termination_reason": reason},
                )
                break

            command_key = canonicalize_command(command)
            if command_key in seen_commands:
                repeated_command_count += 1
                if repeated_command_count >= config.get("max_repeat_commands", 2):
                    penalty = config.get("repeat_command_penalty", -0.03)
                    total_reward += penalty
                    trajectory["rewards"].append(penalty)
                    event["reward"] = penalty
                    event["termination_reason"] = "repeat_command_loop"
                    event["output_preview"] = (
                        "Stopped locally because the model repeated the same command pattern too often."
                    )
                    trajectory["events"].append(event)
                    final_obs = Observation(
                        output=event["output_preview"],
                        done=True,
                        reward=penalty,
                        metadata={"fixed": False, "termination_reason": "repeat_command_loop"},
                    )
                    break
            else:
                seen_commands.add(command_key)

            final_obs = client.step(command)
            trajectory["rewards"].append(final_obs.reward)
            total_reward += final_obs.reward

            output_preview = truncate_text(final_obs.output, config["max_observation_chars"])
            event["reward"] = final_obs.reward
            event["fixed"] = final_obs.metadata.get("fixed", False)
            event["output_preview"] = output_preview[:320]
            trajectory["events"].append(event)

            if verbose:
                print(
                    f"      Turn {turn + 1}: {command} | "
                    f"reward={final_obs.reward:+.2f} fixed={final_obs.metadata.get('fixed', False)}"
                )

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "tool", "content": f"<output>\n{output_preview}\n</output>"})

            if final_obs.done:
                break

        if not final_obs.done:
            penalty = config.get("unfinished_episode_penalty", -0.05)
            total_reward += penalty
            final_obs = Observation(
                output="Episode stopped locally after hitting the notebook turn budget.",
                done=True,
                reward=penalty,
                metadata={"fixed": False, "termination_reason": "max_turns_local"},
            )
            trajectory["rewards"].append(penalty)
            trajectory["events"].append(
                {
                    "turn": len(trajectory["events"]) + 1,
                    "thinking_preview": "",
                    "response_preview": "",
                    "command": None,
                    "reward": penalty,
                    "fixed": False,
                    "termination_reason": "max_turns_local",
                    "output_preview": final_obs.output,
                }
            )
    finally:
        client.close()

    trajectory["total_reward"] = total_reward
    trajectory["fixed"] = final_obs.metadata.get("fixed", False)
    trajectory["num_commands"] = len([cmd for cmd in trajectory["commands"] if cmd])
    trajectory["termination_reason"] = final_obs.metadata.get("termination_reason", "unknown")
    trajectory["elapsed_seconds"] = round(time.time() - started_at, 2)
    return trajectory


def collect_trajectories_live(
    client_factory,
    model,
    tokenizer,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect multiple episodes and keep a failure log."""
    trajectories: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    base_seed = int(config.get("seed", 42))
    seed_stride = int(config.get("seed_stride", 97))
    episode_counter = int(config.get("_episode_counter", 0))
    train_scenarios = list(config.get("train_scenarios") or DEFAULT_TRAIN_SCENARIOS)
    schedule = config.get("scenario_schedule", "round_robin")

    for episode_idx in range(config["episodes_per_step"]):
        global_episode_idx = episode_counter + episode_idx
        episode_seed = base_seed + global_episode_idx * seed_stride
        scenario_id = None
        if schedule == "round_robin" and train_scenarios:
            scenario_id = train_scenarios[global_episode_idx % len(train_scenarios)]
        print(
            style(
                f"    planned rollout: scenario={scenario_id or 'random'} seed={episode_seed}",
                "dim",
            )
        )
        try:
            client = client_factory()
            trajectory = run_episode_live(
                client=client,
                model=model,
                tokenizer=tokenizer,
                config=config,
                verbose=False,
                scenario_id=scenario_id,
                episode_seed=episode_seed,
            )
            trajectories.append(trajectory)
            print_episode_log(episode_idx + 1, trajectory)
        except Exception as exc:
            error = {
                "episode": episode_idx + 1,
                "error": repr(exc),
            }
            failures.append(error)
            print_failure_log(episode_idx + 1, str(exc))

    config["_episode_counter"] = episode_counter + config["episodes_per_step"]
    return trajectories, failures


def compute_group_advantages(trajectories: list[dict[str, Any]], group_size: int) -> list[dict[str, Any]]:
    """Normalize rewards group-wise for a lightweight GRPO-style update."""
    if len(trajectories) < group_size:
        return []

    processed: list[dict[str, Any]] = []
    for start in range(0, len(trajectories) - group_size + 1, group_size):
        group = trajectories[start : start + group_size]
        rewards = [traj["total_reward"] for traj in group]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        std_reward = max(variance ** 0.5, 1e-6)

        for traj in group:
            raw_advantage = (traj["total_reward"] - mean_reward) / std_reward
            advantage = max(min(raw_advantage, 2.0), -2.0)
            for prompt, response, reward in zip(
                traj["prompts"],
                traj["responses"],
                traj["rewards"],
            ):
                processed.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "reward": reward,
                        "advantage": advantage,
                    }
                )
    return processed


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a JSONL log file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_step_summary(
    step_index: int,
    trajectories: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    avg_reward: float,
    fix_rate: float,
    avg_commands: float,
    avg_loss: float,
) -> dict[str, Any]:
    """Create a judge-friendly per-step summary object."""
    by_scenario: dict[str, dict[str, Any]] = {}
    termination_counts: Counter[str] = Counter()
    for traj in trajectories:
        stats = by_scenario.setdefault(
            traj["scenario_id"],
            {"attempts": 0, "fixes": 0, "avg_reward_accum": 0.0},
        )
        stats["attempts"] += 1
        stats["fixes"] += int(traj["fixed"])
        stats["avg_reward_accum"] += traj["total_reward"]
        termination_counts[traj["termination_reason"]] += 1

    for stats in by_scenario.values():
        stats["avg_reward"] = round(stats["avg_reward_accum"] / stats["attempts"], 3)
        del stats["avg_reward_accum"]

    example = trajectories[0] if trajectories else None
    return {
        "step": step_index,
        "avg_reward": round(avg_reward, 4),
        "fix_rate": round(fix_rate, 4),
        "avg_commands": round(avg_commands, 3),
        "avg_loss": round(avg_loss, 6),
        "num_successful_episodes": len(trajectories),
        "num_failed_episodes": len(failures),
        "termination_counts": dict(sorted(termination_counts.items())),
        "scenarios": by_scenario,
        "failures": failures,
        "example_episode": example,
    }


def write_judge_report(
    path: str | Path,
    config: dict[str, Any],
    history: dict[str, list[float]],
    scenario_stats: dict[str, dict[str, int]],
    step_summaries: list[dict[str, Any]],
) -> None:
    """Write a compact markdown report for judges."""
    lines = [
        "# GRPO Run Report",
        "",
        "This file is produced automatically by the Colab notebook so reviewers can follow the training story.",
        "",
        "## Configuration",
        "",
        f"- Model: `{config['model_name']}`",
        f"- SFT checkpoint: `{config['sft_checkpoint']}`",
        f"- Environment URL: `{config['env_url']}`",
        f"- Training steps: `{config['num_steps']}`",
        f"- Episodes per step: `{config['episodes_per_step']}`",
        f"- Group size: `{config['group_size']}`",
        f"- Max turns per episode: `{config['max_turns']}`",
        f"- Max new tokens per turn: `{config['max_new_tokens']}`",
        f"- Max sequence length: `{config['max_seq_length']}`",
        "",
    ]

    if history["steps"]:
        lines.extend(
            [
                "## Final Metrics",
                "",
                f"- Last completed step: `{history['steps'][-1]}`",
                f"- Final average reward: `{history['rewards'][-1]:.3f}`",
                f"- Final fix rate: `{history['fix_rates'][-1]:.1%}`",
                f"- Final average commands: `{history['avg_commands'][-1]:.2f}`",
                f"- Final loss: `{history['losses'][-1]:.4f}`",
                "",
            ]
        )

    termination_totals: Counter[str] = Counter()
    for summary in step_summaries:
        termination_totals.update(summary.get("termination_counts", {}))

    if termination_totals:
        lines.extend(["## Termination Reasons", ""])
        for reason, count in sorted(termination_totals.items()):
            lines.append(f"- `{reason}`: {count}")
        lines.append("")

    lines.extend(["## Scenario Totals", ""])
    for scenario_id, stats in sorted(scenario_stats.items()):
        attempts = stats["attempts"]
        fixes = stats["fixes"]
        fix_rate = fixes / attempts if attempts else 0.0
        lines.append(f"- `{scenario_id}`: {fixes}/{attempts} fixed ({fix_rate:.1%})")

    lines.extend(["", "## Step Narrative", ""])
    for summary in step_summaries[-10:]:
        lines.append(
            f"- Step {summary['step']}: reward={summary['avg_reward']:+.3f}, "
            f"fix_rate={summary['fix_rate']:.1%}, avg_cmds={summary['avg_commands']:.2f}, "
            f"loss={summary['avg_loss']:.4f}, failures={summary['num_failed_episodes']}, "
            f"endings={summary.get('termination_counts', {})}"
        )

    example_episode = None
    for summary in reversed(step_summaries):
        if summary.get("example_episode"):
            example_episode = summary["example_episode"]
            break

    if example_episode:
        lines.extend(
            [
                "",
                "## Example Episode",
                "",
                f"- Scenario: `{example_episode['scenario_id']}`",
                f"- Fixed: `{example_episode['fixed']}`",
                f"- Total reward: `{example_episode['total_reward']:+.3f}`",
                f"- Commands: `{example_episode['num_commands']}`",
                "",
            ]
        )

        for event in example_episode["events"][: min(4, len(example_episode["events"]))]:
            lines.append(f"### Turn {event['turn']}")
            if event.get("thinking_preview"):
                lines.append("")
                lines.append(f"Thought preview: `{event['thinking_preview']}`")
            if event.get("command"):
                lines.append("")
                lines.append(f"Command: `{event['command']}`")
            if event.get("output_preview"):
                lines.append("")
                lines.append("```text")
                lines.append(textwrap.shorten(event["output_preview"], width=600, placeholder=" ..."))
                lines.append("```")
            lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
