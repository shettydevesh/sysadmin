"""Gradio app for Sysadmin Game GRPO training on HF Spaces."""

import os
import re
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import gradio as gr
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============== Configuration ==============

@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    num_steps: int = 50
    episodes_per_step: int = 4
    group_size: int = 2
    max_turns: int = 8
    learning_rate: float = 1e-5
    temperature: float = 0.7
    max_seq_length: int = 2048


SYSTEM_PROMPT = """You are an expert SRE agent that diagnoses and fixes Linux system issues.

When given a problem:
1. Think step-by-step about the diagnosis in <think> tags
2. Run ONE command at a time in <bash> tags
3. Analyze the output before running the next command

Example:
<think>
Check the service status first.
</think>
<bash>systemctl status nginx --no-pager -l</bash>"""


# ============== Training State ==============

class TrainingState:
    def __init__(self):
        self.is_training = False
        self.current_step = 0
        self.total_steps = 0
        self.history = {"steps": [], "rewards": [], "fix_rates": [], "losses": []}
        self.logs = []
        self.model = None
        self.tokenizer = None

    def log(self, msg: str):
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

state = TrainingState()


# ============== Model Loading ==============

def load_model(model_name: str, progress=gr.Progress()):
    """Load model for training."""
    progress(0.1, desc="Loading tokenizer...")
    state.log(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    progress(0.3, desc="Loading model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer))

    if device == "cpu":
        model = model.to(device)

    state.model = model
    state.tokenizer = tokenizer
    state.log(f"Model loaded on {device}")

    progress(1.0, desc="Done!")
    return f"✅ Model loaded: {model_name} on {device}"


# ============== Real HTTP Environment ==============

class RealEnvHTTP:
    """HTTP client for the real Sysadmin Game environment server.

    Mirrors the SimulatedEnv dict-based interface so run_episode works
    unchanged whether we're talking to a real Docker sandbox or the fake sim.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.episode_id: Optional[str] = None

    def _post(self, endpoint: str, payload: dict) -> dict:
        from urllib import request as urlrequest
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        req = urlrequest.Request(
            f"{self.base_url}{endpoint}", data=data, headers=headers, method="POST"
        )
        with urlrequest.urlopen(req, timeout=90.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def reset(self, scenario_id: Optional[str] = None) -> dict:
        resp = self._post("/reset", {"scenario_id": scenario_id})
        self.episode_id = resp["metadata"]["episode_id"]
        return {
            "output": resp["output"],
            "done": resp["done"],
            "reward": resp["reward"],
            "metadata": resp["metadata"],
        }

    def step(self, command: str) -> dict:
        resp = self._post("/step", {"command": command, "episode_id": self.episode_id})
        return {
            "output": resp["output"],
            "done": resp["done"],
            "reward": resp["reward"],
            "metadata": resp["metadata"],
        }


def check_env_server(env_url: str) -> str:
    """Ping the env server and return a status string."""
    if not env_url.strip():
        return "⚠️ No server URL — will use simulated environment"
    try:
        from urllib import request as urlrequest
        url = env_url.rstrip("/") + "/health"
        with urlrequest.urlopen(url, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        scenarios_url = env_url.rstrip("/") + "/scenarios"
        with urlrequest.urlopen(scenarios_url, timeout=5.0) as resp:
            sc = json.loads(resp.read().decode("utf-8"))
        return (
            f"✅ Connected to real environment server\n"
            f"Status: {data.get('status')}  Active episodes: {data.get('active_episodes', 0)}\n"
            f"Train scenarios: {', '.join(sc.get('train', []))}"
        )
    except Exception as e:
        return f"❌ Cannot reach server at {env_url}\nError: {e}"


# ============== Simulated Environment (fallback) ==============

SIMULATED_SCENARIOS = [
    {
        "id": "nginx_syntax",
        "complaint": "nginx won't start, getting config errors",
        "solution_pattern": r"nginx -t|vim.*nginx|nano.*nginx|systemctl restart nginx",
        "diagnostics": ["systemctl status nginx", "nginx -t", "cat /etc/nginx"],
    },
    {
        "id": "disk_full",
        "complaint": "Can't save files, disk full errors everywhere",
        "solution_pattern": r"rm |find.*-delete|truncate|df -h",
        "diagnostics": ["df -h", "du -sh", "ls -la /var/log"],
    },
    {
        "id": "port_bound",
        "complaint": "nginx says address already in use on port 80",
        "solution_pattern": r"kill|systemctl stop|fuser -k",
        "diagnostics": ["ss -tlnp", "netstat -tlnp", "lsof -i :80"],
    },
]


class SimulatedEnv:
    """Fallback when no real server URL is provided."""

    def __init__(self):
        self.scenario = None
        self.commands_run = []
        self.fixed = False

    def reset(self, scenario_id=None) -> dict:
        import random
        if scenario_id:
            self.scenario = next(
                (s for s in SIMULATED_SCENARIOS if s["id"] == scenario_id),
                random.choice(SIMULATED_SCENARIOS),
            )
        else:
            self.scenario = random.choice(SIMULATED_SCENARIOS)
        self.commands_run = []
        self.fixed = False
        return {
            "output": self.scenario["complaint"],
            "done": False,
            "reward": 0.0,
            "metadata": {"scenario_id": self.scenario["id"]},
        }

    def step(self, command: str) -> dict:
        self.commands_run.append(command)
        reward = -0.01

        for diag in self.scenario["diagnostics"]:
            if diag.split()[0] in command:
                reward += 0.1
                break

        if re.search(self.scenario["solution_pattern"], command, re.IGNORECASE):
            self.fixed = True
            reward += 1.0

        done = self.fixed or len(self.commands_run) >= 15
        return {
            "output": f"[Simulated output for: {command}]",
            "done": done,
            "reward": reward,
            "metadata": {"scenario_id": self.scenario["id"], "fixed": self.fixed},
        }


# ============== Episode Runner ==============

def parse_response(response: str) -> Optional[str]:
    """Extract command from model response."""
    bash_match = re.search(r"<bash>(.*?)</bash>", response, re.DOTALL)
    return bash_match.group(1).strip() if bash_match else None


def run_episode(env, model, tokenizer, config: Config) -> dict:
    """Run single episode against env (real or simulated)."""
    obs = env.reset()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs["output"]},
    ]

    trajectory = {
        "scenario_id": obs["metadata"]["scenario_id"],
        "prompts": [],
        "responses": [],
        "rewards": [],
    }

    total_reward = 0.0

    for turn in range(config.max_turns):
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_seq_length - 200,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            command = parse_response(response)

            trajectory["prompts"].append(prompt)
            trajectory["responses"].append(response)

            if not command:
                trajectory["rewards"].append(-0.1)
                total_reward -= 0.1
                break

            obs = env.step(command)
            trajectory["rewards"].append(obs["reward"])
            total_reward += obs["reward"]

            messages.append({"role": "assistant", "content": response})
            messages.append(
                {"role": "tool", "content": f"<output>\n{obs['output']}\n</output>"}
            )

            if obs["done"]:
                break

        except Exception as e:
            state.log(f"Turn {turn} error: {str(e)[:80]}")
            trajectory["rewards"].append(-0.1)
            total_reward -= 0.1
            break

    trajectory["total_reward"] = total_reward
    trajectory["fixed"] = obs["metadata"].get("fixed", False)
    return trajectory


# ============== GRPO Training ==============

def compute_advantages(trajectories: list, group_size: int) -> list:
    """Compute GRPO advantages (normalize within group)."""
    processed = []
    for i in range(0, len(trajectories) - group_size + 1, group_size):
        group = trajectories[i : i + group_size]
        rewards = [t["total_reward"] for t in group]
        mean_r = sum(rewards) / len(rewards)
        std_r = max((sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5, 1e-6)

        for traj in group:
            advantage = (traj["total_reward"] - mean_r) / std_r
            for prompt, response, reward in zip(
                traj["prompts"], traj["responses"], traj["rewards"]
            ):
                processed.append(
                    {"prompt": prompt, "response": response, "advantage": advantage}
                )
    return processed


def train_step(model, tokenizer, env, config: Config, optimizer) -> tuple:
    """Single GRPO training step."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trajectories = []
    for _ in range(config.episodes_per_step):
        try:
            traj = run_episode(env, model, tokenizer, config)
            trajectories.append(traj)
        except Exception as e:
            state.log(f"Episode failed: {str(e)[:100]}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not trajectories:
        return None, None, None

    rewards = [t["total_reward"] for t in trajectories]
    fix_rate = sum(1 for t in trajectories if t["fixed"]) / len(trajectories)
    avg_reward = sum(rewards) / len(rewards)

    training_data = compute_advantages(trajectories, config.group_size)
    if not training_data:
        return avg_reward, fix_rate, 0.0

    model.train()
    total_loss = 0.0

    for ex in training_data:
        inputs = tokenizer(
            ex["prompt"] + ex["response"],
            return_tensors="pt",
            truncation=True,
            max_length=config.max_seq_length,
        ).to(model.device)

        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss * ex["advantage"]

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    return avg_reward, fix_rate, total_loss / max(len(training_data), 1)


def run_training(
    num_steps: int,
    episodes_per_step: int,
    learning_rate: float,
    env_url: str,
    progress=gr.Progress(),
):
    """Main training loop — uses real HTTP env if env_url is set, else simulated."""
    if state.model is None:
        return "❌ Load a model first!", None, ""

    config = Config(
        num_steps=int(num_steps),
        episodes_per_step=int(episodes_per_step),
        learning_rate=float(learning_rate),
    )

    state.is_training = True
    state.total_steps = config.num_steps
    state.history = {"steps": [], "rewards": [], "fix_rates": [], "losses": []}

    # Pick environment: real if URL provided, fake otherwise
    if env_url.strip():
        env = RealEnvHTTP(env_url.strip())
        env_label = f"Real server: {env_url.strip()}"
    else:
        env = SimulatedEnv()
        env_label = "Simulated (no server URL)"

    optimizer = torch.optim.AdamW(state.model.parameters(), lr=config.learning_rate)

    state.log(f"Starting training: {config.num_steps} steps, {config.episodes_per_step} eps/step")
    state.log(f"Environment: {env_label}")

    for step in range(config.num_steps):
        if not state.is_training:
            break

        state.current_step = step + 1
        progress((step + 1) / config.num_steps, desc=f"Step {step+1}/{config.num_steps}")

        reward, fix_rate, loss = train_step(
            state.model, state.tokenizer, env, config, optimizer
        )

        if reward is not None:
            state.history["steps"].append(step + 1)
            state.history["rewards"].append(reward)
            state.history["fix_rates"].append(fix_rate)
            state.history["losses"].append(loss)
            state.log(
                f"Step {step+1}: reward={reward:.3f}, fix={fix_rate:.1%}, loss={loss:.4f}"
            )

    state.is_training = False
    fig = create_training_plot()
    final_reward = state.history["rewards"][-1] if state.history["rewards"] else 0.0
    return f"✅ Training complete! Final reward: {final_reward:.3f}", fig, "\n".join(state.logs[-20:])


def stop_training():
    state.is_training = False
    state.log("Training stopped by user")
    return "Training stopped"


def create_training_plot():
    """Create training curves plot."""
    if not state.history["steps"]:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(state.history["steps"], state.history["rewards"], "b-", lw=2)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Average Reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(state.history["steps"], state.history["fix_rates"], "g-", lw=2)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Fix Rate")
    axes[1].set_title("Success Rate")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(state.history["steps"], state.history["losses"], "r-", lw=2)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Policy Loss")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============== Demo Tab ==============

def run_demo(complaint: str):
    """Run a single demo episode."""
    if state.model is None:
        return "Load a model first!"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": complaint},
    ]

    prompt = state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = state.tokenizer(prompt, return_tensors="pt").to(state.model.device)

    with torch.no_grad():
        outputs = state.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=state.tokenizer.eos_token_id,
        )

    response = state.tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return response


# ============== Gradio UI ==============

with gr.Blocks(title="Sysadmin Game - GRPO Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛠️ Sysadmin Game: GRPO Training")
    gr.Markdown(
        "Train LLMs to diagnose and fix Linux systems using reinforcement learning.\n\n"
        "**With real environment:** Start your environment server locally, expose it via "
        "ngrok, paste the URL below. **Without:** falls back to a simulated environment."
    )

    with gr.Tabs():
        # ── Setup Tab ──────────────────────────────────────────────────────────
        with gr.Tab("1. Setup"):
            gr.Markdown("### Load Model")
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=[
                        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        "Qwen/Qwen2.5-Coder-3B-Instruct",
                    ],
                    value="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                    label="Model",
                )
                load_btn = gr.Button("Load Model", variant="primary")
            load_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Environment Server (optional — for real Docker sandbox)")
            with gr.Row():
                env_url_input = gr.Textbox(
                    label="Environment Server URL",
                    placeholder="https://abc123.ngrok-free.app  (leave empty for simulated env)",
                    scale=4,
                )
                check_btn = gr.Button("Check Connection", scale=1)
            env_status = gr.Textbox(label="Environment Status", interactive=False, lines=3)

            load_btn.click(load_model, inputs=[model_dropdown], outputs=[load_status])
            check_btn.click(check_env_server, inputs=[env_url_input], outputs=[env_status])

        # ── Training Tab ───────────────────────────────────────────────────────
        with gr.Tab("2. Training"):
            gr.Markdown("### GRPO Training Configuration")
            with gr.Row():
                num_steps = gr.Slider(10, 200, value=50, step=10, label="Training Steps")
                episodes = gr.Slider(2, 16, value=4, step=2, label="Episodes per Step")
                lr = gr.Number(value=1e-5, label="Learning Rate")

            env_url_train = gr.Textbox(
                label="Environment Server URL (copy from Setup tab)",
                placeholder="https://abc123.ngrok-free.app  or leave empty for simulated",
            )

            with gr.Row():
                train_btn = gr.Button("Start Training", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")

            train_status = gr.Textbox(label="Status", interactive=False)
            train_plot = gr.Plot(label="Training Curves")
            train_logs = gr.Textbox(label="Logs", lines=10, interactive=False)

            train_btn.click(
                run_training,
                inputs=[num_steps, episodes, lr, env_url_train],
                outputs=[train_status, train_plot, train_logs],
            )
            stop_btn.click(stop_training, outputs=[train_status])

        # ── Demo Tab ───────────────────────────────────────────────────────────
        with gr.Tab("3. Demo"):
            gr.Markdown("### Test the Model")
            complaint_input = gr.Textbox(
                label="User Complaint",
                placeholder="e.g., nginx won't start, getting config errors",
                lines=2,
            )
            demo_btn = gr.Button("Get Diagnosis", variant="primary")
            demo_output = gr.Textbox(label="Model Response", lines=10)

            demo_btn.click(run_demo, inputs=[complaint_input], outputs=[demo_output])

            gr.Examples(
                examples=[
                    ["nginx won't start, says something about address already in use"],
                    ["Can't write any files, getting 'No space left on device' errors"],
                    ["Getting permission denied when trying to read /var/log/syslog"],
                ],
                inputs=[complaint_input],
            )

demo.launch(server_name="0.0.0.0", server_port=7860)
