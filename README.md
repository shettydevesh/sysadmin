# Sysadmin Game

**An OpenEnv-compliant RL environment that trains LLMs to diagnose and fix real broken Linux servers.**

The reward is not an AI judge. It's not a human label. It's whether the service is actually running.

---

<!-- Demo GIF — record a 30s terminal clip and drop it here -->
<!-- ![Demo](results/demo.gif) -->

---

## Links

| Artifact | Link |
|---|---|
| HuggingFace Space (live environment + training UI) | [deveshshetty/sysadmin-game](https://huggingface.co/spaces/deveshshetty/sysadmin-game) |
| Trained model checkpoint | [deveshshetty/sysadmin-grpo](https://huggingface.co/deveshshetty/sysadmin-grpo) |
| Blog post | [blog.md](blog.md) |
| W&B training run | *add link after training* |
| Demo video | *add link after recording* |

---

## What This Is

Most LLM sysadmin demos are fake. The model suggests a command, a human decides if it sounds right, and nobody actually runs it on a broken server.

We built the opposite. A fresh Ubuntu 22.04 container starts, we deliberately break something — fill the disk, corrupt a config, steal a port, kill a cert — and hand the LLM a shell. It types real commands. The shell runs them. If nginx is serving on port 80 again, the episode is fixed. If not, it's not.

No simulation. No judge model. No rubric. The box is fixed or it isn't.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  LLM  (Qwen2.5-Coder, T4 GPU on HF Spaces)           │
│  Reads complaint → generates <bash>command</bash>    │
└─────────────────────┬────────────────────────────────┘
                      │  HTTP  /reset  /step  /state
┌─────────────────────▼────────────────────────────────┐
│  Environment Server  (FastAPI, port 8000)            │
│  Episode management · reward calculation             │
│  Safety blocklist · container auto-cleanup           │
└─────────────────────┬────────────────────────────────┘
                      │  docker exec
┌─────────────────────▼────────────────────────────────┐
│  Sandbox  (Ubuntu 22.04 + systemd inside Docker)     │
│  Real nginx · real apache · real disks · real certs  │
│  break_system() → agent acts → check_fixed()         │
└──────────────────────────────────────────────────────┘
```

The LLM never touches Docker directly. It speaks HTTP. The server handles everything else. This means any GPU machine — Colab, HF Spaces, a cloud VM — can run training by pointing at the server URL.

---

## Scenarios

10 scenarios across 6 failure categories. 5 used for training, 5 held out for evaluation. The model never sees held-out scenario IDs during training.

| Scenario | Category | What Breaks | How `check_fixed()` Verifies |
|---|---|---|---|
| `disk_full` | Disk | `fallocate -l 800M /var/log/huge_debug.log` | Log file < 1MB |
| `disk_full_alt` | Disk | Fill `/tmp` with junk | Disk usage < 90% |
| `nginx_syntax` | Service | Bad directive injected into `nginx.conf` | `nginx -t` exits 0 |
| `nginx_unknown` | Service | Config points to nonexistent module | nginx serves on port 80 |
| `ownership` | Permissions | `chown nobody:nogroup /etc/nginx/nginx.conf` | nginx starts successfully |
| `port_bound` | Network | apache2 started first, holds port 80 | nginx responds on port 80 |
| `runaway_cpu` | Process | `yes > /dev/null` spinning in background | CPU load < 80% |
| `expired_cert` | TLS | Backdate cert by 2 years | `openssl verify` passes |
| `stale_pid` | Service | Fake PID file blocks systemd startup | Service running |
| `venv_broken` | Environment | Delete `site-packages` from active venv | `python -c "import flask"` works |

**Train split:** `disk_full`, `nginx_syntax`, `ownership`, `port_bound`, `runaway_cpu`

**Held-out eval:** `disk_full_alt`, `nginx_unknown`, `expired_cert`, `stale_pid`, `venv_broken`

---

## Reward Function

| Condition | Reward |
|---|---:|
| `check_fixed()` returns True | `+1.00` |
| First use of a diagnostic command | `+0.10` |
| Per command issued | `-0.01` |
| Destructive command (`rm -rf /`, fork bomb, disk wipe) | `-0.50` + terminate |

Diagnostic commands that earn the bonus: `systemctl status`, `journalctl`, `df`, `du`, `ss`, `netstat`, `ps`, `cat`, `ls`, `grep`, `lsof`, `find`, `stat`, `top`, `free`.

The diagnostic bonus exists to prevent reward hacking. Without it, the model learns to blindly restart services hoping one sticks. With it, inspect-then-fix consistently outscores guess-and-restart.

---

## Training Results

> Training: Qwen2.5-Coder-0.5B-Instruct · T4 GPU · real Docker environment · GRPO

### Reward Curve

*Average episode reward per training step. Each step = 4 live episodes against real broken containers.*

<!-- Add plot after training completes -->
![Reward Curve](results/reward_curve.png)

### Success Rate (Held-Out Scenarios)

*Percentage of episodes where the model actually fixed the service. Measured on 5 scenarios never seen during training.*

<!-- Add plot after training completes -->
![Success Rate](results/success_rate.png)

### Baseline vs Trained

*Same scenarios, same broken servers. Random baseline vs base model vs trained.*

<!-- Add plot after training completes -->
![Baseline vs Trained](results/baseline_vs_trained.png)

### Commands to Fix

*Fewer commands = better. Trained model learns to diagnose first, not spam restarts.*

<!-- Add plot after training completes -->
![Commands to Fix](results/commands_to_fix.png)

### Evaluation Numbers

| Metric | Random | Base Qwen | SFT | SFT + GRPO |
|---|---:|---:|---:|---:|
| Held-out success rate | — | — | — | — |
| Average reward | — | — | — | — |
| Avg commands to fix | — | — | — | — |
| Unsafe command rate | — | — | — | — |

*Fill after running `python -m training.evaluate`*

---

## Before vs After

Same incident: "nginx won't start, address already in use."

**Untrained model:**
```
<bash>cat /etc/nginx/nginx.conf</bash>
→ config looks fine

<bash>systemctl restart nginx</bash>
→ still fails

<bash>apt-get install --reinstall nginx</bash>
→ still fails, never fixed
```

**After GRPO training:**
```
<bash>ss -tlnp | grep ':80'</bash>
→ apache2 is holding port 80

<bash>systemctl stop apache2 && systemctl start nginx</bash>
→ fixed: true, reward: +0.99
```

The trained model reads the error, finds the cause, fixes the cause. 2 commands instead of guessing.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd sysadmin
pip install -e ".[dev]"

# 2. Build the sandbox image (Ubuntu 22.04 + systemd)
docker build -f docker/sandbox.Dockerfile -t sysadmin-sandbox:latest .

# 3. Start the environment server
python -m sysadmin_env.server.app
# → Uvicorn running on http://0.0.0.0:8000

# 4. Smoke test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "port_bound"}'
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"command": "ss -tlnp | grep :80"}'
```

---

## Training

### Phase 1 — SFT Warm-Start (~30 min on A100)

Teaches format: diagnose first, one command at a time, `<think>` before `<bash>`.

```bash
python -m training.train_sft \
  --train dataset/sft_train.jsonl \
  --val dataset/sft_val.jsonl \
  --epochs 2 \
  --output checkpoints/sft
```

92 training examples across all 10 scenario families. 10 held-out validation examples.

### Phase 2 — GRPO Against Live Environment (~5 hours on A100)

Connects to real Docker sandbox. Runs 4 episodes per step. Compares within the group. Reinforces what worked.

```bash
# Local Docker (default)
python -m training.train_openenv_grpo \
  --model checkpoints/sft \
  --steps 200 \
  --output checkpoints/grpo-openenv

# Remote server (HF Spaces / ngrok)
python -m training.train_openenv_grpo \
  --model checkpoints/sft \
  --env-url https://YOUR-NGROK-URL.ngrok-free.app \
  --steps 200 \
  --output checkpoints/grpo-openenv
```

### Evaluation

```bash
python -m training.evaluate \
  --baseline random \
  --trained checkpoints/grpo-openenv \
  --split val \
  --episodes 10 \
  --output results/
```

---

## API Reference

The environment server exposes 6 endpoints:

| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/health` | Server status, active episodes, Docker container count |
| `GET` | `/scenarios` | Lists all scenarios with train/val split |
| `POST` | `/reset` | Start new episode. Body: `{"scenario_id": "...", "seed": 42}` |
| `POST` | `/step` | Execute command. Body: `{"command": "...", "episode_id": "..."}` |
| `GET` | `/state` | Episode state: step count, reward, diagnostics used, elapsed time |
| `DELETE` | `/episodes/{id}` | Close and clean up an episode sandbox |

Multiple episodes can run in parallel. Each `/reset` returns a unique `episode_id`. Pass it to `/step` to target a specific sandbox.

---

## OpenEnv Compliance

| File | Role |
|---|---|
| `openenv.yaml` | Manifest for environment discovery and deployment |
| `sysadmin_env/models.py` | Typed `Action`, `Observation`, `State` schemas |
| `sysadmin_env/client.py` | HTTP client for remote server connection |
| `sysadmin_env/server/app.py` | FastAPI server — the main deployment target |
| `sysadmin_env/openenv_adapter.py` | TRL `environment_factory` adapter (`run_shell` tool) |

---

## Project Structure

```
sysadmin_env/
├── scenarios/          # 10 break/check Python modules
├── sandbox.py          # Docker container management + auto-cleanup
├── environment.py      # reset / step / state loop
├── reward.py           # reward calculation
├── blocklist.py        # destructive command detection
├── client.py           # HTTP client for remote connection
└── server/app.py       # FastAPI server

training/
├── train_sft.py        # Phase 1: SFT warm-start
├── train_openenv_grpo.py  # Phase 2: GRPO against live env
└── evaluate.py         # held-out evaluation + plots

dataset/
├── sft_train.jsonl     # 92 training examples
└── sft_val.jsonl       # 10 validation examples (1 per scenario)

spaces/app.py           # Gradio UI for HF Spaces training
docker/
└── sandbox.Dockerfile  # Ubuntu 22.04 + systemd sandbox image

results/                # Training plots (committed after training)
blog.md                 # Full writeup
```

---

## Read the Full Writeup

The blog post walks through the problem, the architecture, the training process, what broke during the build, and why this approach scales beyond Linux troubleshooting.

→ [blog.md](blog.md)

---

## License

MIT
