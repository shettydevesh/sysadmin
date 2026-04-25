# Sysadmin Game

An OpenEnv-compliant RL environment for training LLMs on real Linux troubleshooting.

## Overview

Train Qwen2.5-Coder-7B to diagnose and fix broken Ubuntu containers using SFT warm-start + GRPO reinforcement learning.

**Why it matters**: LLMs hallucinate file paths, skip diagnostics, and propose fixes before reading errors. This environment provides unfakeable reward — a service is up or it isn't.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Start the HTTP server (requires Docker)
uv run sysadmin-server
```

## Full Training Pipeline

### 1. Prepare Dataset

Create `dataset/sft_train.jsonl` with training examples in chat format:

```json
{
  "messages": [
    {"role": "system", "content": "You are an SRE agent..."},
    {"role": "user", "content": "nginx won't start after I edited the config."},
    {"role": "assistant", "content": "<think>\nCheck nginx config syntax.\n</think>\n<bash>nginx -t</bash>"},
    {"role": "tool", "content": "<output>\nnginx: syntax error...\n</output>"},
    {"role": "assistant", "content": "<think>\nFound syntax error. Fix the config.\n</think>\n<bash>cat /etc/nginx/sites-available/default</bash>"}
  ],
  "scenario_id": "nginx_syntax"
}
```

### 2. Train with SFT (Phase 1)

```bash
# Install training dependencies
uv sync --extra train

# Run SFT training (~30 min on A100)
uv run python -m training.train_sft \
  --train dataset/sft_train.jsonl \
  --val dataset/sft_val.jsonl \
  --epochs 2 \
  --output checkpoints/sft
```

### 3. Evaluate Trained vs Baseline

```bash
# Install eval dependencies
uv sync --extra eval

# Run evaluation (requires Docker)
uv run python -m training.evaluate \
  --baseline random \
  --trained checkpoints/sft \
  --scenarios ownership nginx_syntax disk_full port_bound \
  --episodes 3 \
  --output results/
```

This generates:
- `results/success_rate.png` - Success rate comparison
- `results/avg_reward.png` - Average reward comparison
- `results/commands_to_fix.png` - Efficiency comparison
- `results/baseline_vs_trained.png` - Combined comparison
- `results/eval_results.json` - Full metrics

### 4. Interactive Demo

```bash
# Compare agents side-by-side
uv run python -m training.demo \
  --scenario ownership \
  --trained checkpoints/sft
```

## HTTP API

```bash
# Reset to a new episode
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "nginx_syntax"}'

# Execute a command
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"command": "systemctl status nginx"}'

# Get current state
curl http://localhost:8000/state

# List scenarios
curl http://localhost:8000/scenarios
```

## Scenarios

| ID | Category | Description |
|----|----------|-------------|
| disk_full | Disk | /var/log filled with large file |
| disk_full_alt | Disk | Disk filled via syslog |
| nginx_syntax | Service | Invalid nginx config syntax |
| nginx_unknown | Service | Unknown directive in nginx config |
| ownership | Permissions | Wrong file ownership |
| port_bound | Network | Another process using port 80 |
| runaway_cpu | Process | Infinite loop consuming CPU |
| expired_cert | TLS | SSL certificate expired |
| stale_pid | Service | Leftover PID file |
| venv_broken | Environment | Broken Python venv symlinks |

## Reward Structure

| Condition | Reward |
|-----------|--------|
| Issue fixed | +1.0 |
| Per command | -0.01 |
| First diagnostic command | +0.1 |
| Destructive command | -0.5 + termination |

**Diagnostic commands** (bonus on first use): systemctl status, journalctl, df, ls, cat, ps, ss, netstat, grep, head, tail, lsof, du, free, top

## Episode Constraints

- 25 command limit
- 60 second timeout
- 2KB output truncation per command
- Fresh container on each reset

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Training Loop (Colab/Local)                            │
│  Unsloth + TRL, LoRA r=16, 4-bit quantization          │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP API
┌──────────────────────▼──────────────────────────────────┐
│  OpenEnv Wrapper (sysadmin_env)                         │
│  reset() → breaks container, returns user complaint     │
│  step(action) → executes command, returns reward        │
└──────────────────────┬──────────────────────────────────┘
                       │ docker exec
┌──────────────────────▼──────────────────────────────────┐
│  Sandbox Layer (Docker)                                 │
│  ubuntu:22.04 + systemd, 10 scenario modules           │
└─────────────────────────────────────────────────────────┘
```

## Agent Response Format

The model outputs reasoning in `<think>` tags, then a single command in `<bash>` tags:

```
<think>
Port 80 is taken. Find the holder.
</think>
<bash>ss -tlnp | grep ':80 '</bash>
```

## Project Structure

```
sysadmin_env/
├── __init__.py          # Package exports
├── models.py            # Action, Observation, State
├── environment.py       # Core SysadminEnv class
├── sandbox.py           # Docker container management
├── reward.py            # Reward calculation
├── blocklist.py         # Destructive command detection
├── scenarios/           # 10 scenario implementations
└── server/app.py        # FastAPI HTTP server

training/
├── agent.py             # SysadminAgent, RandomAgent
├── train_sft.py         # SFT training script
├── evaluate.py          # Evaluation + plotting
└── demo.py              # Interactive comparison demo

dataset/
├── sft_train.jsonl      # Training examples
└── sft_val.jsonl        # Validation examples

results/                 # Generated plots and metrics
```

## Expected Results

After SFT training on 92 examples:

| Metric | Random Baseline | Trained Model |
|--------|-----------------|---------------|
| Success Rate | ~10% | ~60-80% |
| Avg Reward | -0.15 | +0.70 |
| Commands to Fix | N/A | 4-8 |

## License

MIT
