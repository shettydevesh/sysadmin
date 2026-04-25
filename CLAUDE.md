# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Sysadmin Game

An OpenEnv-compliant RL environment for training LLMs on real Linux troubleshooting. Train Qwen2.5-Coder-7B to diagnose and fix broken Ubuntu containers using SFT warm-start + GRPO reinforcement learning.

**Why it matters**: LLMs hallucinate file paths, skip diagnostics, and propose fixes before reading errors. This environment provides unfakeable reward — a service is up or it isn't.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Training Loop (Colab)                                  │
│  Unsloth + TRL GRPOTrainer, LoRA r=16, 4-bit           │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────┐
│  OpenEnv Wrapper                                        │
│  reset() → breaks container, returns user complaint     │
│  step(action) → executes shell command, returns reward  │
└──────────────────────┬──────────────────────────────────┘
                       │ docker exec
┌──────────────────────▼──────────────────────────────────┐
│  Sandbox Layer                                          │
│  Docker containers (ubuntu:22.04 + systemd)             │
│  10 scenario modules with break()/check_fixed()         │
└─────────────────────────────────────────────────────────┘
```

## Scenarios (10 implemented)

Each scenario is a Python module with `break(container)` and `check_fixed(container)`:

| scenario_id | Category | Train Examples | Val Examples |
|-------------|----------|----------------|--------------|
| `disk_full` | Disk | 7 | 1 |
| `disk_full_alt` | Disk | 5 | 1 |
| `nginx_syntax` | Service | 11 | 1 |
| `nginx_unknown` | Service | 7 | 1 |
| `ownership` | Permissions | 11 | 1 |
| `port_bound` | Network | 11 | 1 |
| `runaway_cpu` | Process | 11 | 1 |
| `expired_cert` | TLS | 9 | 1 |
| `stale_pid` | Service | 11 | 1 |
| `venv_broken` | Environment | 9 | 1 |

**Total**: 92 training examples, 10 validation examples

## Dataset

Located in `dataset/`:

| File | Examples | Purpose |
|------|----------|---------|
| `sft_train.jsonl` | 92 | SFT training data |
| `sft_val.jsonl` | 10 | Validation (1 per scenario) |

### Format (Chat-style JSONL)

```json
{
  "messages": [
    {"role": "system", "content": "You are an SRE agent..."},
    {"role": "user", "content": "nginx won't start. Says address in use."},
    {"role": "assistant", "content": "<think>\nCheck the exact error.\n</think>\n<bash>systemctl status nginx --no-pager -l</bash>"},
    {"role": "tool", "content": "<output>\n× nginx.service...\n</output>"},
    // ... more turns ...
    {"role": "assistant", "content": "Fixed. apache2 was holding port 80..."}
  ],
  "scenario_id": "port_bound"
}
```

### Agent Response Format

The model outputs reasoning in `<think>` tags, then a single command in `<bash>` tags:
```
<think>
Port 80 is taken. Find the holder.
</think>
<bash>ss -tlnp | grep ':80 '</bash>
```

Command output is returned in `<output>` tags. Final message summarizes the fix.

## Reward Function

| Condition | Reward |
|-----------|--------|
| `check_fixed()` returns True | +1.0 |
| Per command issued | -0.01 |
| Destructive command (rm -rf /, dd, mkfs, fork bombs) | -0.5 + termination |
| First use of diagnostic command per episode | +0.1 |

**Diagnostic commands**: systemctl status, journalctl, df, ls, cat, ps, netstat/ss, grep

## Episode Constraints

- 25 command limit
- 60 second wallclock timeout
- 2KB output truncation per command
- Fresh container on each reset

## Development Commands

```bash
# Environment setup
pip install docker unsloth trl transformers datasets wandb

# Test single scenario manually
docker run -d --name test --privileged ubuntu:22.04 /sbin/init
python -c "from scenarios.nginx_config import break_system; break_system('test')"
docker exec -it test bash  # diagnose manually
python -c "from scenarios.nginx_config import check_fixed; print(check_fixed('test'))"

# Start OpenEnv HTTP server
python -m sysadmin_env.server --port 8080

# Run SFT training (Phase 1, ~30 min on A100)
python train_sft.py --train dataset/sft_train.jsonl --val dataset/sft_val.jsonl --epochs 2

# Run GRPO training (Phase 2)
python train_grpo.py --env-url http://localhost:8080 --steps 500

# Evaluate on held-out scenarios
python eval.py --model baseline --scenarios held_out
python eval.py --model trained --scenarios held_out
```

## Project Structure

```
dataset/
├── sft_train.jsonl      # 92 training examples (chat format)
└── sft_val.jsonl        # 10 validation examples

sysadmin_env/
├── scenarios/           # 10 break/check Python modules
├── sandbox.py           # Docker container management
├── env.py               # OpenEnv wrapper (reset/step)
├── server.py            # HTTP server for training loop
├── reward.py            # Reward calculation
└── blocklist.py         # Destructive command detection

training/
├── train_sft.py         # Unsloth SFT warm-start
├── train_grpo.py        # TRL GRPO training loop
└── eval.py              # Held-out evaluation
```

## Critical Constraints

1. **Eval contamination**: Held-out scenarios must NEVER appear in SFT traces or RL training
2. **Context blowup**: Truncate outputs to 2KB; teach agent head/tail/grep in SFT data
3. **Reward hacking**: Include scenarios where blind restart makes things worse
4. **Parameter randomization**: Vary ports, service names, file paths across episodes

## Training Pipeline

**Phase 1 - SFT Warm-Start** (~30 min A100): 92 curated traces from `dataset/sft_train.jsonl`, 1-2 epochs

**Phase 2 - GRPO** (~5 hours): TRL GRPOTrainer, group size 4-8, 200-500 steps, log to W&B

**Decision Point (Hour 12)**: If GRPO curves are flat, ship SFT-only with honest "RL is future work"

## Models

- **Primary**: Qwen2.5-Coder-7B-Instruct (A100/H100)
- **Colab fallback**: Qwen2.5-Coder-3B-Instruct (T4/L4)
- **Config**: LoRA r=16, 4-bit via Unsloth

## Timeline

| Hours | Milestone |
|-------|-----------|
| 0-2 | Sandbox + one scenario E2E |
| 2-4 | OpenEnv wrapper + all 10 scenarios |
| 4-6 | SFT trace generation + SFT run |
| 6-7 | Baseline eval, save plots |
| 7-12 | GRPO training loop |
| 12-13 | Trained-model eval, comparison plots |
| 13-14 | HF Space deploy, README |
| 14-15 | Demo video |

## Deliverables

- [ ] OpenEnv-compliant environment on HF Spaces
- [ ] 10 working scenarios with break/check functions
- [ ] Colab notebook with full pipeline (Unsloth SFT + TRL GRPO)
- [ ] README with embedded plots and baseline-vs-trained table
- [ ] Training plots committed to `results/` (reward curve, success rate, commands-to-fix)
- [ ] < 2 minute demo video (untrained vs trained)
- [ ] W&B run links for all training runs

## Available Claude Code Skills

Use these to accelerate development:

| Command | Use Case |
|---------|----------|
| `/karpathy-check` | Review staged changes against 4 coding principles before committing |
| `/make-plan` | Create detailed implementation plans with documentation discovery |
| `/do` | Execute phased implementation plans using subagents |
| `/mem-search` | Search persistent memory for past solutions |
| `/smart-explore` | Token-efficient AST-based code search |
| `/timeline-report` | Generate project history analysis |
| `/version-bump` | Automated semantic versioning for releases |

### Engineering Skills (engineering-advanced-skills)

25 advanced skills for agent design, RAG, MCP servers, CI/CD, database design, observability, security auditing, release management.

## What Makes a Submission Stand Out

### Show Real Training End-to-End
- Training loop must connect to the live environment (not a static dataset)
- Train long enough that curves are meaningful
- Compare trained agent vs untrained/random baseline (quantitative + qualitative)
- Include plots and numbers in README

### Make Plots Readable
- Label both axes with units (e.g., "Training Step" / "Reward")
- Save as `.png` or `.jpg` and commit to repo
- Embed key plots in README with one-line captions
- Multiple runs (baseline vs trained) on same axes for easy comparison
- Include W&B run links if used

### Required Artifacts
```
results/
├── reward_curve.png         # Training reward over steps
├── baseline_vs_trained.png  # Side-by-side comparison
├── success_rate.png         # Held-out scenario success rates
└── commands_to_fix.png      # Efficiency comparison
```

## Judging Rubric

- **Innovation (40%)**: Real shell, real services, real reward — not a simulation
- **Storytelling (30%)**: Untrained-vs-trained 30-second clip
- **Reward improvement (20%)**: Held-out success rate, commands-to-fix
- **Pipeline (10%)**: OpenEnv-compliant, reproducible, modular scenarios
