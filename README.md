# Sysadmin Game

An OpenEnv-compatible RL environment for training LLM agents to diagnose and
fix real Linux incidents in sandboxed Ubuntu containers.

## Submission Links

Fill these before final submission:

| Artifact | Link |
|---|---|
| Hugging Face Space | TODO |
| Colab training notebook | TODO |
| W&B run or training logs | TODO |
| Demo video / blog / slides | TODO |
| Trained model checkpoint | TODO |

## Why This Exists

LLMs often sound confident while skipping the actual sysadmin workflow: inspect
the error, find the root cause, make a minimal fix, then verify. Sysadmin Game
turns that behavior into a trainable environment. The reward is grounded in the
machine state: nginx starts or it does not, disk pressure is relieved or it is
not, the certificate validates or it does not.

## What The Agent Sees And Does

Each episode starts with a realistic user complaint. The agent can run one bash
command at a time through the environment. The environment returns command
output, reward, done state, and metadata.

```text
User complaint:
nginx won't start. Says 'Address already in use'. I need the web server back up ASAP!

Agent action:
ss -tlnp | grep ':80 '

Environment output:
LISTEN ... users:(("apache2",pid=...))
reward=+0.09 done=False fixed=False
```

The model is rewarded for fixing the incident, gently rewarded for useful first
diagnostics, penalized per command, and terminated for destructive commands.

## OpenEnv Shape

This repo includes the OpenEnv-facing pieces required for packaging and remote
training:

- `openenv.yaml`: manifest for environment discovery.
- `sysadmin_env/models.py`: typed `Action`, `Observation`, and `State` schemas.
- `sysadmin_env/client.py`: HTTP client plus optional OpenEnv-native client.
- `sysadmin_env/server/app.py`: HTTP server with reset, step, state, health, and scenario endpoints.
- `sysadmin_env/server/openenv_app.py`: optional OpenEnv-native `create_app` entrypoint.
- `sysadmin_env/openenv_adapter.py`: TRL `environment_factory` adapter exposing `run_shell(command)`.

## Scenarios

| Split | Scenario IDs |
|---|---|
| Train | `disk_full`, `nginx_syntax`, `ownership`, `port_bound`, `runaway_cpu` |
| Held-out eval | `disk_full_alt`, `nginx_unknown`, `expired_cert`, `stale_pid`, `venv_broken` |

All 10 scenarios are available through `/scenarios`. The default live RL and SFT
commands use the train split, while evaluation defaults to held-out scenarios.

## Local Setup

```bash
# Install project dependencies
uv sync --extra dev --extra train --extra eval

# Build the Ubuntu/systemd sandbox used inside episodes
docker build -f docker/sandbox.Dockerfile -t sysadmin-sandbox:latest .

# Run unit tests
uv run pytest tests/ -v

# Start the environment API
uv run sysadmin-server
```

If `uv` is not installed, install it first or use your virtualenv/pip equivalent.

## API Smoke Test

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "port_bound"}'

curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"command": "ss -tlnp"}'

curl http://localhost:8000/state
```

Each reset returns an `episode_id`. Pass it to `/step` when running parallel
clients.

## Training

### Phase 1: SFT Warm Start

```bash
uv run python -m training.train_sft \
  --train dataset/sft_train.jsonl \
  --val dataset/sft_val.jsonl \
  --scenario-split official \
  --epochs 2 \
  --output checkpoints/sft
```

Use `--scenario-split all` only for ablations. The official split keeps held-out
scenario IDs out of the SFT training set.

### Phase 2: GRPO Against Live Environment

Recommended path for the hackathon evidence:

```bash
uv run python -m training.train_openenv_grpo \
  --model checkpoints/sft \
  --steps 200 \
  --num-generations 4 \
  --output checkpoints/grpo-openenv
```

To train against a deployed Space/server:

```bash
uv run python -m training.train_openenv_grpo \
  --model checkpoints/sft \
  --env-url https://YOUR-SPACE.hf.space \
  --steps 200 \
  --output checkpoints/grpo-openenv
```

The older `training.train_grpo` script remains as a fallback custom rollout
loop. The submission should prefer `training.train_openenv_grpo`.

## Evaluation

```bash
uv run python -m training.evaluate \
  --baseline random \
  --trained checkpoints/grpo-openenv \
  --split val \
  --episodes 3 \
  --output results/
```

Remote evaluation:

```bash
uv run python -m training.evaluate \
  --baseline random \
  --trained checkpoints/grpo-openenv \
  --env-url https://YOUR-SPACE.hf.space \
  --split val \
  --episodes 3 \
  --output results/
```

Expected output files:

- `results/success_rate.png`
- `results/avg_reward.png`
- `results/commands_to_fix.png`
- `results/baseline_vs_trained.png`
- `results/per_scenario.png`
- `results/eval_results.json`

Commit the final plots and metrics before submission.

## Demo

```bash
uv run python -m training.demo \
  --scenario port_bound \
  --trained checkpoints/grpo-openenv \
  --no-pause
```

The strongest video is a side-by-side story: random/base agent loops or restarts
blindly; trained agent checks status/logs, identifies root cause, fixes, and
verifies.

## Hugging Face Space

The Dockerfile is configured for the HF Spaces default port (`7860`):

```bash
docker build -t sysadmin-game:latest .
docker run -p 7860:7860 sysadmin-game:latest
```

For OpenEnv deployment, validate and push from the repo root after installing
the latest OpenEnv CLI:

```bash
openenv validate --verbose
openenv push --repo-id YOUR_USER/sysadmin-game
```

Note: the full local training environment uses Docker to create fresh Ubuntu
systemd sandboxes. If the deployment target does not provide a Docker daemon,
use the Space as the discoverable API/demo surface and run training/evaluation
on local or cloud GPU machines with Docker enabled.

## Reward

| Condition | Reward |
|---|---:|
| Incident fixed (`check_fixed()` true) | `+1.0` |
| Command cost | `-0.01` |
| First useful diagnostic command | `+0.1` |
| Destructive command | `-0.5` and terminate |

## Results To Report

Before submission, replace this section with real numbers:

| Metric | Random/Base | SFT | SFT + GRPO |
|---|---:|---:|---:|
| Held-out success rate | TODO | TODO | TODO |
| Average reward | TODO | TODO | TODO |
| Commands to fix | TODO | TODO | TODO |
| Unsafe command rate | TODO | TODO | TODO |

Embed the final plots here with one-line captions so judges can understand the
improvement in under a minute.

## License

MIT
