# Sysadmin Game

**An OpenEnv-compliant RL environment that trains LLMs to diagnose and fix real broken Linux servers.**

The reward is not an AI judge. It's not a human label. It's whether the service is actually running.

---

## Links

| Artifact | Link |
|---|---|
| 🤗 HuggingFace Space | [deveshshetty/sysadmin-game](https://huggingface.co/spaces/deveshshetty/sysadmin-game) |
| 🧠 Trained model checkpoint | [deveshshetty/sysadmin-grpo](https://huggingface.co/deveshshetty/sysadmin-grpo) |
| 📓 SFT Colab notebook | [train_sysadmin.ipynb](https://colab.research.google.com/github/shettydevesh/sysadmin/blob/main/notebooks/train_sysadmin.ipynb) |
| 📓 GRPO Colab notebook | [train_grpo.ipynb](https://colab.research.google.com/github/shettydevesh/sysadmin/blob/main/notebooks/train_grpo.ipynb) |
| 📝 Mini-blog | [Read on HuggingFace](https://huggingface.co/spaces/deveshshetty/sysadmin-game) |
| 🎬 Demo video | *coming soon* |

---

## The Problem

Most LLM sysadmin demos are fake. The model suggests a command, a human decides if it sounds right, and nobody actually runs it on a broken server.

LLMs hallucinate file paths. They suggest fixes before reading the error. Ask a chatbot to fix your broken web server and you'll get something that sounds plausible and doesn't work.

We wanted to fix that. So we built an AI a video game.

---

## The Game

A fresh Ubuntu 22.04 container starts. We deliberately break it — fill the disk, corrupt a config, steal a port, kill a cert. We hand the LLM a one-line complaint and a shell.

```
User:    "nginx won't start. Says 'Address already in use'."

Agent:   <think>Something else has port 80. Find it.</think>
         <bash>ss -tlnp | grep ':80'</bash>

Output:  LISTEN 0 511 0.0.0.0:80  users:(("apache2",pid=288))

Agent:   <bash>systemctl stop apache2 && systemctl start nginx</bash>

Result:  ✅ fixed: true  reward: +0.99  commands: 2
```

The AI types real commands. The shell runs them inside a real container. If nginx serves on port 80, the episode is fixed. If not, it's not. No judge model. No human review. No rubric. The box is fixed or it isn't.

---

## Training Results

> Qwen2.5-Coder-0.5B-Instruct · T4 GPU · 50 steps · 4 episodes/step · real Docker environment

### Reward — 0.25 → 0.65 over 50 steps

*Average episode reward per step. Each step = 4 live episodes against real broken Ubuntu containers.*

![Average Reward](https://d2qt0ksb47ks9g.cloudfront.net/1777195101830_reward.png)

### Success Rate — 6% → 65%

*Percentage of episodes where the model actually fixed the service. Starts nearly random, climbs consistently.*

![Success Rate](https://d2qt0ksb47ks9g.cloudfront.net/1777195082383_success_rate.png)

### Commands to Fix — 23 → 6

*Average commands needed to fix a scenario. The model learns to diagnose first instead of spamming restarts.*

![Commands to Fix](https://d2qt0ksb47ks9g.cloudfront.net/1777195108774_commands.png)

### Baseline vs Trained — Same Scenarios, Same Broken Servers

*Untrained model vs GRPO-trained model on 4 held-out scenarios it never saw during training.*

![Baseline vs Trained](https://d2qt0ksb47ks9g.cloudfront.net/1777195115503_baseline_vs_trained.png)

### Live Training Log — The Model Learning in Real Time

*Step 1 → reward -0.12, fix rate 0%. Step 2 → reward +0.45, fix rate 50%. Each step is 2 live episodes against real broken containers.*

![Training Log](https://d2qt0ksb47ks9g.cloudfront.net/1777200223681_training_log.jpeg)

### Evaluation Numbers

| Metric | Untrained | GRPO Trained |
|---|---:|---:|
| Success rate (held-out) | ~8% | ~68% |
| Average reward per episode | 0.25 | 0.65 |
| Avg commands to fix | 23 | 7 |
| Improvement | — | **8.5× more successful, 3× more efficient** |

---

## Before vs After

Same scenario: "nginx won't start, address already in use."

**Untrained model — guesses without checking:**
```
<bash>cat /etc/nginx/nginx.conf</bash>     # config looks fine to it
<bash>systemctl restart nginx</bash>        # still fails
<bash>apt-get install --reinstall nginx</bash>  # still fails
... hits 25-command limit. Never fixed.
```

**Trained model — diagnoses, then fixes:**
```
<bash>ss -tlnp | grep ':80'</bash>          # apache2 is on port 80
<bash>systemctl stop apache2 && systemctl start nginx</bash>
→ fixed: true  reward: +0.99  in 2 commands
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  LLM  (Qwen2.5-Coder, T4 GPU on HF Spaces)          │
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

The LLM never touches Docker directly. It speaks HTTP. Any GPU machine — Colab, HF Spaces, a cloud VM — can run training by pointing at the server URL. This is how the Colab notebooks connect to the environment over ngrok.

---

## Reward Function

| Condition | Reward |
|---|---:|
| `check_fixed()` returns True | `+1.00` |
| First use of a diagnostic command | `+0.10` |
| Per command issued | `-0.01` |
| Destructive command (`rm -rf /`, fork bomb, disk wipe) | `-0.50` + terminate |

The diagnostic bonus (`df`, `ss`, `journalctl`, `ps`, `cat`, `ls`, `grep`...) exists to prevent reward hacking. Without it, the model figures out that blindly restarting services is the fastest path to +1.0. With it, inspect-first consistently outscores guess-and-restart.

---

## Scenarios

10 scenarios across 6 failure categories. 5 for training, 5 held out for evaluation. The model never sees held-out scenario IDs during training.

| Scenario | Category | What Breaks | How `check_fixed()` Verifies |
|---|---|---|---|
| `disk_full` | Disk | `fallocate -l 800M /var/log/huge_debug.log` | File < 1MB |
| `disk_full_alt` | Disk | Fill `/tmp` with junk | Disk usage < 90% |
| `nginx_syntax` | Service | Bad directive in `nginx.conf` | `nginx -t` exits 0 |
| `nginx_unknown` | Service | Config points to missing module | nginx serves on port 80 |
| `ownership` | Permissions | `chown nobody:nogroup /etc/nginx/nginx.conf` | nginx starts |
| `port_bound` | Network | apache2 holds port 80 before nginx | nginx responds on 80 |
| `runaway_cpu` | Process | `yes > /dev/null` spinning in background | CPU load < 80% |
| `expired_cert` | TLS | Backdate cert by 2 years | `openssl verify` passes |
| `stale_pid` | Service | Fake PID file blocks systemd | Service running |
| `venv_broken` | Environment | Delete `site-packages` from active venv | `python -c "import flask"` works |

**Train split:** `disk_full` · `nginx_syntax` · `ownership` · `port_bound` · `runaway_cpu`

**Held-out eval:** `disk_full_alt` · `nginx_unknown` · `expired_cert` · `stale_pid` · `venv_broken`

---

## How We Trained

### Phase 1 — SFT Warm-Start

92 hand-written examples of correct sysadmin behavior. Each is a full conversation: complaint → diagnose → read output → fix → verify. Two epochs, ~30 minutes on a T4.

This teaches the model the format and the rhythm. Inspect first. One command at a time. Read the output before the next move.

**→ [Open SFT Colab](https://colab.research.google.com/github/shettydevesh/sysadmin/blob/main/notebooks/train_sysadmin.ipynb)**

### Phase 2 — GRPO Against Live Environment

The model plays the game itself. We run 4 episodes per step against real broken Docker containers. At the end of each group:

- Calculate average reward across the 4 runs
- Runs above average → reinforce those commands
- Runs below average → discourage them
- Update weights. Next step.

No gold labels. No AI judge. Just: which of your own attempts worked better than average?

> **Note:** The GRPO Colab uses the SFT checkpoint as a warm-start base. This is optional — the GRPO notebook works fine starting from the base model if you don't have an SFT checkpoint.

**→ [Open GRPO Colab](https://colab.research.google.com/github/shettydevesh/sysadmin/blob/main/notebooks/train_grpo.ipynb)**

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/shettydevesh/sysadmin.git
cd sysadmin
pip install -e ".[dev]"

# 2. Build the Ubuntu 22.04 + systemd sandbox image
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

## API Reference

| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/health` | Server status, active episodes, Docker container count |
| `GET` | `/scenarios` | All scenarios with train/val split |
| `POST` | `/reset` | Start new episode. Body: `{"scenario_id": "port_bound", "seed": 42}` |
| `POST` | `/step` | Run command. Body: `{"command": "ss -tlnp", "episode_id": "..."}` |
| `GET` | `/state` | Episode state: steps, reward, diagnostics used, elapsed time |
| `DELETE` | `/episodes/{id}` | Close and clean up a sandbox |

Multiple episodes run in parallel. Each `/reset` returns a unique `episode_id`.

---

## OpenEnv Compliance

| File | Role |
|---|---|
| `openenv.yaml` | Manifest for environment discovery |
| `sysadmin_env/models.py` | Typed `Action`, `Observation`, `State` schemas |
| `sysadmin_env/client.py` | HTTP client for remote connection |
| `sysadmin_env/server/app.py` | FastAPI server — the main deployment target |
| `sysadmin_env/openenv_adapter.py` | TRL `environment_factory` adapter exposing `run_shell` tool |

---

## Project Structure

```
sysadmin_env/
├── scenarios/             # 10 break/check Python modules
├── sandbox.py             # Docker management + auto-cleanup
├── environment.py         # reset / step / state loop
├── reward.py              # reward calculation
├── blocklist.py           # destructive command detection
├── client.py              # HTTP client
└── server/app.py          # FastAPI server

training/
├── train_sft.py           # Phase 1: SFT warm-start
├── train_openenv_grpo.py  # Phase 2: GRPO against live env
└── evaluate.py            # held-out evaluation + plots

notebooks/
├── train_sysadmin.ipynb   # SFT Colab notebook
└── train_grpo.ipynb       # GRPO Colab notebook

dataset/
├── sft_train.jsonl        # 92 training examples
└── sft_val.jsonl          # 10 validation examples

results/
├── reward.png
├── success_rate.png
├── commands.png
└── baseline_vs_trained.png

spaces/app.py              # Gradio UI for HF Spaces training
docker/sandbox.Dockerfile  # Ubuntu 22.04 + systemd image
```

---

## Why This Matters

Most AI is trained on text that already exists. The model gets good at producing text that resembles what it saw. Useful, but it has a ceiling.

What we're doing is different. The AI gets better by trying things and seeing what happens. The training signal isn't "does this look right?" but "did this work?" That's a fundamentally different kind of feedback, and one that keeps scaling as long as we have problems with verifiable answers.

Servers are a good first test because the answer is so unambiguous. But the same idea works for anything with a clear success condition. Did the unit test pass? Did the program compile? Did the equation come out right?

Our project is a small bet on a bigger idea. The next leap in useful AI probably won't come from feeding bigger models more text. It'll come from putting them in environments where their actions have consequences.

---

## License

MIT
