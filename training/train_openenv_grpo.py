#!/usr/bin/env python3
"""GRPO training through TRL's OpenEnv environment_factory integration.

Use this script for the official hackathon training evidence. It connects the
trainer to live Sysadmin Game episodes instead of replaying a static dataset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_prompt_dataset(scenario_ids: list[str], repeats: int, start_seed: int):
    """Build a tiny routing dataset for TRL/OpenEnv rollouts."""
    from datasets import Dataset
    from sysadmin_env.openenv_adapter import SYSADMIN_RL_PROMPT

    rows = []
    seed = start_seed
    for _ in range(repeats):
        for scenario_id in scenario_ids:
            rows.append({
                "prompt": [{"role": "user", "content": SYSADMIN_RL_PROMPT}],
                "scenario_id": scenario_id,
                "seed": seed,
            })
            seed += 1
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Train Sysadmin Game with TRL GRPO + OpenEnv")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct",
                        help="Base model or SFT checkpoint")
    parser.add_argument("--output", default="checkpoints/grpo-openenv",
                        help="Output checkpoint directory")
    parser.add_argument("--env-url", default=None,
                        help="Remote Sysadmin Game server/HF Space URL. Omit for local Docker env.")
    parser.add_argument("--steps", type=int, default=200,
                        help="Maximum GRPO optimizer steps")
    parser.add_argument("--repeats", type=int, default=200,
                        help="Prompt rows per train scenario")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="GRPO group size / completions per prompt")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max-completion-length", type=int, default=2048,
                        help="Total multi-turn completion token budget")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Scenario IDs to train on. Defaults to official train split.")
    parser.add_argument("--seed", type=int, default=1000,
                        help="Starting seed for routed episodes")
    parser.add_argument("--wandb-project", default="sysadmin-game",
                        help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: uv sync --extra train") from exc

    from sysadmin_env.openenv_adapter import SysadminToolEnv, sysadmin_reward
    from sysadmin_env.scenarios import TRAIN_SCENARIO_IDS

    if args.env_url:
        os.environ["SYSADMIN_ENV_URL"] = args.env_url.rstrip("/")

    scenario_ids = args.scenarios or TRAIN_SCENARIO_IDS
    train_dataset = build_prompt_dataset(scenario_ids, args.repeats, args.seed)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    report_to = "none" if args.no_wandb else "wandb"
    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    grpo_args = GRPOConfig(
        output_dir=args.output,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=1,
        save_steps=max(args.steps // 4, 25),
        report_to=report_to,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=sysadmin_reward,
        environment_factory=SysadminToolEnv,
    )
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
