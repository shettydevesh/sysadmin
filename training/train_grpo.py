#!/usr/bin/env python3
"""GRPO training script for reinforcement learning with live environment feedback."""

import argparse
import re
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_agent_response(response: str) -> tuple[str | None, str | None]:
    """Extract thinking and bash command from agent response.

    Returns:
        Tuple of (thinking, command) - either can be None
    """
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    bash_match = re.search(r"<bash>(.*?)</bash>", response, re.DOTALL)

    thinking = think_match.group(1).strip() if think_match else None
    command = bash_match.group(1).strip() if bash_match else None

    return thinking, command


def main():
    parser = argparse.ArgumentParser(description="GRPO training for Sysadmin agent")
    parser.add_argument("--sft-checkpoint", type=str, default="checkpoints/sft",
                        help="Path to SFT checkpoint")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct",
                        help="Base model (if no SFT checkpoint)")
    parser.add_argument("--output", type=str, default="checkpoints/grpo",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (number of episodes per step)")
    parser.add_argument("--group-size", type=int, default=4,
                        help="GRPO group size")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max turns per episode")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--wandb-project", type=str, default="sysadmin-game",
                        help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Save checkpoint every N steps")
    args = parser.parse_args()

    # Import dependencies
    try:
        import torch
        from datasets import Dataset

        # Try Unsloth first (GPU), fall back to transformers (CPU/MPS)
        USE_UNSLOTH = False
        try:
            from unsloth import FastLanguageModel
            USE_UNSLOTH = True
            print("Using Unsloth (GPU mode)")
        except (ImportError, NotImplementedError):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("Using Transformers (CPU/MPS mode - slower)")

    except ImportError as e:
        print(f"Error importing training libraries: {e}")
        print("Install with: uv sync --extra train")
        return

    # Import environment
    try:
        from sysadmin_env.environment import SysadminEnv
        from sysadmin_env.models import Action
    except ImportError as e:
        print(f"Error importing sysadmin_env: {e}")
        print("Make sure you're in the project root directory")
        return

    # Initialize W&B
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name="grpo-training", config=vars(args))
        except Exception as e:
            print(f"W&B init failed: {e}, continuing without logging")
            args.no_wandb = True

    # Load model
    model_path = args.sft_checkpoint if Path(args.sft_checkpoint).exists() else args.model
    print(f"Loading model from: {model_path}")

    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Add LoRA if loading base model
        if not Path(args.sft_checkpoint).exists():
            print("No SFT checkpoint found, adding LoRA adapters to base model")
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
    else:
        # CPU/MPS mode with transformers
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # System prompt for the agent
    SYSTEM_PROMPT = """You are an expert SRE agent that diagnoses and fixes Linux system issues.

When given a problem:
1. Think step-by-step about the diagnosis in <think> tags
2. Run ONE command at a time in <bash> tags
3. Analyze the output before running the next command
4. Continue until the issue is fixed

Example response:
<think>
The user reports nginx won't start. First, let me check the service status to see the error.
</think>
<bash>systemctl status nginx --no-pager -l</bash>

After seeing output, respond with your next diagnostic step or fix."""

    def run_episode(env: SysadminEnv, model, tokenizer, max_turns: int) -> dict:
        """Run a single episode and return trajectory data."""
        obs = env.reset()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs.output},
        ]

        trajectory = {
            "prompts": [],
            "responses": [],
            "rewards": [],
        }

        total_reward = 0.0

        for turn in range(max_turns):
            # Format prompt
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Generate response
            if USE_UNSLOTH:
                FastLanguageModel.for_inference(model)
            model.eval()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            # Parse response
            thinking, command = parse_agent_response(response)

            # Store trajectory
            trajectory["prompts"].append(prompt)
            trajectory["responses"].append(response)

            # If no command, episode ends with penalty
            if not command:
                trajectory["rewards"].append(-0.1)
                total_reward -= 0.1
                break

            # Execute command
            action = Action(command=command)
            obs = env.step(action)

            trajectory["rewards"].append(obs.reward)
            total_reward += obs.reward

            # Add to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "tool", "content": f"<output>\n{obs.output}\n</output>"})

            if obs.done:
                break

        trajectory["total_reward"] = total_reward
        trajectory["fixed"] = obs.metadata.get("fixed", False)
        trajectory["commands"] = len(trajectory["rewards"])

        return trajectory

    def collect_trajectories(model, tokenizer, num_episodes: int, max_turns: int) -> list[dict]:
        """Collect multiple episode trajectories."""
        trajectories = []

        with SysadminEnv() as env:
            for i in range(num_episodes):
                try:
                    traj = run_episode(env, model, tokenizer, max_turns)
                    trajectories.append(traj)
                    print(f"  Episode {i+1}: reward={traj['total_reward']:.2f}, "
                          f"fixed={traj['fixed']}, commands={traj['commands']}")
                except Exception as e:
                    print(f"  Episode {i+1} failed: {e}")
                    continue

        return trajectories

    def compute_grpo_advantages(trajectories: list[dict], group_size: int) -> list[dict]:
        """Compute GRPO advantages for a group of trajectories."""
        if len(trajectories) < group_size:
            return []

        # Group trajectories and compute relative advantages
        processed = []

        for i in range(0, len(trajectories) - group_size + 1, group_size):
            group = trajectories[i:i + group_size]
            rewards = [t["total_reward"] for t in group]

            # Compute mean and std for the group
            mean_reward = sum(rewards) / len(rewards)
            std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
            std_reward = max(std_reward, 1e-6)  # Avoid division by zero

            # Normalize advantages within group
            for j, traj in enumerate(group):
                advantage = (traj["total_reward"] - mean_reward) / std_reward

                # Create training examples from trajectory
                for k, (prompt, response, reward) in enumerate(zip(
                    traj["prompts"], traj["responses"], traj["rewards"]
                )):
                    processed.append({
                        "prompt": prompt,
                        "response": response,
                        "reward": reward,
                        "advantage": advantage,
                    })

        return processed

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Training loop
    print(f"\nStarting GRPO training for {args.steps} steps")
    print(f"Batch size: {args.batch_size}, Group size: {args.group_size}")
    print("=" * 50)

    all_rewards = []
    all_fix_rates = []

    for step in range(args.steps):
        print(f"\nStep {step + 1}/{args.steps}")

        # Collect trajectories
        trajectories = collect_trajectories(
            model, tokenizer,
            num_episodes=args.batch_size * args.group_size,
            max_turns=args.max_turns,
        )

        if not trajectories:
            print("  No successful trajectories, skipping step")
            continue

        # Compute metrics
        rewards = [t["total_reward"] for t in trajectories]
        fix_rate = sum(1 for t in trajectories if t["fixed"]) / len(trajectories)
        avg_reward = sum(rewards) / len(rewards)

        all_rewards.append(avg_reward)
        all_fix_rates.append(fix_rate)

        print(f"  Avg reward: {avg_reward:.3f}, Fix rate: {fix_rate:.1%}")

        # Log to W&B
        if not args.no_wandb:
            import wandb
            wandb.log({
                "reward/mean": avg_reward,
                "reward/min": min(rewards),
                "reward/max": max(rewards),
                "success/fix_rate": fix_rate,
                "episode/avg_commands": sum(t["commands"] for t in trajectories) / len(trajectories),
                "step": step,
            })

        # Compute advantages and create training batch
        training_data = compute_grpo_advantages(trajectories, args.group_size)

        if not training_data:
            continue

        # Simple policy gradient update
        if USE_UNSLOTH:
            FastLanguageModel.for_training(model)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        total_loss = 0.0
        for example in training_data:
            inputs = tokenizer(
                example["prompt"] + example["response"],
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)

            outputs = model(**inputs, labels=inputs.input_ids)

            # Weight loss by advantage
            loss = outputs.loss * example["advantage"]

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            total_loss += loss.item()

        # Gradient step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        avg_loss = total_loss / max(len(training_data), 1)
        print(f"  Loss: {avg_loss:.4f}")

        if not args.no_wandb:
            wandb.log({"train/loss": avg_loss, "step": step})

        # Save checkpoint
        if (step + 1) % args.save_steps == 0:
            checkpoint_path = f"{args.output}/step_{step + 1}"
            print(f"  Saving checkpoint to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

    # Save final model
    print(f"\nSaving final model to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Plot training curves
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Reward curve
        axes[0].plot(all_rewards, label="Average Reward")
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Reward")
        axes[0].set_title("GRPO Training - Reward Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Fix rate curve
        axes[1].plot(all_fix_rates, label="Fix Rate", color="green")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Fix Rate")
        axes[1].set_title("GRPO Training - Success Rate")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/grpo_training_curves.png", dpi=150)
        print("Saved training curves to results/grpo_training_curves.png")

    except Exception as e:
        print(f"Could not save plots: {e}")

    print("\nGRPO training complete!")


if __name__ == "__main__":
    main()
