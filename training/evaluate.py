#!/usr/bin/env python3
"""Evaluation script comparing baseline vs trained models."""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EpisodeResult:
    """Result of a single episode."""
    scenario_id: str
    fixed: bool
    total_reward: float
    commands_used: int
    time_taken: float
    termination_reason: str
    trajectory: list[dict]


@dataclass
class EvalResults:
    """Aggregated evaluation results."""
    model_name: str
    episodes: list[EpisodeResult]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.fixed) / len(self.episodes)

    @property
    def avg_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.total_reward for e in self.episodes) / len(self.episodes)

    @property
    def avg_commands(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.commands_used for e in self.episodes) / len(self.episodes)

    @property
    def avg_commands_when_fixed(self) -> float:
        fixed = [e for e in self.episodes if e.fixed]
        if not fixed:
            return 0.0
        return sum(e.commands_used for e in fixed) / len(fixed)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "avg_commands": self.avg_commands,
            "avg_commands_when_fixed": self.avg_commands_when_fixed,
            "num_episodes": len(self.episodes),
            "episodes": [asdict(e) for e in self.episodes],
        }


def run_episode(env, agent, scenario_id: Optional[str] = None) -> EpisodeResult:
    """Run a single evaluation episode.

    Args:
        env: The SysadminEnv
        agent: The agent (SysadminAgent or RandomAgent)
        scenario_id: Optional specific scenario to run

    Returns:
        EpisodeResult with the episode data
    """
    from sysadmin_env import Action

    start_time = time.time()
    trajectory = []

    # Reset environment and agent
    obs = env.reset(scenario_id=scenario_id)
    agent.reset()

    trajectory.append({
        "role": "system",
        "content": obs.output,
        "metadata": obs.metadata,
    })

    # Run episode
    while not obs.done:
        # Get agent action
        command, thinking = agent.get_action(obs.output)

        trajectory.append({
            "role": "assistant",
            "thinking": thinking,
            "command": command,
        })

        # Execute action
        obs = env.step(Action(command=command))

        trajectory.append({
            "role": "environment",
            "output": obs.output[:500],  # Truncate for storage
            "reward": obs.reward,
            "done": obs.done,
            "metadata": obs.metadata,
        })

    elapsed = time.time() - start_time

    return EpisodeResult(
        scenario_id=obs.metadata.get("scenario_id", "unknown"),
        fixed=obs.metadata.get("fixed", False),
        total_reward=obs.metadata.get("total_reward", 0.0),
        commands_used=obs.metadata.get("command_count", 0),
        time_taken=elapsed,
        termination_reason=obs.metadata.get("termination_reason", "unknown"),
        trajectory=trajectory,
    )


def evaluate_agent(
    agent,
    scenario_ids: list[str],
    num_episodes_per_scenario: int = 1,
    model_name: str = "unknown",
) -> EvalResults:
    """Evaluate an agent across multiple scenarios.

    Args:
        agent: The agent to evaluate
        scenario_ids: List of scenario IDs to test
        num_episodes_per_scenario: How many times to run each scenario
        model_name: Name for logging

    Returns:
        EvalResults with all episode data
    """
    from sysadmin_env import SysadminEnv

    print(f"\nEvaluating {model_name} on {len(scenario_ids)} scenarios...")
    episodes = []

    with SysadminEnv() as env:
        for scenario_id in scenario_ids:
            for i in range(num_episodes_per_scenario):
                print(f"  Running {scenario_id} ({i+1}/{num_episodes_per_scenario})...", end=" ")
                try:
                    result = run_episode(env, agent, scenario_id=scenario_id)
                    episodes.append(result)
                    status = "✓ Fixed" if result.fixed else "✗ Failed"
                    print(f"{status} (reward={result.total_reward:.2f}, cmds={result.commands_used})")
                except Exception as e:
                    print(f"Error: {e}")

    return EvalResults(model_name=model_name, episodes=episodes)


def plot_comparison(baseline: EvalResults, trained: EvalResults, output_dir: str):
    """Generate comparison plots.

    Args:
        baseline: Results from baseline model
        trained: Results from trained model
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Success Rate Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    models = [baseline.model_name, trained.model_name]
    success_rates = [baseline.success_rate * 100, trained.success_rate * 100]
    colors = ["#ff6b6b", "#4ecdc4"]

    bars = ax.bar(models, success_rates, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Scenario Fix Success Rate", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)

    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path / "success_rate.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'success_rate.png'}")

    # 2. Average Reward
    fig, ax = plt.subplots(figsize=(8, 6))
    rewards = [baseline.avg_reward, trained.avg_reward]

    bars = ax.bar(models, rewards, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.set_title("Average Episode Reward", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    for bar, reward in zip(bars, rewards):
        y_pos = bar.get_height() + 0.02 if reward >= 0 else bar.get_height() - 0.05
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{reward:.3f}", ha="center", va="bottom" if reward >= 0 else "top",
                fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path / "avg_reward.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'avg_reward.png'}")

    # 3. Commands to Fix (only for successful episodes)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmds = [baseline.avg_commands_when_fixed, trained.avg_commands_when_fixed]

    bars = ax.bar(models, cmds, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Average Commands", fontsize=12)
    ax.set_title("Commands to Fix (Successful Episodes)", fontsize=14, fontweight="bold")

    for bar, cmd in zip(bars, cmds):
        if cmd > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{cmd:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path / "commands_to_fix.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'commands_to_fix.png'}")

    # 4. Per-Scenario Comparison
    scenarios = list(set(e.scenario_id for e in baseline.episodes + trained.episodes))
    scenarios.sort()

    if len(scenarios) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(scenarios))
        width = 0.35

        baseline_rates = []
        trained_rates = []
        for s in scenarios:
            b_eps = [e for e in baseline.episodes if e.scenario_id == s]
            t_eps = [e for e in trained.episodes if e.scenario_id == s]
            baseline_rates.append(sum(1 for e in b_eps if e.fixed) / len(b_eps) * 100 if b_eps else 0)
            trained_rates.append(sum(1 for e in t_eps if e.fixed) / len(t_eps) * 100 if t_eps else 0)

        ax.bar(x - width/2, baseline_rates, width, label=baseline.model_name, color=colors[0])
        ax.bar(x + width/2, trained_rates, width, label=trained.model_name, color=colors[1])

        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title("Per-Scenario Success Rate", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(output_path / "per_scenario.png", dpi=150)
        plt.close()
        print(f"Saved: {output_path / 'per_scenario.png'}")

    # 5. Combined comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Success Rate
    axes[0].bar(models, success_rates, color=colors, edgecolor="black")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(0, 100)

    # Reward
    axes[1].bar(models, rewards, color=colors, edgecolor="black")
    axes[1].set_ylabel("Average Reward")
    axes[1].set_title("Average Reward")
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Commands
    axes[2].bar(models, cmds, color=colors, edgecolor="black")
    axes[2].set_ylabel("Commands")
    axes[2].set_title("Commands to Fix")

    plt.suptitle("Baseline vs Trained Model Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "baseline_vs_trained.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'baseline_vs_trained.png'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sysadmin agents")
    parser.add_argument("--baseline", type=str, default="random",
                        choices=["random", "base-qwen"],
                        help="Baseline model type")
    parser.add_argument("--trained", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--scenarios", type=str, nargs="+",
                        default=["ownership", "nginx_syntax", "disk_full", "port_bound"],
                        help="Scenarios to evaluate")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Episodes per scenario")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for plots and results")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Save full trajectories to JSON")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create baseline agent
    if args.baseline == "random":
        from training.agent import RandomAgent
        baseline_agent = RandomAgent()
        baseline_name = "Random Baseline"
    else:
        from training.agent import SysadminAgent, AgentConfig
        baseline_agent = SysadminAgent(AgentConfig())
        baseline_name = "Base Qwen"

    # Evaluate baseline
    baseline_results = evaluate_agent(
        baseline_agent,
        args.scenarios,
        num_episodes_per_scenario=args.episodes,
        model_name=baseline_name,
    )

    # Create trained agent if checkpoint provided
    if args.trained:
        from training.agent import load_trained_agent
        trained_agent = load_trained_agent(args.trained)
        trained_name = "Trained Model"
    else:
        # Use same baseline for demo (in real use, provide --trained)
        print("\nNo trained model provided. Using base Qwen for comparison demo.")
        from training.agent import SysadminAgent, AgentConfig
        trained_agent = SysadminAgent(AgentConfig())
        trained_name = "Base Qwen (no training)"

    # Evaluate trained
    trained_results = evaluate_agent(
        trained_agent,
        args.scenarios,
        num_episodes_per_scenario=args.episodes,
        model_name=trained_name,
    )

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\n{baseline_results.model_name}:")
    print(f"  Success Rate: {baseline_results.success_rate*100:.1f}%")
    print(f"  Avg Reward:   {baseline_results.avg_reward:.3f}")
    print(f"  Avg Commands: {baseline_results.avg_commands:.1f}")

    print(f"\n{trained_results.model_name}:")
    print(f"  Success Rate: {trained_results.success_rate*100:.1f}%")
    print(f"  Avg Reward:   {trained_results.avg_reward:.3f}")
    print(f"  Avg Commands: {trained_results.avg_commands:.1f}")

    improvement = (trained_results.success_rate - baseline_results.success_rate) * 100
    print(f"\nImprovement: {improvement:+.1f}% success rate")
    print("="*60)

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(baseline_results, trained_results, args.output)

    # Save results JSON
    results = {
        "baseline": baseline_results.to_dict(),
        "trained": trained_results.to_dict(),
    }
    with open(output_path / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {output_path / 'eval_results.json'}")

    if args.save_trajectories:
        with open(output_path / "trajectories.json", "w") as f:
            json.dump({
                "baseline": [asdict(e) for e in baseline_results.episodes],
                "trained": [asdict(e) for e in trained_results.episodes],
            }, f, indent=2, default=str)
        print(f"Saved: {output_path / 'trajectories.json'}")


if __name__ == "__main__":
    main()
