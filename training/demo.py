#!/usr/bin/env python3
"""Interactive demo showing trained vs untrained agent behavior."""

import argparse
from typing import Optional


def run_interactive_demo(
    scenario_id: str = "ownership",
    trained_checkpoint: Optional[str] = None,
    show_thinking: bool = True,
):
    """Run an interactive demo comparing agents.

    Args:
        scenario_id: Which scenario to demonstrate
        trained_checkpoint: Path to trained model (None = use base model)
        show_thinking: Whether to show agent reasoning
    """
    from sysadmin_env import SysadminEnv, Action
    from training.agent import RandomAgent, SysadminAgent, AgentConfig, load_trained_agent

    print("="*70)
    print("SYSADMIN GAME - Interactive Demo")
    print("="*70)

    # Create agents
    print("\nInitializing agents...")
    random_agent = RandomAgent()

    if trained_checkpoint:
        trained_agent = load_trained_agent(trained_checkpoint)
        trained_name = "Trained Qwen"
    else:
        trained_agent = SysadminAgent(AgentConfig())
        trained_name = "Base Qwen"

    # Run demo for each agent
    for agent, name in [(random_agent, "Random Baseline"), (trained_agent, trained_name)]:
        print(f"\n{'='*70}")
        print(f"Agent: {name}")
        print("="*70)

        with SysadminEnv() as env:
            # Reset
            obs = env.reset(scenario_id=scenario_id)
            agent.reset()

            print(f"\n📋 Scenario: {scenario_id}")
            print(f"💬 User complaint: {obs.output}")
            print("-"*50)

            step = 0
            while not obs.done:
                step += 1
                command, thinking = agent.get_action(obs.output)

                if show_thinking and thinking:
                    print(f"\n🧠 Thinking: {thinking[:200]}...")

                print(f"\n[Step {step}] $ {command}")

                obs = env.step(Action(command=command))

                # Show truncated output
                output_preview = obs.output[:300]
                if len(obs.output) > 300:
                    output_preview += "\n... (truncated)"
                print(output_preview)

                print(f"   Reward: {obs.reward:+.2f} | Total: {obs.metadata.get('total_reward', 0):.2f}")

            # Final result
            print("\n" + "="*50)
            if obs.metadata.get("fixed"):
                print("✅ FIXED!")
            else:
                print(f"❌ Failed ({obs.metadata.get('termination_reason', 'unknown')})")
            print(f"   Commands used: {obs.metadata.get('command_count', 0)}")
            print(f"   Total reward: {obs.metadata.get('total_reward', 0):.3f}")
            print("="*50)

        input("\nPress Enter to continue to next agent...")


def main():
    parser = argparse.ArgumentParser(description="Interactive sysadmin demo")
    parser.add_argument("--scenario", type=str, default="ownership",
                        help="Scenario to demonstrate")
    parser.add_argument("--trained", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Hide agent thinking")
    args = parser.parse_args()

    run_interactive_demo(
        scenario_id=args.scenario,
        trained_checkpoint=args.trained,
        show_thinking=not args.no_thinking,
    )


if __name__ == "__main__":
    main()
