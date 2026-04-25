"""Reward calculation for the Sysadmin Game environment."""

import re
from typing import Set

# Diagnostic commands that earn bonus on first use
# Pattern prefix ensures we match command names, not filenames
_CMD = r"(?:^|[|;&]\s*)"  # Start of string or after pipe/semicolon/ampersand

DIAGNOSTIC_COMMANDS = {
    "systemctl": r"\bsystemctl\s+(status|is-active|is-enabled|list-units)",
    "journalctl": _CMD + r"journalctl\b",
    "df": _CMD + r"df\b",
    "ls": _CMD + r"ls\b",
    "cat": _CMD + r"cat\b",
    "ps": _CMD + r"ps\b",
    "ss": _CMD + r"ss\b",
    "netstat": _CMD + r"netstat\b",
    "grep": r"\|\s*grep\b|" + _CMD + r"grep\b",  # grep often after pipe
    "head": r"\|\s*head\b|" + _CMD + r"head\b",
    "tail": r"\|\s*tail\b|" + _CMD + r"tail\b",
    "less": r"\|\s*less\b|" + _CMD + r"less\b",
    "find": _CMD + r"find\b",
    "lsof": _CMD + r"lsof\b",
    "top": _CMD + r"top\b",
    "htop": _CMD + r"htop\b",
    "free": _CMD + r"free\b",
    "du": _CMD + r"du\b",
    "stat": _CMD + r"stat\s",  # stat requires argument
    "file": _CMD + r"file\s",  # file requires argument
    "which": _CMD + r"which\b",
    "whereis": _CMD + r"whereis\b",
    "id": _CMD + r"id\b",
    "whoami": _CMD + r"whoami\b",
}

# Rewards
REWARD_FIXED = 1.0
REWARD_PER_COMMAND = -0.01
REWARD_DIAGNOSTIC_FIRST_USE = 0.1
REWARD_DESTRUCTIVE = -0.5


def get_diagnostic_used(command: str) -> str | None:
    """Check if a command uses a diagnostic tool.

    Args:
        command: Shell command to check

    Returns:
        Name of diagnostic tool used, or None
    """
    for name, pattern in DIAGNOSTIC_COMMANDS.items():
        if re.search(pattern, command):
            return name
    return None


def calculate_reward(
    command: str,
    fixed: bool,
    diagnostics_used: Set[str],
    is_destructive: bool
) -> tuple[float, Set[str]]:
    """Calculate reward for a command execution.

    Args:
        command: The shell command executed
        fixed: Whether the issue is now fixed
        diagnostics_used: Set of diagnostics already used this episode
        is_destructive: Whether the command was destructive

    Returns:
        Tuple of (reward, updated diagnostics_used set)
    """
    reward = 0.0
    new_diagnostics = diagnostics_used.copy()

    # Destructive command penalty
    if is_destructive:
        return REWARD_DESTRUCTIVE, new_diagnostics

    # Per-command cost
    reward += REWARD_PER_COMMAND

    # Check for diagnostic command bonus
    diagnostic = get_diagnostic_used(command)
    if diagnostic and diagnostic not in diagnostics_used:
        reward += REWARD_DIAGNOSTIC_FIRST_USE
        new_diagnostics.add(diagnostic)

    # Fixed bonus
    if fixed:
        reward += REWARD_FIXED

    return reward, new_diagnostics
