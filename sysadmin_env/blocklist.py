"""Destructive command detection for safety."""

import re

# Patterns that indicate destructive commands
DESTRUCTIVE_PATTERNS = [
    # Recursive delete of root or important directories
    # Must match exactly / or /* or /etc etc, not /tmp or /opt
    r"rm\s+(-[rf]+\s+)*(/\*|/etc|/var|/usr|/home|/boot|/bin|/sbin|/lib)($|\s|/)",
    r"rm\s+(-[rf]+\s+)*/($|\s)",  # rm / or rm -rf /
    r"rm\s+-[rf]*\s+--no-preserve-root",

    # Direct disk writes
    r"dd\s+.*of=/dev/[sh]d[a-z]",
    r"dd\s+.*of=/dev/nvme",

    # Filesystem formatting
    r"mkfs\.",
    r"mkswap\s+/dev/",

    # Fork bomb
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:",

    # Dangerous permission changes
    r"chmod\s+(-R\s+)?[0-7]*0+\s+/($|\s)",
    r"chmod\s+-R\s+000\s+/",

    # Wipe commands
    r"shred\s+.*(/dev/|/etc|/var|/home)",
    r"wipefs",

    # Kernel panic / system halt in destructive way
    r"echo\s+['\"]?c['\"]?\s*>\s*/proc/sysrq-trigger",

    # Overwrite critical files
    r">\s*/etc/passwd",
    r">\s*/etc/shadow",
    r"truncate\s+.*(/etc|/var|/usr)",

    # Kill all processes
    r"kill\s+-9\s+-1",
    r"killall\s+-9\s+",
    r"pkill\s+-9\s+\.",
]

# Compiled patterns for efficiency
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DESTRUCTIVE_PATTERNS]


def is_destructive(command: str) -> bool:
    """Check if a command is destructive.

    Args:
        command: Shell command to check

    Returns:
        True if the command matches a destructive pattern
    """
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(command):
            return True
    return False
