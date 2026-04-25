"""Tests for reward calculation."""

import pytest
from sysadmin_env.reward import (
    calculate_reward,
    get_diagnostic_used,
    REWARD_FIXED,
    REWARD_PER_COMMAND,
    REWARD_DIAGNOSTIC_FIRST_USE,
    REWARD_DESTRUCTIVE,
)


class TestDiagnosticDetection:
    """Test diagnostic command detection."""

    @pytest.mark.parametrize("command,expected", [
        ("systemctl status nginx", "systemctl"),
        ("systemctl is-active nginx", "systemctl"),
        ("journalctl -xe", "journalctl"),
        ("df -h", "df"),
        ("ls -la /etc/nginx", "ls"),
        ("cat /etc/nginx/nginx.conf", "cat"),
        ("ps aux", "ps"),
        ("ss -tlnp", "ss"),
        ("netstat -tlnp", "netstat"),
        ("grep error /var/log/syslog", "grep"),
        ("head -n 100 /var/log/nginx/error.log", "head"),
        ("tail -f /var/log/syslog", "tail"),
        ("lsof -i :80", "lsof"),
        ("du -sh /var/log", "du"),
        ("free -m", "free"),
    ])
    def test_detects_diagnostics(self, command, expected):
        """Verify diagnostic commands are detected."""
        assert get_diagnostic_used(command) == expected

    @pytest.mark.parametrize("command", [
        "apt-get install nginx",
        "systemctl restart nginx",
        "chown root:root /etc/nginx/nginx.conf",
        "rm /tmp/file.txt",
        "echo hello",
    ])
    def test_non_diagnostics(self, command):
        """Verify non-diagnostic commands return None."""
        assert get_diagnostic_used(command) is None


class TestRewardCalculation:
    """Test reward calculation logic."""

    def test_per_command_cost(self):
        """Every command costs."""
        reward, _ = calculate_reward("echo hello", False, set(), False)
        assert reward == REWARD_PER_COMMAND

    def test_diagnostic_first_use_bonus(self):
        """First use of diagnostic gets bonus."""
        reward, diagnostics = calculate_reward("df -h", False, set(), False)
        assert reward == REWARD_PER_COMMAND + REWARD_DIAGNOSTIC_FIRST_USE
        assert "df" in diagnostics

    def test_diagnostic_repeat_no_bonus(self):
        """Repeat diagnostic use gets no bonus."""
        reward, diagnostics = calculate_reward("df -h", False, {"df"}, False)
        assert reward == REWARD_PER_COMMAND
        assert diagnostics == {"df"}

    def test_fixed_bonus(self):
        """Fixing the issue gives big reward."""
        reward, _ = calculate_reward("chown root /etc/nginx/nginx.conf", True, set(), False)
        assert reward == REWARD_PER_COMMAND + REWARD_FIXED

    def test_fixed_with_diagnostic(self):
        """Fix command that is also diagnostic."""
        reward, diags = calculate_reward("systemctl status nginx", True, set(), False)
        assert reward == REWARD_PER_COMMAND + REWARD_FIXED + REWARD_DIAGNOSTIC_FIRST_USE
        assert "systemctl" in diags

    def test_destructive_penalty(self):
        """Destructive commands get penalized."""
        reward, _ = calculate_reward("rm -rf /", False, set(), True)
        assert reward == REWARD_DESTRUCTIVE

    def test_destructive_no_other_rewards(self):
        """Destructive commands don't get other rewards."""
        reward, diagnostics = calculate_reward("rm -rf /", True, set(), True)
        assert reward == REWARD_DESTRUCTIVE
        assert len(diagnostics) == 0
