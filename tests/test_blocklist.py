"""Tests for destructive command detection."""

import pytest
from sysadmin_env.blocklist import is_destructive


class TestBlocklist:
    """Test the destructive command blocklist."""

    @pytest.mark.parametrize("command", [
        "rm -rf /",
        "rm -rf /*",
        "rm -rf /etc",
        "rm -rf /var",
        "rm -rf /usr",
        "rm -rf /home",
        "rm -rf --no-preserve-root /",
        "dd of=/dev/sda",
        "dd if=/dev/zero of=/dev/nvme0n1",
        "mkfs.ext4 /dev/sda1",
        "mkswap /dev/sda2",
        ":(){ :|:& };:",
        "chmod -R 000 /",
        "shred /dev/sda",
        "echo c > /proc/sysrq-trigger",
        "> /etc/passwd",
        "truncate -s 0 /etc/shadow",
        "kill -9 -1",
    ])
    def test_detects_destructive_commands(self, command):
        """Verify destructive commands are detected."""
        assert is_destructive(command) is True, f"Should detect: {command}"

    @pytest.mark.parametrize("command", [
        "ls -la",
        "systemctl status nginx",
        "journalctl -xe",
        "df -h",
        "cat /etc/nginx/nginx.conf",
        "ps aux",
        "netstat -tlnp",
        "ss -tlnp",
        "grep error /var/log/syslog",
        "chmod 644 /etc/nginx/nginx.conf",
        "chown root:root /etc/nginx/nginx.conf",
        "rm /tmp/myfile.txt",
        "rm -rf /tmp/mydir",
        "apt-get install nginx",
        "systemctl restart nginx",
        "kill 12345",
        "pkill nginx",
    ])
    def test_allows_safe_commands(self, command):
        """Verify safe commands are allowed."""
        assert is_destructive(command) is False, f"Should allow: {command}"
