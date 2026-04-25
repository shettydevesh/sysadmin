"""Disk full alt scenario: Disk filled via syslog."""

from .base import BaseScenario
from sysadmin_env.sandbox import DockerSandbox


class DiskFullAltScenario(BaseScenario):
    """Scenario where disk is full due to runaway syslog."""

    @property
    def id(self) -> str:
        return "disk_full_alt"

    @property
    def category(self) -> str:
        return "Disk"

    def get_complaint(self) -> str:
        return "Disk is full again! We cleared it yesterday but something keeps filling it up. Apps are crashing."

    def setup(self, sandbox: DockerSandbox) -> None:
        sandbox.exec("apt-get update && apt-get install -y rsyslog nginx >/dev/null 2>&1")

    def break_system(self, sandbox: DockerSandbox) -> None:
        self.setup(sandbox)

        # Create a large syslog file
        sandbox.exec("fallocate -l 700M /var/log/syslog 2>/dev/null || dd if=/dev/zero of=/var/log/syslog bs=1M count=700 2>/dev/null")

        # Also create rotated logs
        sandbox.exec("fallocate -l 100M /var/log/syslog.1 2>/dev/null")

        sandbox.exec("systemctl stop nginx")

    def check_fixed(self, sandbox: DockerSandbox) -> bool:
        # Check disk usage - should be under 90%
        result = sandbox.exec("df -h / | tail -1 | awk '{print $5}' | tr -d '%'")
        try:
            usage = int(result.strip())
            return usage < 90
        except ValueError:
            return False
