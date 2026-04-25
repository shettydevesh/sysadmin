"""Disk full scenario: /var/log filled with large file."""

from .base import BaseScenario
from sysadmin_env.sandbox import DockerSandbox


class DiskFullScenario(BaseScenario):
    """Scenario where disk is full due to large log file."""

    @property
    def id(self) -> str:
        return "disk_full"

    @property
    def category(self) -> str:
        return "Disk"

    def get_complaint(self) -> str:
        return "Server is acting weird. Can't save any files, getting 'No space left on device' errors everywhere."

    def setup(self, sandbox: DockerSandbox) -> None:
        # Install nginx for a service to test
        sandbox.exec("apt-get update && apt-get install -y nginx >/dev/null 2>&1")

    def break_system(self, sandbox: DockerSandbox) -> None:
        self.setup(sandbox)

        # Create a large file in /var/log to fill disk
        # Using fallocate is faster than dd
        sandbox.exec("fallocate -l 800M /var/log/huge_debug.log 2>/dev/null || dd if=/dev/zero of=/var/log/huge_debug.log bs=1M count=800 2>/dev/null")

        # Stop nginx to simulate service issues
        sandbox.exec("systemctl stop nginx")

    def check_fixed(self, sandbox: DockerSandbox) -> bool:
        # Check disk usage - should be under 90%
        result = sandbox.exec("df -h / | tail -1 | awk '{print $5}' | tr -d '%'")
        try:
            usage = int(result.strip())
            return usage < 90
        except ValueError:
            return False
