"""Stale PID scenario: Leftover PID file prevents service from starting."""

from .base import BaseScenario
from sysadmin_env.sandbox import DockerSandbox


class StalePidScenario(BaseScenario):
    """Scenario where a stale PID file prevents nginx from starting."""

    @property
    def id(self) -> str:
        return "stale_pid"

    @property
    def category(self) -> str:
        return "Service"

    def get_complaint(self) -> str:
        return "nginx won't start after the server crashed. Something about a PID file but the old process isn't running."

    def setup(self, sandbox: DockerSandbox) -> None:
        sandbox.exec("apt-get update && apt-get install -y nginx >/dev/null 2>&1")

    def break_system(self, sandbox: DockerSandbox) -> None:
        self.setup(sandbox)

        # Stop nginx
        sandbox.exec("systemctl stop nginx 2>/dev/null || true")
        sandbox.exec("pkill nginx 2>/dev/null || true")

        # Create stale PID file with a non-existent PID
        sandbox.exec("""
mkdir -p /run/nginx
echo "99999" > /run/nginx.pid
chmod 644 /run/nginx.pid
""")

    def check_fixed(self, sandbox: DockerSandbox) -> bool:
        # Check nginx is running
        result = sandbox.exec("systemctl is-active nginx")
        if "active" not in result:
            return False

        # Check it responds
        result = sandbox.exec("curl -s -o /dev/null -w '%{http_code}' http://localhost:80 2>/dev/null || echo 'failed'")
        return "200" in result or "301" in result
