"""Port bound scenario: Another process is using port 80."""

from .base import BaseScenario
from sysadmin_env.sandbox import DockerSandbox


class PortBoundScenario(BaseScenario):
    """Scenario where nginx can't start because port 80 is in use."""

    @property
    def id(self) -> str:
        return "port_bound"

    @property
    def category(self) -> str:
        return "Network"

    def get_complaint(self) -> str:
        return "nginx won't start. Says 'Address already in use'. I need the web server back up ASAP!"

    def setup(self, sandbox: DockerSandbox) -> None:
        sandbox.exec("apt-get update && apt-get install -y nginx apache2 >/dev/null 2>&1")

    def break_system(self, sandbox: DockerSandbox) -> None:
        self.setup(sandbox)

        # Stop nginx and start apache2 on port 80
        sandbox.exec("systemctl stop nginx 2>/dev/null || true")
        sandbox.exec("systemctl start apache2 2>/dev/null || true")

        # If apache2 didn't start, use a simple python http server
        sandbox.exec("""
if ! ss -tlnp | grep -q ':80 '; then
    nohup python3 -m http.server 80 --bind 0.0.0.0 > /dev/null 2>&1 &
fi
""")

    def check_fixed(self, sandbox: DockerSandbox) -> bool:
        # Check nginx is running
        result = sandbox.exec("systemctl is-active nginx")
        if "active" not in result:
            return False

        # Check nginx is actually on port 80
        result = sandbox.exec("ss -tlnp | grep ':80 '")
        if "nginx" not in result.lower():
            return False

        # Check it responds
        result = sandbox.exec("curl -s -o /dev/null -w '%{http_code}' http://localhost:80 2>/dev/null || echo 'failed'")
        return "200" in result or "301" in result
