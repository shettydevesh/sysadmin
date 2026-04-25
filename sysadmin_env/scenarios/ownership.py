"""Ownership scenario: Wrong file ownership prevents service from starting."""

from .base import BaseScenario
from sysadmin_env.sandbox import DockerSandbox


class OwnershipScenario(BaseScenario):
    """Scenario where nginx config has wrong ownership."""

    @property
    def id(self) -> str:
        return "ownership"

    @property
    def category(self) -> str:
        return "Permissions"

    def get_complaint(self) -> str:
        return "nginx won't start. I didn't change anything, it just stopped working after I ran some chmod commands."

    def setup(self, sandbox: DockerSandbox) -> None:
        # Install nginx
        sandbox.exec("apt-get update && apt-get install -y nginx >/dev/null 2>&1")

    def break_system(self, sandbox: DockerSandbox) -> None:
        self.setup(sandbox)

        # Change ownership of nginx config to a non-root user
        sandbox.exec("chown nobody:nogroup /etc/nginx/nginx.conf")
        sandbox.exec("chmod 600 /etc/nginx/nginx.conf")

        # Stop nginx so agent needs to fix and restart
        sandbox.exec("systemctl stop nginx")

    def check_fixed(self, sandbox: DockerSandbox) -> bool:
        # Check if nginx is running and responding
        result = sandbox.exec("systemctl is-active nginx")
        if "active" not in result:
            return False

        # Also verify we can actually access it
        result = sandbox.exec("curl -s -o /dev/null -w '%{http_code}' http://localhost:80 2>/dev/null || echo 'failed'")
        return "200" in result or "301" in result
