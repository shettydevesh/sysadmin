"""Nginx unknown directive scenario: Invalid directive in config."""

from .base import BaseScenario
from sysadmin_env.sandbox import DockerSandbox


class NginxUnknownScenario(BaseScenario):
    """Scenario where nginx has an unknown directive."""

    @property
    def id(self) -> str:
        return "nginx_unknown"

    @property
    def category(self) -> str:
        return "Service"

    def get_complaint(self) -> str:
        return "nginx keeps failing to start. The error mentions something about an unknown directive but I can't figure out which one."

    def setup(self, sandbox: DockerSandbox) -> None:
        sandbox.exec("apt-get update && apt-get install -y nginx >/dev/null 2>&1")

    def break_system(self, sandbox: DockerSandbox) -> None:
        self.setup(sandbox)

        # Add an unknown directive
        sandbox.exec("""cat > /etc/nginx/sites-available/default << 'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;

    root /var/www/html;
    index index.html index.htm;

    server_name _;

    enable_magic_cache on;

    location / {
        try_files $uri $uri/ =404;
    }
}
EOF""")

        sandbox.exec("systemctl stop nginx 2>/dev/null || true")

    def check_fixed(self, sandbox: DockerSandbox) -> bool:
        # Check nginx config is valid
        result = sandbox.exec("nginx -t 2>&1")
        if "syntax is ok" not in result.lower() and "test is successful" not in result.lower():
            return False

        # Check nginx is running
        result = sandbox.exec("systemctl is-active nginx")
        if "active" not in result:
            return False

        # Check it responds
        result = sandbox.exec("curl -s -o /dev/null -w '%{http_code}' http://localhost:80 2>/dev/null || echo 'failed'")
        return "200" in result or "301" in result
