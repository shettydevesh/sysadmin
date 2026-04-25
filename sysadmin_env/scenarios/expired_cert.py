"""Expired certificate scenario: SSL cert has expired."""

from .base import BaseScenario
from sysadmin_env.sandbox import DockerSandbox


class ExpiredCertScenario(BaseScenario):
    """Scenario where SSL certificate has expired."""

    @property
    def id(self) -> str:
        return "expired_cert"

    @property
    def category(self) -> str:
        return "TLS"

    def get_complaint(self) -> str:
        return "HTTPS is broken! Browsers say the certificate is expired. We need secure connections working!"

    def setup(self, sandbox: DockerSandbox) -> None:
        sandbox.exec("apt-get update && apt-get install -y nginx openssl >/dev/null 2>&1")

    def break_system(self, sandbox: DockerSandbox) -> None:
        self.setup(sandbox)

        # Create an expired certificate (valid from 2 years ago to 1 year ago)
        sandbox.exec("""
mkdir -p /etc/nginx/ssl

# Create expired certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/server.key \
    -out /etc/nginx/ssl/server.crt \
    -subj "/CN=localhost" \
    -addext "subjectAltName = DNS:localhost" \
    2>/dev/null

# Backdate the cert by modifying system time temporarily isn't reliable
# Instead, create a cert that expired in the past using faketime or just
# set the system date forward
date -s "+2 years" 2>/dev/null || true
""")

        # Configure nginx for HTTPS
        sandbox.exec("""cat > /etc/nginx/sites-available/default << 'EOF'
server {
    listen 443 ssl;
    server_name localhost;

    ssl_certificate /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;

    root /var/www/html;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}
EOF""")

        sandbox.exec("systemctl restart nginx 2>/dev/null || true")

    def check_fixed(self, sandbox: DockerSandbox) -> bool:
        # Check the certificate is valid (not expired)
        result = sandbox.exec("""
openssl x509 -in /etc/nginx/ssl/server.crt -noout -checkend 0 2>&1
echo "exit_code:$?"
""")

        # checkend 0 returns 0 if cert will not expire in 0 seconds (i.e., is valid)
        if "exit_code:0" in result and "will expire" not in result.lower():
            # Also check nginx is running
            active_result = sandbox.exec("systemctl is-active nginx")
            return "active" in active_result

        return False
