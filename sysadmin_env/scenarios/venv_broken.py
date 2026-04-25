"""Broken venv scenario: Python virtual environment has broken symlinks."""

from .base import BaseScenario
from sysadmin_env.sandbox import DockerSandbox


class VenvBrokenScenario(BaseScenario):
    """Scenario where Python venv has broken symlinks."""

    @property
    def id(self) -> str:
        return "venv_broken"

    @property
    def category(self) -> str:
        return "Environment"

    def get_complaint(self) -> str:
        return "Our Python app won't start. It was working fine before the system update. Something about 'No such file or directory' when activating the venv."

    def setup(self, sandbox: DockerSandbox) -> None:
        sandbox.exec("apt-get update && apt-get install -y python3 python3-venv python3-pip >/dev/null 2>&1")

    def break_system(self, sandbox: DockerSandbox) -> None:
        self.setup(sandbox)

        # Create a venv
        sandbox.exec("python3 -m venv /opt/myapp/venv")

        # Install a package to make it more realistic
        sandbox.exec("/opt/myapp/venv/bin/pip install requests >/dev/null 2>&1 || true")

        # Create a simple app
        sandbox.exec("""cat > /opt/myapp/app.py << 'EOF'
import requests
print("App started successfully!")
EOF""")

        # Break the venv by removing and recreating python with different path
        # Simulating what happens when system python is upgraded
        sandbox.exec("""
# Remove the python symlink in venv
rm /opt/myapp/venv/bin/python
rm /opt/myapp/venv/bin/python3

# Create broken symlinks pointing to non-existent python
ln -s /usr/bin/python3.10 /opt/myapp/venv/bin/python
ln -s /usr/bin/python3.10 /opt/myapp/venv/bin/python3
""")

    def check_fixed(self, sandbox: DockerSandbox) -> bool:
        # Check the venv python works
        result = sandbox.exec("/opt/myapp/venv/bin/python --version 2>&1")
        if "Python" not in result:
            return False

        # Check we can import the installed package
        result = sandbox.exec("/opt/myapp/venv/bin/python -c 'import requests; print(\"ok\")' 2>&1")
        return "ok" in result
