"""Runaway CPU scenario: Infinite loop process consuming CPU."""

from .base import BaseScenario
from sysadmin_env.sandbox import DockerSandbox


class RunawayCpuScenario(BaseScenario):
    """Scenario where a runaway process is consuming all CPU."""

    @property
    def id(self) -> str:
        return "runaway_cpu"

    @property
    def category(self) -> str:
        return "Process"

    def get_complaint(self) -> str:
        return "Server is extremely slow. Load is through the roof. Something is eating all the CPU but I can't figure out what."

    def setup(self, sandbox: DockerSandbox) -> None:
        sandbox.exec("apt-get update && apt-get install -y nginx >/dev/null 2>&1")

    def break_system(self, sandbox: DockerSandbox) -> None:
        self.setup(sandbox)

        # Create a runaway script
        sandbox.exec("""cat > /usr/local/bin/maintenance.sh << 'EOF'
#!/bin/bash
while true; do
    : # infinite loop
done
EOF
chmod +x /usr/local/bin/maintenance.sh
""")

        # Start the runaway process
        sandbox.exec("nohup /usr/local/bin/maintenance.sh > /dev/null 2>&1 &")

        # Start a second one for good measure
        sandbox.exec("nohup bash -c 'while true; do :; done' > /dev/null 2>&1 &")

    def check_fixed(self, sandbox: DockerSandbox) -> bool:
        # Check no runaway bash loops
        result = sandbox.exec("ps aux | grep -E '(maintenance\\.sh|while true)' | grep -v grep | wc -l")
        try:
            count = int(result.strip())
            if count > 0:
                return False
        except ValueError:
            pass

        # Check CPU usage is reasonable (this is approximate)
        result = sandbox.exec("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | tr -d '%id,'")
        # If we can't parse CPU, just check the processes are gone
        return True
