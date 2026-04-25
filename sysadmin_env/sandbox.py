"""Docker container sandbox management."""

import time
import docker
from docker.errors import NotFound, APIError


MAX_OUTPUT_SIZE = 2048  # 2KB truncation


class DockerSandbox:
    """Manages a Docker container for running scenarios."""

    IMAGE = "sysadmin-sandbox:latest"
    CONTAINER_PREFIX = "sysadmin_env_"

    def __init__(self):
        self.client = docker.from_env()
        self.container = None
        self.container_id = None

    def create(self, name_suffix: str = None) -> str:
        """Create and start a new container.

        Args:
            name_suffix: Optional suffix for container name

        Returns:
            Container ID
        """
        # Generate unique name
        name = f"{self.CONTAINER_PREFIX}{name_suffix or int(time.time() * 1000)}"

        # Remove existing container with same name if exists
        try:
            old = self.client.containers.get(name)
            old.remove(force=True)
        except NotFound:
            pass

        # Create container with systemd support
        self.container = self.client.containers.run(
            self.IMAGE,
            command="/lib/systemd/systemd",
            name=name,
            detach=True,
            privileged=True,
            # tmpfs for faster operation
            tmpfs={"/run": "", "/run/lock": ""},
            # Cgroup settings for systemd
            cgroupns="host",
            volumes={"/sys/fs/cgroup": {"bind": "/sys/fs/cgroup", "mode": "rw"}},
            # Don't auto-remove so we can inspect on failure
            remove=False,
        )
        self.container_id = self.container.id

        # Wait for systemd to initialize
        self._wait_for_systemd()

        return self.container_id

    def _wait_for_systemd(self, timeout: float = 10.0):
        """Wait for systemd to be ready in the container."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                result = self.exec("systemctl is-system-running --wait 2>/dev/null || true")
                if "running" in result or "degraded" in result:
                    return
            except Exception:
                pass
            time.sleep(0.5)

    def exec(self, command: str, timeout: float = 30.0) -> str:
        """Execute a command in the container.

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds

        Returns:
            Combined stdout and stderr, truncated to MAX_OUTPUT_SIZE
        """
        if not self.container:
            raise RuntimeError("Container not created")

        try:
            # Refresh container state
            self.container.reload()

            # Execute command
            exit_code, output = self.container.exec_run(
                ["bash", "-c", command],
                demux=False,
                workdir="/root",
            )

            # Decode and truncate output
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")

            if len(output) > MAX_OUTPUT_SIZE:
                output = output[:MAX_OUTPUT_SIZE] + "\n... [output truncated to 2KB]"

            return output

        except APIError as e:
            return f"Error executing command: {e}"

    def destroy(self):
        """Force remove the container."""
        if self.container:
            try:
                self.container.remove(force=True)
            except NotFound:
                pass
            except APIError:
                pass
            self.container = None
            self.container_id = None

    def is_running(self) -> bool:
        """Check if the container is running."""
        if not self.container:
            return False
        try:
            self.container.reload()
            return self.container.status == "running"
        except (NotFound, APIError):
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
