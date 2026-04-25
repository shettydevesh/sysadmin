"""Base class for all scenarios."""

from abc import ABC, abstractmethod
from sysadmin_env.sandbox import DockerSandbox


class BaseScenario(ABC):
    """Abstract base class for sysadmin scenarios."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this scenario."""
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        """Category of the scenario (Disk, Service, Network, etc.)."""
        ...

    @abstractmethod
    def get_complaint(self) -> str:
        """Get the user complaint that starts the episode.

        Returns:
            A realistic user complaint describing the problem
        """
        ...

    @abstractmethod
    def break_system(self, sandbox: DockerSandbox) -> None:
        """Break the system in a specific way.

        Args:
            sandbox: The Docker sandbox to break
        """
        ...

    @abstractmethod
    def check_fixed(self, sandbox: DockerSandbox) -> bool:
        """Check if the issue has been fixed.

        This method must be side-effect free.

        Args:
            sandbox: The Docker sandbox to check

        Returns:
            True if the issue is fixed
        """
        ...

    def setup(self, sandbox: DockerSandbox) -> None:
        """Optional setup before breaking (install packages, etc.).

        Args:
            sandbox: The Docker sandbox to set up
        """
        pass
