"""Wspólne typy i interfejs agentów."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentContext:
    """Stan współdzielony przez agentów podczas jednego zadania."""

    task: str
    results: dict[str, str] = field(default_factory=dict)


class Agent(ABC):
    """Bazowy interfejs agenta."""

    name: str

    @abstractmethod
    def run(self, context: AgentContext) -> str:
        """Wykonuje krok agenta i zwraca jego wynik."""
