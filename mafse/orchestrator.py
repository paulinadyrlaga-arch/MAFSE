"""Koordynacja przepływu pracy między agentami."""

from collections.abc import Sequence

from mafse.agent import Agent, AgentContext


class Orchestrator:
    """Uruchamia agentów po kolei, przekazując im wspólny kontekst."""

    def __init__(self, agents: Sequence[Agent]) -> None:
        if not agents:
            raise ValueError("Orkiestrator wymaga co najmniej jednego agenta.")
        self.agents = tuple(agents)

    def run(self, task: str) -> str:
        if not task.strip():
            raise ValueError("Opis zadania nie może być pusty.")

        context = AgentContext(task=task)
        for agent in self.agents:
            context.results[agent.name] = agent.run(context)

        return context.results[self.agents[-1].name]
