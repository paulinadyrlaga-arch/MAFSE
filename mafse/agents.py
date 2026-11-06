"""Przykładowi agenci demonstracyjni."""

from mafse.agent import Agent, AgentContext
from mafse.gemini import GeminiClient


class GeminiAgent(Agent):
    def __init__(self, client: GeminiClient) -> None:
        self.client = client


class PlannerAgent(GeminiAgent):
    name = "planista"

    def run(self, context: AgentContext) -> str:
        return self.client.generate(
            "Jesteś agentem planującym. Przygotuj krótki, ponumerowany plan "
            f"realizacji zadania.\n\nZadanie: {context.task}"
        )


class ResearchAgent(GeminiAgent):
    name = "analityk"

    def run(self, context: AgentContext) -> str:
        plan = context.results.get("planista", "")
        return self.client.generate(
            "Jesteś agentem badawczym. Przeanalizuj zadanie i przygotuj "
            "zwięzłe ustalenia. Nie twierdź, że sprawdziłeś źródła zewnętrzne, "
            "jeśli nie zostały dostarczone.\n\n"
            f"Zadanie: {context.task}\n\nPlan:\n{plan}"
        )


class WriterAgent(GeminiAgent):
    name = "redaktor"

    def run(self, context: AgentContext) -> str:
        plan = context.results.get("planista", "")
        analysis = context.results.get("analityk", "")
        return self.client.generate(
            "Jesteś agentem redagującym. Na podstawie zadania i ustaleń "
            "przygotuj jasną, kompletną odpowiedź po polsku.\n\n"
            f"Zadanie: {context.task}\n\nPlan:\n{plan}\n\n"
            f"Ustalenia:\n{analysis}"
        )
