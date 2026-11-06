"""Punkt wejścia do demonstracyjnej aplikacji MAFSE."""

import argparse

from mafse import (
    GeminiClient,
    Orchestrator,
    PlannerAgent,
    ResearchAgent,
    WriterAgent,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Uruchom prosty przepływ MAFSE.")
    parser.add_argument(
        "task",
        nargs="?",
        default="Przygotuj krótkie podsumowanie przykładowego tematu.",
        help="Opis zadania przekazywanego agentom.",
    )
    args = parser.parse_args()

    client = GeminiClient.from_env()
    app = Orchestrator(
        agents=[
            PlannerAgent(client),
            ResearchAgent(client),
            WriterAgent(client),
        ]
    )
    print(app.run(args.task))


if __name__ == "__main__":
    main()