import typer
from pathlib import Path

from src.pipeline import run

app = typer.Typer(help="DevOps log analyzer with AI-powered error enrichment.")


@app.command()
def analyze(
    log_file: Path = typer.Argument(..., help="Path to the raw log file."),
    output: Path = typer.Option("data/output/report.json", help="Path for the JSON report."),
) -> None:
    run(log_file, output)


if __name__ == "__main__":
    app()
