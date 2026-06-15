import structlog
import typer
from pathlib import Path

from src.config.settings import load_config
from src.config.logging_setup import setup_logging
from src.pipeline import run

app = typer.Typer(help="DevOps log analyzer with AI-powered error enrichment.")
logger = structlog.get_logger(__name__)


@app.command()
def analyze(
    log_file: Path = typer.Argument(..., help="Path to the raw log file."),
    output: Path = typer.Option("data/output/report.json", help="Path for the JSON report."),
    config_path: Path = typer.Option("config/settings.yaml", help="Path to config file."),
) -> None:
    config = load_config(config_path)
    setup_logging(config.logging)
    logger.info("config_loaded", path=str(config_path))
    run(log_file, output, config=config)


if __name__ == "__main__":
    app()
