import yaml
from pydantic import ValidationError
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
    log_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to the raw log file."),
    output: Path = typer.Option("data/output/report.json", help="Path for the JSON report."),
    config_path: Path = typer.Option("config/settings.yaml",exists=True, readable=True, help="Path to config file."),
) -> None:
    try:
        config = load_config(config_path)
        setup_logging(config.logging)
        logger.info("config_loaded", path=str(config_path))
        run(log_file, output, config=config)
    except yaml.YAMLError as e:
        typer.echo(f"Config file is not valid YAML: {e}", err=True)
        raise typer.Exit(1)
    except ValidationError as e:
        typer.echo(f"Config error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("pipeline_failed", error=str(e))
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
