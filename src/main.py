"""Main entry point for simple-rag CLI application."""

import logging
from datetime import datetime
from pathlib import Path

import colorlog
import typer

from rag.cli import app as rag_app

logging.basicConfig(level=logging.INFO)

for logger_name in (
    "transformers.tokenization_utils_base",
    "transformers.configuration_utils",
    "transformers.modeling_utils",
    "sentence_transformers.SentenceTransformer",
    "chromadb.db.mixins.embeddings_queue",
):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

app = typer.Typer(
    name="simple-rag",
    help="A simple RAG CLI tool using ChromaDB for local vector storage",
    no_args_is_help=True,
)

app.add_typer(rag_app, name="")


def setup_logging(debug: bool = False) -> None:
    """Setup colored logging configuration.

    Arguments:
        debug: Enable debug-level logging if True
    """
    if debug:
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "[ %(asctime)s ] %(log_color)s%(levelname)7s%(reset)s: %(module)s.%(funcName)s: %(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )
        logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
    else:
        log_dir = Path.home() / ".logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"simple-rag.log.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "[ %(asctime)s ] %(log_color)s%(levelname)7s%(reset)s: %(module)s.%(funcName)s: %(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("[ %(asctime)s ] %(levelname)7s: %(module)s.%(funcName)s: %(message)s")
        )

        logging.basicConfig(level=logging.INFO, handlers=[handler, file_handler], force=True)


@app.callback()
def main(
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging to stdout"),
) -> None:
    """Simple RAG CLI tool using ChromaDB for local vector storage."""
    setup_logging(debug)


if __name__ == "__main__":
    app()
