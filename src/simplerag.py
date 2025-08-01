"""Main entry point for simple-rag CLI application."""

import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import colorlog
import typer

# Set environment variables to suppress transformer warnings before any imports
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src directory to Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Local imports (after path modification)
# pylint: disable=wrong-import-position
from rag.cli import app as rag_app

# Filter out FutureWarning about encoder_attention_mask deprecation and model sharding warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*layers were not sharded.*", category=UserWarning)

logging.basicConfig(level=logging.INFO)

for logger_name in (
    "transformers.tokenization_utils_base",
    "transformers.configuration_utils",
    "transformers.modeling_utils",
    "sentence_transformers.SentenceTransformer",
    "chromadb.db.mixins.embeddings_queue",
    "transformers.models.bert.modeling_bert",
):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

app = typer.Typer(
    name="simple-rag",
    help="A simple RAG CLI tool using ChromaDB for local vector storage",
    no_args_is_help=True,
)

app.add_typer(rag_app, name="")


def setup_logging(debug: bool = False) -> str:
    """Setup logging configuration with file output only.

    Arguments:
        debug: Enable debug-level logging if True

    Returns:
        Path to the log file being used
    """
    log_dir = Path.home() / ".logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"simplerag.log.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Only log to file, no console output for regular operations
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(levelname)7s: %(module)s.%(funcName)s: %(message)s"))

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, handlers=[file_handler], force=True)

    return str(log_file)


@app.callback()
def callback(
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Simple RAG CLI tool using ChromaDB for local vector storage."""
    log_file = setup_logging(debug)
    # Show log file path on stderr
    typer.echo(f"Logging to: {log_file}", err=True)


if __name__ == "__main__":
    app()
