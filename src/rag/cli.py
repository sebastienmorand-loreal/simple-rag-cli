"""CLI commands for simple-rag application."""

import logging
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from commons.database.vector_store import (
    VectorStoreService,
    VectorStoreError,
    CollectionNotFoundError,
    DuplicateEntryError,
)

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="Simple RAG operations using ChromaDB")


def get_vector_store() -> VectorStoreService:
    """Get vector store service instance.

    Returns:
        Configured VectorStoreService instance

    Exceptions:
        VectorStoreError: If service initialization fails
    """
    try:
        return VectorStoreService()
    except Exception as e:
        logger.error("Failed to initialize vector store: %s", str(e))
        console.print(f"[red]Error: Failed to initialize vector store: {str(e)}[/red]")
        raise typer.Exit(1) from e


@app.command("load")
def load_command(
    index: str = typer.Option(..., "--index", help="Index name to store the data"),
    file_path: Optional[Path] = typer.Option(None, "-f", "--file", help="JSONL file path for batch loading"),
    key: Optional[str] = typer.Argument(None, help="Key text to vectorize (for single entry)"),
    value: Optional[str] = typer.Argument(None, help="Associated value (for single entry)"),
) -> None:
    """Load data into the vector store.

    Two modes of operation:
    1. Single entry: simple-rag load --index projects "SDDS Project" "itg-btdppublished-gbl-ww-pd"
    2. Batch from file: simple-rag load --index projects -f data.jsonl

    For batch loading, the JSONL file should contain one JSON object per line:
    {"key": "SDDS project", "value": "itg-btdppublished-gbl-ww-pd"}
    """
    if file_path and (key or value):
        console.print("[red]Error: Cannot specify both file and individual key/value arguments[/red]")
        raise typer.Exit(1)

    if not file_path and (not key or not value):
        console.print("[red]Error: Must specify either --file or both key and value arguments[/red]")
        raise typer.Exit(1)

    vector_store = get_vector_store()

    try:
        if file_path:
            # Batch loading from JSONL file
            console.print(f"Loading data from {file_path} into index '{index}'...")

            if not file_path.exists():
                console.print(f"[red]Error: File not found: {file_path}[/red]")
                raise typer.Exit(1)

            loaded_count = vector_store.load_from_jsonl(index, file_path)
            console.print(f"[green]Successfully loaded {loaded_count} entries into index '{index}'[/green]")

        else:
            # Single entry loading
            console.print(f"Loading key '{key}' into index '{index}'...")

            vector_store.load_single(index, key, value)
            console.print(f"[green]Successfully loaded key '{key}' with value '{value}' into index '{index}'[/green]")

    except DuplicateEntryError as e:
        console.print(f"[yellow]Warning: {str(e)}[/yellow]")
        console.print("Use the retrieve command to see existing data.")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)

    except VectorStoreError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)

    except Exception as e:
        logger.error("Unexpected error during load: %s", str(e))
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("retrieve")
def retrieve_command(
    index: str = typer.Option(..., "--index", help="Index name to search in"),
    query: str = typer.Argument(..., help="Query text to search for"),
    limit: int = typer.Option(5, "--limit", "-n", help="Maximum number of results"),
    show_details: bool = typer.Option(False, "--details", "-d", help="Show detailed results including distances"),
) -> None:
    """Retrieve similar documents using RAG search with cosine distance.

    Example:
    simple-rag retrieve --index projects "SDDS project"
    """
    if limit <= 0:
        console.print("[red]Error: Limit must be positive[/red]")
        raise typer.Exit(1)

    vector_store = get_vector_store()

    try:
        console.print(f"Searching for '{query}' in index '{index}'...")
        results = vector_store.retrieve(index, query, limit)

        if not results:
            _handle_no_results(vector_store, query, index)
            return

        _display_results(results, show_details)

    except CollectionNotFoundError as e:
        _handle_collection_not_found(vector_store, e)
        raise typer.Exit(1)

    except VectorStoreError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)

    except Exception as e:
        logger.error("Unexpected error during retrieve: %s", str(e))
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


def _handle_no_results(vector_store: VectorStoreService, query: str, index: str) -> None:
    """Handle case when no results are found."""
    console.print(f"[yellow]No results found for query '{query}' in index '{index}'[/yellow]")
    _show_available_collections(vector_store)


def _handle_collection_not_found(vector_store: VectorStoreService, error: Exception) -> None:
    """Handle collection not found error."""
    console.print(f"[red]Error: {str(error)}[/red]")
    _show_available_collections(vector_store)


def _show_available_collections(vector_store: VectorStoreService) -> None:
    """Show available collections to the user."""
    try:
        collections = vector_store.list_collections()
        if collections:
            console.print(f"\nAvailable indexes: {', '.join(collections)}")
        else:
            console.print("\nNo indexes found. Use 'load' command to add data first.")
    except (VectorStoreError, CollectionNotFoundError):
        # Gracefully handle collection listing errors in helper function
        logger.debug("Failed to list collections in helper function")


def _display_results(results: list, show_details: bool) -> None:
    """Display search results in formatted panels."""
    console.print(f"\n[green]Found {len(results)} results:[/green]\n")

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        document = result.get("document", "")
        distance = result.get("distance")

        # Create result panel
        title = f"Result {i}"
        if distance is not None:
            title += f" (distance: {distance:.4f})"

        content = f"[bold]Key:[/bold] {document}\n"
        content += f"[bold]Value:[/bold] {metadata.get('value', 'N/A')}"

        if show_details:
            content += f"\n[bold]ID:[/bold] {result.get('id', 'N/A')}"
            content += f"\n[bold]Type:[/bold] {metadata.get('type', 'N/A')}"

        panel = Panel(content, title=title, expand=False)
        console.print(panel)


@app.command("list")
def list_command() -> None:
    """List all available indexes and their information."""
    vector_store = get_vector_store()

    try:
        collections = vector_store.list_collections()

        if not collections:
            console.print("[yellow]No indexes found. Use 'load' command to add data first.[/yellow]")
            return

        console.print(f"\n[green]Found {len(collections)} indexes:[/green]\n")

        # Create table
        table = Table(title="Available Indexes")
        table.add_column("Index Name", style="cyan", no_wrap=True)
        table.add_column("Document Count", style="magenta", justify="right")
        table.add_column("Storage Path", style="green")

        for collection_name in collections:
            try:
                info = vector_store.get_collection_info(collection_name)
                table.add_row(collection_name, str(info["document_count"]), str(info["storage_path"]))
            except (VectorStoreError, CollectionNotFoundError) as e:
                logger.warning("Failed to get info for collection %s: %s", collection_name, str(e))
                table.add_row(collection_name, "Error", "Unknown")

        console.print(table)

    except VectorStoreError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)

    except Exception as e:
        logger.error("Unexpected error during list: %s", str(e))
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("info")
def info_command(index: str = typer.Option(..., "--index", help="Index name to get information about")) -> None:
    """Get detailed information about a specific index."""
    vector_store = get_vector_store()

    try:
        info = vector_store.get_collection_info(index)

        # Display information
        content = f"[bold]Name:[/bold] {info['name']}\n"
        content += f"[bold]Document Count:[/bold] {info['document_count']}\n"
        content += f"[bold]Embedding Model:[/bold] {info['embedding_model']}\n"
        content += f"[bold]Storage Path:[/bold] {info['storage_path']}"

        panel = Panel(content, title=f"Index Information: {index}", expand=False)
        console.print(panel)

    except CollectionNotFoundError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)

    except VectorStoreError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)

    except Exception as e:
        logger.error("Unexpected error during info: %s", str(e))
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)
