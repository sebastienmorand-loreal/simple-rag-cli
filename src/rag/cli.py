"""CLI commands for simple-rag application."""

import json
import logging
import sys
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
        raise typer.Exit(1) from e


@app.command("load")
def load_command(
    index: str = typer.Option(..., "--index", "-i", help="Index name to store the data"),
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
    index: str = typer.Option(..., "--index", "-i", help="Index name to search in"),
    query: str = typer.Argument(..., help="Query text to search for"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Distance threshold to consider"),
    number: int = typer.Option(1, "--number", "-n", help="Number of values to display"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output in JSON format"),
) -> None:
    """Retrieve similar documents using RAG search with cosine distance.

    Example:
    simple-rag retrieve --index projects "SDDS project"
    simple-rag retrieve --index projects "SDDS project" --threshold 0.5 --number 3 --json
    """
    if number <= 0:
        logger.error("Number must be positive: %d", number)
        raise typer.Exit(1)

    vector_store = get_vector_store()

    try:
        # Retrieve more results than needed to apply threshold filtering
        max_results = max(number * 2, 10)
        results = vector_store.retrieve(index, query, max_results)

        if not results:
            logger.info("No results found for query '%s' in index '%s'", query, index)
            raise typer.Exit(0)

        # Apply threshold filtering if specified
        if threshold is not None:
            results = [r for r in results if r.get("distance", 0) <= threshold]

        # Limit to requested number
        results = results[:number]

        if not results:
            logger.info("No results found within threshold %s", threshold)
            raise typer.Exit(0)

        # Output results
        if json_output:
            _output_json_results(results)
        else:
            _output_simple_results(results)

    except CollectionNotFoundError as e:
        logger.error("Collection not found: %s", str(e))
        raise typer.Exit(1)

    except VectorStoreError as e:
        logger.error("Vector store error: %s", str(e))
        raise typer.Exit(1)

    except Exception as e:
        logger.error("Unexpected error during retrieve: %s", str(e))
        raise typer.Exit(1)


def _output_simple_results(results: list) -> None:
    """Output simple results - just the values to stdout."""
    for result in results:
        metadata = result.get("metadata", {})
        value = metadata.get("value", "")
        print(value)


def _output_json_results(results: list) -> None:
    """Output JSON results with value, key, and distance."""
    output_data = []
    for result in results:
        metadata = result.get("metadata", {})
        output_data.append(
            {
                "value": metadata.get("value", ""),
                "key": metadata.get("key", ""),
                "distance": result.get("distance", 0.0),
            }
        )

    print(json.dumps(output_data, indent=2))


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
def info_command(index: str = typer.Option(..., "--index", "-i", help="Index name to get information about")) -> None:
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


@app.command("dump")
def dump_command(index: str = typer.Option(..., "--index", "-i", help="Index name to dump data from")) -> None:
    """Dump all data from a specific index in JSONL format."""
    vector_store = get_vector_store()

    try:
        # Get all documents from the collection
        data = vector_store.dump_all_data(index)

        if not data:
            console.print(f"[yellow]Index '{index}' is empty[/yellow]")
            return

        # Output each entry as JSONL
        for entry in data:
            metadata = entry.get("metadata", {})
            output_entry = {"key": metadata.get("key", ""), "value": metadata.get("value", "")}
            print(json.dumps(output_entry))

    except CollectionNotFoundError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        console.print("Available indexes:")
        try:
            collections = vector_store.list_collections()
            for collection in collections:
                console.print(f"  - {collection}")
        except VectorStoreError:
            console.print("  Unable to list available indexes")
        raise typer.Exit(1)

    except VectorStoreError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)

    except Exception as e:
        logger.error("Unexpected error during dump: %s", str(e))
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("drop")
def drop_command(index: str = typer.Option(..., "--index", "-i", help="Index name to delete")) -> None:
    """Delete an entire index and all its data."""
    vector_store = get_vector_store()

    try:
        # Confirm deletion
        if not typer.confirm(f"Are you sure you want to delete index '{index}' and all its data?"):
            console.print("[yellow]Operation cancelled[/yellow]")
            return

        vector_store.delete_collection(index)
        console.print(f"[green]Successfully deleted index '{index}'[/green]")

    except CollectionNotFoundError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        console.print("Available indexes:")
        try:
            collections = vector_store.list_collections()
            for collection in collections:
                console.print(f"  - {collection}")
        except VectorStoreError:
            console.print("  Unable to list available indexes")
        raise typer.Exit(1)

    except VectorStoreError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)

    except Exception as e:
        logger.error("Unexpected error during drop: %s", str(e))
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("remove")
def remove_command(
    index: str = typer.Option(..., "--index", "-i", help="Index name to remove data from"),
    key: str = typer.Argument(..., help="Key to remove from the index"),
) -> None:
    """Remove a specific key from an index."""
    vector_store = get_vector_store()

    try:
        vector_store.remove_document(index, key)
        console.print(f"[green]Successfully removed key '{key}' from index '{index}'[/green]")

    except CollectionNotFoundError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        console.print("Available indexes:")
        try:
            collections = vector_store.list_collections()
            for collection in collections:
                console.print(f"  - {collection}")
        except VectorStoreError:
            console.print("  Unable to list available indexes")
        raise typer.Exit(1)

    except VectorStoreError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)

    except Exception as e:
        logger.error("Unexpected error during remove: %s", str(e))
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("mcp")
def mcp_command() -> None:
    """Start the MCP (Model Context Protocol) server with stdio transport."""
    import asyncio
    import sys

    try:
        # Import MCP server functionality
        from simplerag_mcp.mcp_server import main as mcp_main

        # DO NOT print anything to stdout - it interferes with MCP protocol
        # Only log errors to stderr if needed

        # Run the MCP server directly
        asyncio.run(mcp_main())

    except KeyboardInterrupt:
        # Exit quietly on Ctrl+C
        sys.exit(0)
    except ImportError as e:
        # Print error to stderr, not stdout
        print(f"Error: MCP dependencies not installed: {str(e)}", file=sys.stderr)
        print("Run: uv add mcp", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Log to stderr only
        print(f"Failed to start MCP server: {str(e)}", file=sys.stderr)
        sys.exit(1)
