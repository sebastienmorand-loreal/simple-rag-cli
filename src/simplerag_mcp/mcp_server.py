"""MCP server implementation for simple-rag retrieve functionality."""

import json
import logging
import sys
from typing import Any, Dict, List, Optional
import asyncio

from mcp import server, stdio_server, Tool
from mcp.types import TextContent

from commons.database.vector_store import (
    VectorStoreService,
    VectorStoreError,
    CollectionNotFoundError,
)

logger = logging.getLogger(__name__)

# Initialize the MCP server
app = server.Server("simple-rag")

# Global vector store instance
vector_store = None


def get_vector_store() -> VectorStoreService:
    """Get or create vector store service instance.

    Returns:
        Configured VectorStoreService instance

    Exceptions:
        VectorStoreError: If service initialization fails
    """
    global vector_store
    if vector_store is None:
        try:
            vector_store = VectorStoreService()
            logger.info("Vector store service initialized")
        except Exception as e:
            logger.error("Failed to initialize vector store: %s", str(e))
            raise VectorStoreError(f"Failed to initialize vector store: {str(e)}") from e
    return vector_store


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools.

    Returns:
        List of available tools
    """
    return [
        Tool(
            name="retrieve",
            description="Retrieve similar documents using RAG search with cosine distance",
            inputSchema={
                "type": "object",
                "properties": {
                    "index": {
                        "type": "string",
                        "description": "Index name to search in. This is the type of object we are trying to retrieve.",
                    },
                    "query": {"type": "string", "description": "Query text to search for"},
                    "threshold": {
                        "type": "number",
                        "description": "Distance threshold to consider (optional)",
                    },
                    "number": {
                        "type": "integer",
                        "description": "Number of results to return (default: 1)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 1,
                    },
                },
                "required": ["index", "query"],
                "additionalProperties": False,
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls.

    Arguments:
        name: Name of the tool to call
        arguments: Tool arguments

    Returns:
        List of text content responses

    Exceptions:
        ValueError: If tool name is unknown or arguments are invalid
        VectorStoreError: If retrieval fails
    """
    if name != "retrieve":
        raise ValueError(f"Unknown tool: {name}")

    # Validate required arguments
    if "index" not in arguments or "query" not in arguments:
        raise ValueError("Missing required arguments: index and query")

    index_name = arguments["index"]
    query = arguments["query"]
    threshold = arguments.get("threshold")
    number = arguments.get("number", 1)

    # Validate arguments
    if not isinstance(index_name, str) or not index_name.strip():
        raise ValueError("Index must be a non-empty string")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if threshold is not None and (not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 2):
        raise ValueError("Threshold must be a number between 0 and 2")

    if not isinstance(number, int) or number < 1 or number > 100:
        raise ValueError("Number must be an integer between 1 and 100")

    try:
        # Get vector store service
        store = get_vector_store()

        # Retrieve more results than needed to apply threshold filtering
        max_results = max(number * 2, 10)
        results = store.retrieve(index_name, query, max_results)

        if not results:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": True,
                            "results": [],
                            "message": f"No results found for query '{query}' in index '{index_name}'",
                        },
                        indent=2,
                    ),
                )
            ]

        # Apply threshold filtering if specified
        if threshold is not None:
            results = [r for r in results if r.get("distance", 0) <= threshold]

        # Limit to requested number
        results = results[:number]

        if not results:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"success": True, "results": [], "message": f"No results found within threshold {threshold}"},
                        indent=2,
                    ),
                )
            ]

        # Format results for MCP response
        formatted_results = []
        for result in results:
            metadata = result.get("metadata", {})
            formatted_results.append(
                {
                    "value": metadata.get("value", ""),
                    "key": metadata.get("key", ""),
                    "distance": result.get("distance", 0.0),
                }
            )

        response = {
            "success": True,
            "results": formatted_results,
            "count": len(formatted_results),
            "query": query,
            "index": index_name,
        }

        if threshold is not None:
            response["threshold"] = threshold

        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    except CollectionNotFoundError as e:
        # Get available indexes for helpful error message
        try:
            available_indexes = store.list_collections()
            error_response = {"success": False, "error": str(e), "available_indexes": available_indexes}
        except VectorStoreError:
            error_response = {"success": False, "error": str(e), "available_indexes": []}

        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]

    except VectorStoreError as e:
        return [
            TextContent(
                type="text", text=json.dumps({"success": False, "error": f"Vector store error: {str(e)}"}, indent=2)
            )
        ]

    except Exception as e:
        logger.error("Unexpected error in retrieve tool: %s", str(e))
        return [
            TextContent(
                type="text", text=json.dumps({"success": False, "error": f"Unexpected error: {str(e)}"}, indent=2)
            )
        ]


async def main():
    """Main entry point for the MCP server."""
    # Set up logging for MCP server - only to stderr to avoid interfering with MCP protocol
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log level to avoid noise
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )

    try:
        # Initialize vector store early to catch initialization errors
        get_vector_store()

        # Run the stdio server
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    except KeyboardInterrupt:
        # Handle graceful shutdown
        pass
    except Exception as e:
        logger.error("Failed to start MCP server: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
