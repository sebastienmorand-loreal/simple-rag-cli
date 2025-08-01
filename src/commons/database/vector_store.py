"""ChromaDB vector store service for RAG operations."""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.models.Collection import Collection

from commons.models.embedding import TransformerEmbeddingFunction

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Base exception for vector store operations."""


class CollectionNotFoundError(VectorStoreError):
    """Raised when a collection is not found."""


class DuplicateEntryError(VectorStoreError):
    """Raised when attempting to add duplicate entries."""


class VectorStoreService:
    """Service for managing ChromaDB vector store operations.

    This service provides high-level operations for storing and retrieving
    vectorized data using ChromaDB with custom embedding functions.
    """

    def __init__(self, storage_path: Optional[Path] = None, embedding_model: str = "w601sxs/b1ade-embed"):
        """Initialize the vector store service.

        Arguments:
            storage_path: Path to store ChromaDB data (defaults to ~/.simple-rag)
            embedding_model: HuggingFace model identifier for embeddings

        Exceptions:
            VectorStoreError: If the storage path cannot be created or accessed
            OSError: If the embedding model cannot be loaded
        """
        if storage_path is None:
            storage_path = Path.home() / ".simple-rag"

        self.storage_path = storage_path
        self.embedding_model = embedding_model

        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.info("Using storage path: %s", self.storage_path)
        except Exception as e:
            logger.error("Failed to create storage directory %s: %s", self.storage_path, str(e))
            raise VectorStoreError(f"Failed to create storage directory: {str(e)}") from e

        try:
            self.embedding_function = TransformerEmbeddingFunction(embedding_model)
            self.client = chromadb.PersistentClient(path=str(self.storage_path))
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize ChromaDB client: %s", str(e))
            raise VectorStoreError(f"Failed to initialize ChromaDB client: {str(e)}") from e

    def get_or_create_collection(self, index_name: str) -> Collection:
        """Get or create a collection by index name.

        Arguments:
            index_name: Name of the collection/index

        Returns:
            ChromaDB Collection object

        Exceptions:
            VectorStoreError: If collection creation fails
            ValueError: If index_name is empty or invalid
        """
        if not index_name or not index_name.strip():
            raise ValueError("Index name cannot be empty")

        # Sanitize collection name for ChromaDB
        sanitized_name = index_name.strip().lower().replace(" ", "_").replace("-", "_")

        try:
            collection = self.client.get_or_create_collection(
                name=sanitized_name, embedding_function=self.embedding_function
            )
            logger.debug("Collection '%s' ready", sanitized_name)
            return collection
        except Exception as e:
            logger.error("Failed to get/create collection '%s': %s", sanitized_name, str(e))
            raise VectorStoreError(f"Failed to get/create collection '{sanitized_name}': {str(e)}") from e

    def load_single(self, index_name: str, key: str, value: str) -> None:
        """Load a single key-value pair into the specified index.

        Arguments:
            index_name: Name of the index/collection
            key: Text to be vectorized and stored
            value: Associated value/metadata

        Exceptions:
            VectorStoreError: If loading fails
            ValueError: If key or value is empty
            DuplicateEntryError: If the key already exists
        """
        if not key or not key.strip():
            raise ValueError("Key cannot be empty")
        if not value or not value.strip():
            raise ValueError("Value cannot be empty")

        collection = self.get_or_create_collection(index_name)

        # Generate unique ID based on key
        doc_id = f"{index_name}_{hash(key.strip()) % (10**8):08d}"

        try:
            # Check if document already exists
            existing = collection.get(ids=[doc_id])
            if existing["ids"]:
                logger.warning("Key '%s' already exists in index '%s'", key, index_name)
                raise DuplicateEntryError(f"Key '{key}' already exists in index '{index_name}'")

            # Add the document
            collection.add(
                documents=[key.strip()],
                metadatas=[{"type": index_name, "value": value.strip(), "key": key.strip()}],
                ids=[doc_id],
            )

            logger.info("Successfully loaded key '%s' into index '%s'", key, index_name)

        except DuplicateEntryError:
            raise
        except Exception as e:
            logger.error("Failed to load key '%s' into index '%s': %s", key, index_name, str(e))
            raise VectorStoreError(f"Failed to load data into index '{index_name}': {str(e)}") from e

    def load_from_jsonl(self, index_name: str, file_path: Path) -> int:
        """Load multiple key-value pairs from a JSONL file.

        Arguments:
            index_name: Name of the index/collection
            file_path: Path to JSONL file with {"key": "...", "value": "..."} lines

        Returns:
            Number of successfully loaded entries

        Exceptions:
            VectorStoreError: If file reading or loading fails
            ValueError: If file format is invalid
            FileNotFoundError: If the file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        collection = self.get_or_create_collection(index_name)
        documents, metadatas, ids = self._parse_jsonl_file(file_path, index_name)

        if not documents:
            raise ValueError("No valid entries found in file")

        try:
            # Batch add to ChromaDB
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            loaded_count = len(documents)
            logger.info("Loaded %d entries into index '%s'", loaded_count, index_name)
            return loaded_count

        except Exception as e:
            logger.error("Failed to load from file %s: %s", file_path, str(e))
            raise VectorStoreError(f"Failed to load from file {file_path}: {str(e)}") from e

    def _parse_jsonl_file(self, file_path: Path, index_name: str) -> tuple:
        """Parse JSONL file and return documents, metadatas, and ids.
        
        Arguments:
            file_path: Path to JSONL file
            index_name: Name of the index
            
        Returns:
            Tuple of (documents, metadatas, ids)
        """
        documents = []
        metadatas = []
        ids = []
        error_count = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    entry_data = self._parse_jsonl_line(line, line_num)
                    if entry_data is None:
                        error_count += 1
                        continue

                    key, value = entry_data
                    doc_id = f"{index_name}_{hash(key) % (10**8):08d}"

                    documents.append(key)
                    metadatas.append({"type": index_name, "value": value, "key": key})
                    ids.append(doc_id)

            if error_count > 0:
                logger.warning("Encountered %d errors while parsing file", error_count)

        except Exception as e:
            logger.error("Failed to parse file %s: %s", file_path, str(e))
            raise VectorStoreError(f"Failed to parse file {file_path}: {str(e)}") from e

        return documents, metadatas, ids

    def _parse_jsonl_line(self, line: str, line_num: int) -> tuple:
        """Parse a single JSONL line.
        
        Arguments:
            line: JSON line to parse
            line_num: Line number for error reporting
            
        Returns:
            Tuple of (key, value) or None if parsing fails
        """
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                logger.warning("Line %d: Expected object, got %s", line_num, type(data))
                return None

            if "key" not in data or "value" not in data:
                logger.warning("Line %d: Missing required 'key' or 'value' field", line_num)
                return None

            key = str(data["key"]).strip()
            value = str(data["value"]).strip()

            if not key or not value:
                logger.warning("Line %d: Empty key or value", line_num)
                return None

            return key, value

        except json.JSONDecodeError as e:
            logger.warning("Line %d: Invalid JSON - %s", line_num, str(e))
            return None

    def retrieve(self, index_name: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar documents using cosine distance.

        Arguments:
            index_name: Name of the index/collection to search
            query: Text query to search for
            n_results: Maximum number of results to return

        Returns:
            List of dictionaries containing search results with keys:
            - id: Document ID
            - document: Original text
            - metadata: Associated metadata
            - distance: Cosine distance (lower is more similar)

        Exceptions:
            CollectionNotFoundError: If the specified index doesn't exist
            VectorStoreError: If search fails
            ValueError: If query is empty or n_results is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if n_results <= 0:
            raise ValueError("n_results must be positive")

        try:
            collection = self.get_or_create_collection(index_name)

            # Check if collection has any documents
            count = collection.count()
            if count == 0:
                logger.warning("Index '%s' is empty", index_name)
                return []

            # Perform similarity search with type filter
            results = collection.query(
                query_texts=[query.strip()], n_results=min(n_results, count), where={"type": index_name}
            )

            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None,
                }
                formatted_results.append(result)

            logger.info("Found %d results for query in index '%s'", len(formatted_results), index_name)
            return formatted_results

        except Exception as e:
            logger.error("Failed to retrieve from index '%s': %s", index_name, str(e))
            if "does not exist" in str(e).lower():
                raise CollectionNotFoundError(f"Index '{index_name}' does not exist") from e
            raise VectorStoreError(f"Failed to retrieve from index '{index_name}': {str(e)}") from e

    def list_collections(self) -> List[str]:
        """List all available collections/indexes.

        Returns:
            List of collection names

        Exceptions:
            VectorStoreError: If listing fails
        """
        try:
            collections = self.client.list_collections()
            names = [col.name for col in collections]
            logger.debug("Found %d collections: %s", len(names), names)
            return names
        except Exception as e:
            logger.error("Failed to list collections: %s", str(e))
            raise VectorStoreError(f"Failed to list collections: {str(e)}") from e

    def get_collection_info(self, index_name: str) -> Dict[str, Any]:
        """Get information about a specific collection.

        Arguments:
            index_name: Name of the collection

        Returns:
            Dictionary with collection information

        Exceptions:
            CollectionNotFoundError: If collection doesn't exist
            VectorStoreError: If operation fails
        """
        try:
            collection = self.get_or_create_collection(index_name)
            count = collection.count()

            return {
                "name": index_name,
                "document_count": count,
                "embedding_model": self.embedding_model,
                "storage_path": str(self.storage_path),
            }
        except Exception as e:
            logger.error("Failed to get collection info for '%s': %s", index_name, str(e))
            raise VectorStoreError(f"Failed to get collection info: {str(e)}") from e
