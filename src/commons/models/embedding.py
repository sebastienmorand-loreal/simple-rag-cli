"""Custom embedding function for ChromaDB using transformer models."""

import logging
import sys
import io
import contextlib
from typing import List
import torch
from chromadb.api.types import EmbeddingFunction, Documents
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_sharding_output():
    """Context manager to suppress transformer model sharding output."""
    # Capture both stdout and stderr to filter out sharding warnings
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()

    try:
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr
        yield
    finally:
        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Filter and print non-sharding content from stdout
        stdout_content = captured_stdout.getvalue()
        for line in stdout_content.splitlines():
            if "layers were not sharded" not in line:
                print(line)

        # Filter and print non-sharding content from stderr
        stderr_content = captured_stderr.getvalue()
        for line in stderr_content.splitlines():
            if "layers were not sharded" not in line:
                print(line, file=original_stderr)


class TransformerEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using SentenceTransformers.

    This class implements the ChromaDB EmbeddingFunction interface
    to use transformer models for generating embeddings.
    """

    def __init__(self, model_name: str = "w601sxs/b1ade-embed", use_cuda: bool = True):
        """Initialize the transformer embedding function.

        Arguments:
            model_name: HuggingFace model identifier for embeddings
            use_cuda: Whether to use CUDA GPU acceleration if available

        Exceptions:
            RuntimeError: If CUDA is requested but not available
            OSError: If the specified model cannot be loaded
            ValueError: If model_name is empty or invalid
        """
        if not model_name:
            raise ValueError("Model name cannot be empty")

        self.model_name = model_name
        self.use_cuda = use_cuda and torch.cuda.is_available()

        if use_cuda and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.use_cuda = False

        try:
            logger.info("Loading embedding model: %s", model_name)
            with suppress_sharding_output():
                self.model = SentenceTransformer(model_name)

            if self.use_cuda:
                self.model = self.model.to("cuda")
                logger.info("Model loaded on CUDA device")
            else:
                logger.info("Model loaded on CPU")

        except Exception as e:
            logger.error("Failed to load embedding model %s: %s", model_name, str(e))
            raise OSError(f"Failed to load embedding model {model_name}") from e

    def __call__(self, input_texts: Documents) -> List[List[float]]:
        """Generate embeddings for input texts.

        Arguments:
            input_texts: List of text documents to embed

        Returns:
            List of embedding vectors as lists of floats

        Exceptions:
            ValueError: If input_texts is empty or contains invalid data
            RuntimeError: If embedding generation fails
        """
        if not input_texts:
            raise ValueError("Input texts cannot be empty")

        if not isinstance(input_texts, list):
            raise ValueError("Input texts must be a list")

        for i, text in enumerate(input_texts):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string, got {type(text)}")
            if not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty or whitespace only")

        try:
            logger.debug("Generating embeddings for %d texts", len(input_texts))
            embeddings = self.model.encode(input_texts, convert_to_tensor=False)

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            result = embeddings.tolist()
            logger.debug("Generated %d embeddings of dimension %d", len(result), len(result[0]) if result else 0)
            return result

        except Exception as e:
            logger.error("Failed to generate embeddings: %s", str(e))
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e
