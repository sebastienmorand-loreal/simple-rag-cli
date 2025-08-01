# Simple RAG CLI

A lightweight Retrieval Augmented Generation (RAG) CLI tool using ChromaDB for local vector storage. This tool allows you to vectorize text data and perform semantic similarity search using cosine distance.

## Features

- **Local Vector Storage**: Uses ChromaDB with storage in `~/.simple-rag`
- **Custom Embedding Model**: Uses `w601sxs/b1ade-embed` for high-quality embeddings
- **CLI Interface**: Easy-to-use command-line interface built with Typer
- **Multiple Loading Methods**: Single key-value pairs or batch loading from JSONL files
- **Semantic Search**: Retrieve similar documents using cosine distance
- **Rich Output**: Beautiful formatted output with Rich library
- **Comprehensive Error Handling**: Robust error management with detailed logging

## Installation

### Prerequisites

- Python 3.11 or higher
- UV package manager

### Install Dependencies

```bash
uv sync
```

### Install the CLI Tool

```bash
uv run pip install -e .
```

## Usage

### Loading Data

#### Single Entry Loading

Load a single key-value pair into an index:

```bash
simple-rag load --index projects "SDDS Project" "itg-btdppublished-gbl-ww-pd"
```

#### Batch Loading from JSONL File

Create a JSONL file with key-value pairs:

```json
{"key": "SDDS project", "value": "itg-btdppublished-gbl-ww-pd"}
{"key": "Frontend project", "value": "itg-btdpfront-gbl-ww-pd"}
{"key": "Backend project", "value": "itg-btdpback-gbl-ww-pd"}
```

Load the file:

```bash
simple-rag load --index projects -f data.jsonl
```

### Retrieving Data

Search for similar documents using semantic similarity:

```bash
simple-rag retrieve --index projects "SDDS project"
```

With additional options:

```bash
# Filter by distance threshold and limit results
simple-rag retrieve --index projects "SDDS project" --threshold 0.5 --number 3

# Output in JSON format with key, value, and distance
simple-rag retrieve --index projects "SDDS project" --json
```

### Managing Indexes

#### List All Indexes

```bash
simple-rag list
```

#### Get Index Information

```bash
simple-rag info --index projects
```

### Command Reference

#### `load` Command

Load data into the vector store.

**Options:**
- `--index`: Index name to store the data (required)
- `-f, --file`: JSONL file path for batch loading
- `key`: Key text to vectorize (for single entry)
- `value`: Associated value (for single entry)

**Examples:**
```bash
# Single entry
simple-rag load --index projects "SDDS Project" "itg-btdppublished-gbl-ww-pd"

# Batch loading
simple-rag load --index projects -f data.jsonl
```

#### `retrieve` Command

Retrieve similar documents using RAG search.

**Options:**
- `--index`: Index name to search in (required)
- `--threshold, -t`: Distance threshold to filter results (optional)
- `--number, -n`: Number of results to display (default: 1)
- `--json, -j`: Output in JSON format with key, value, and distance (optional)

**Examples:**
```bash
# Basic search (returns just the value)
simple-rag retrieve --index projects "SDDS project"

# Filter by threshold and get multiple results
simple-rag retrieve --index projects "SDDS project" --threshold 0.5 --number 3

# JSON output with detailed information
simple-rag retrieve --index projects "SDDS project" --json
```

#### `list` Command

List all available indexes and their information.

```bash
simple-rag list
```

#### `info` Command

Get detailed information about a specific index.

**Options:**
- `--index`: Index name to get information about (required)

**Example:**
```bash
simple-rag info --index projects
```

### Global Options

- `--debug`: Enable debug logging to stdout

## Architecture

### Components

1. **Vector Store Service** (`commons.database.vector_store`): Manages ChromaDB operations
2. **Custom Embedding Function** (`commons.models.embedding`): Handles text embedding using transformer models
3. **CLI Interface** (`rag.cli`): Command-line interface with rich formatting
4. **Main Entry Point** (`main`): Application orchestration and logging setup

### Data Storage

- **Storage Location**: `~/.simple-rag/`
- **Embedding Model**: `w601sxs/b1ade-embed`
- **Vector Database**: ChromaDB with persistent storage
- **Similarity Metric**: Cosine distance

### Error Handling

The application includes comprehensive error handling for:

- **VectorStoreError**: Base exception for vector store operations
- **CollectionNotFoundError**: When specified index doesn't exist
- **DuplicateEntryError**: When attempting to add duplicate entries
- **FileNotFoundError**: When JSONL file doesn't exist
- **ValueError**: For invalid input parameters
- **RuntimeError**: For embedding generation failures
- **OSError**: For model loading failures

### Recovery Procedures

| Error Type | Cause | Solution |
|------------|-------|----------|
| Collection not found | Index doesn't exist | Use `simple-rag list` to see available indexes |
| Duplicate entry | Key already exists | Use `simple-rag retrieve` to see existing data |
| File not found | JSONL file missing | Check file path and ensure file exists |
| Empty index | No data loaded | Use `simple-rag load` to add data first |
| Model loading error | Network/disk issues | Check internet connection and disk space |

## Dependencies

### Core Dependencies

- **typer**: CLI framework
- **chromadb**: Vector database
- **transformers**: HuggingFace transformers library
- **torch**: PyTorch for model operations
- **sentence-transformers**: Sentence embedding models
- **rich**: Rich text and beautiful formatting
- **colorlog**: Colored logging output

### Development Dependencies

- **black**: Code formatting
- **pylint**: Code quality analysis
- **bandit**: Security linting
- **pytest**: Testing framework
- **pytest-cov**: Test coverage

## Development

### Code Quality

The project maintains high code quality standards:

- **Pylint Score**: 9.87/10
- **Security**: No security issues (verified with Bandit)
- **Formatting**: Black with 120 character line length
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings for all functions

### Project Structure

```
src/
├── main.py                     # Main CLI entry point
├── commons/                    # Technical classes
│   ├── database/              # Database wrappers
│   │   └── vector_store.py    # ChromaDB service
│   └── models/                # Model definitions
│       └── embedding.py       # Custom embedding function
└── rag/                       # RAG functionality
    └── cli.py                 # CLI commands
```

### Testing

Run tests with:

```bash
uv run pytest
```

### Linting and Formatting

```bash
# Format code
uv run black -l 120 src/

# Check code quality
uv run pylint src/

# Security scan
uv run bandit -r src/
```

## Permissions Required

### Local System

- **Read/Write Access**: `~/.simple-rag/` directory for vector storage
- **Read Access**: JSONL files for batch loading
- **Network Access**: For downloading embedding models (first run only)

### Python Environment

- **Package Installation**: Access to install required dependencies
- **Model Cache**: `~/.cache/huggingface/` for model storage

## Troubleshooting

### Common Issues

1. **Model Download Errors**: Ensure internet connectivity for first-time model download
2. **Storage Permission Errors**: Check write permissions for `~/.simple-rag/` directory
3. **Memory Issues**: The embedding model requires ~1GB RAM minimum
4. **CUDA Errors**: Model automatically falls back to CPU if CUDA unavailable

### Logging

Logs are automatically saved to `~/.logs/simplerag.log.<datetime>` files. The CLI displays the log file name on stderr for each run.

Enable debug logging for troubleshooting:

```bash
simple-rag --debug <command>
```

## License

MIT License