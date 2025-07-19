# Vision-RAG: Multimodal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that processes both text and images from PDFs, storing embeddings in PostgreSQL and enabling multimodal question answering using OpenAI's vision models.

## Features

- **PDF Processing**: Extract text and images from PDF documents
- **Image Processing**: Handle standalone images and images extracted from PDFs
- **Text Embeddings**: Use OpenAI embeddings for text content with semantic chunking
- **Image Embeddings**: Use Cohere's multimodal embeddings for image content
- **Multimodal RAG**: Query both text and images simultaneously
- **Vision-Capable Responses**: Generate answers using OpenAI's vision models with actual images
- **PostgreSQL Storage**: Store embeddings and base64-encoded images in PostgreSQL

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for PostgreSQL only)
- OpenAI API key
- Cohere API key

## Quick Start

### 1. Environment Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
POSTGRES_CONNECTION_STRING=postgresql://username:password@localhost:5432/vision_rag_db
```

### 2. Start PostgreSQL Database

```bash
# Start PostgreSQL with automatic database initialization
docker-compose up -d

# Check that database is running
docker-compose ps
```

The database will be automatically initialized with the required tables and indexes.

### 3. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Add Your Documents

Place your PDF files and images in the `docs/` folder:

```
docs/
├── document1.pdf
├── document2.pdf
├── image1.png
└── subfolder/
    └── image2.jpg
```

### 5. Run the Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the application
python main.py
```

## Usage

### Basic Commands

```bash
# Start PostgreSQL database
docker-compose up -d

# Stop PostgreSQL database
docker-compose down

# View database logs
docker-compose logs -f postgres

# Check database status
docker-compose ps
```

### Python Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run main application (processes docs/ folder and runs example queries)
python main.py


### Database Management

```bash
# Connect to PostgreSQL directly
docker-compose exec postgres psql -U username -d vision_rag_db

# Check table contents
docker-compose exec postgres psql -U username -d vision_rag_db -c "SELECT COUNT(*) FROM text_embeddings;"
docker-compose exec postgres psql -U username -d vision_rag_db -c "SELECT COUNT(*) FROM image_embeddings;"
```

### Complete Cleanup Commands

```bash
# Stop and remove containers with volumes (removes all data)
docker-compose down -v

# Remove project-specific Docker images
docker-compose down --rmi all

# Complete project cleanup (containers, volumes, images, and networks)
docker-compose down -v --rmi all --remove-orphans

# If you want to remove everything Docker Compose created for this project:
docker-compose down -v --rmi all --remove-orphans && docker volume prune -f
```

## System Architecture

### Components

1. **PostgreSQL Database** (Docker): Stores text and image embeddings with metadata
2. **Vision-RAG Application** (Local): Main processing and query engine
3. **PDF Processor**: Extracts text and images from PDFs
4. **Text Ingestion**: Semantic chunking and OpenAI embeddings
5. **Image Ingestion**: Cohere multimodal embeddings
6. **Query Engine**: Multimodal retrieval and OpenAI vision responses

### Database Schema

The database is automatically initialized with:

**text_embeddings table**:
- `id`: Primary key (UUID)
- `text_content`: Chunked text content
- `chunk_index`: Position in original document
- `source_file`: Original file path
- `embedding`: OpenAI text embedding vector
- `metadata`: Additional metadata (JSON)
- `created_at`: Timestamp

**image_embeddings table**:
- `id`: Primary key (UUID)
- `image_name`: Image filename
- `image_path`: Original file path
- `source_file`: Source document (if extracted from PDF)
- `base64_data`: Base64-encoded image data
- `mime_type`: Image MIME type
- `embedding`: Cohere multimodal embedding vector
- `metadata`: Additional metadata (JSON)
- `created_at`: Timestamp

## Usage Examples

### Processing Documents

```python
from ingestion import UnifiedIngestionPipe

# Process all files in docs/ folder
pipe = UnifiedIngestionPipe()
result = pipe.process_all_files()

print(f"Processed {len(result['text_doc_ids'])} text chunks")
print(f"Processed {len(result['image_doc_ids'])} images")
```

### Querying the System

```python
from query import RagQuery

# Initialize query system
rag = RagQuery()

# Ask questions
result = rag.query("What information can you extract from the documents?")
print(f"Answer: {result['answer']}")

# Access context
print(f"Found {len(result['text_context'])} relevant text chunks")
print(f"Found {len(result['image_context'])} relevant images")
```

### Processing Specific Files

```python
from ingestion import TextIngestionPipe, ImageIngestionPipe

# Process text only
text_pipe = TextIngestionPipe()
text_ids = text_pipe.process_text("Your text content here", "source_file.txt")

# Process images only
image_pipe = ImageIngestionPipe()
image_id = image_pipe.process_image("path/to/image.png", "source_file.pdf")
```

### Advanced Usage

```python
from ingestion import UnifiedIngestionPipe
from query import RagQuery

# Initialize components
ingestion = UnifiedIngestionPipe(docs_folder="custom_docs")
query_engine = RagQuery()

# Process documents
result = ingestion.process_all_files()

# Custom query with specific parameters
similar_texts = query_engine.search_similar_texts("machine learning", top_k=5)
similar_images = query_engine.search_similar_images("charts and graphs", top_k=3)

# Generate answer with custom context
answer = query_engine.generate_answer("Explain the data", similar_texts, similar_images)
```

## Configuration

The system uses configuration from `config.py`. Key settings:

- **OpenAI Model**: `gpt-4o-mini` (vision-capable)
- **Text Chunking**: Semantic chunking with configurable thresholds
- **Retrieval**: Configurable top-k for both text and images
- **Temperature**: Response randomness control

## Troubleshooting

### Common Issues

1. **Database Connection**: Ensure PostgreSQL is running with `docker-compose ps`
2. **API Key Errors**: Check `.env` file has correct API keys
3. **Import Errors**: Ensure virtual environment is activated and dependencies installed
4. **Port Conflicts**: PostgreSQL runs on port 5432, ensure it's not in use

### Debug Commands

```bash
# Check database connection
docker-compose exec postgres psql -U username -d vision_rag_db -c "SELECT version();"

# View database logs
docker-compose logs -f postgres

# Check Python environment
python -c "import psycopg2, cohere, openai; print('All imports successful')"
```

### Reset Database

```bash
# Stop and remove database with all data
docker-compose down -v

# Start fresh database
docker-compose up -d
```

## Development

### Local Development

```bash
# Activate virtual environment
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt

# Run application
python main.py
```

### Adding New Features

1. Modify source code
2. Test locally: `python main.py`
3. No container rebuilding needed!

## Performance Optimization

- **Batch Processing**: Process multiple documents simultaneously
- **Embedding Caching**: Embeddings are stored in PostgreSQL for reuse
- **Image Optimization**: Images are automatically resized for optimal processing
- **Database Indexing**: Automatic GIN indexes for vector similarity search

## Security Considerations

- API keys are loaded from environment variables
- Database credentials should be changed in production
- PostgreSQL is only exposed on localhost:5432
- Consider using connection pooling for production

## File Structure

```
vision_rag/
├── main.py                 # Main application entry point
├── ingestion.py           # Document ingestion classes
├── query.py               # Query and retrieval classes
├── config.py              # Configuration settings
├── pdf_processor.py       # PDF processing utilities
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # PostgreSQL setup
├── init_db.sql           # Database initialization
├── .env                  # Environment variables (create this)
└── docs/                 # Place your documents here
    ├── document1.pdf
    └── image1.png
```

### Module Overview

- **main.py**: Entry point that initializes ingestion and query components
- **ingestion.py**: Contains all document processing classes:
  - `ImageProcessor`: Image utility functions
  - `TextIngestionPipe`: Text processing and embedding
  - `ImageIngestionPipe`: Image processing and embedding
  - `UnifiedIngestionPipe`: Unified PDF and image processing
- **query.py**: Contains the query and retrieval system:
  - `RagQuery`: Multimodal search and answer generation
- **config.py**: Configuration management
- **pdf_processor.py**: PDF text and image extraction utilities

## License

This project is open source. Please refer to the license file for details. 