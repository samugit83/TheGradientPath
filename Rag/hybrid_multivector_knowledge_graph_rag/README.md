# Knowledge Graph Extraction with Neo4j

This project extracts knowledge graphs from text documents using LangChain, OpenAI, and Neo4j database.

## Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- OpenAI API key

## Setup Instructions

### 1. Clone and Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Start Neo4j Database with Docker

#### Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15-community
    container_name: neo4j-kg
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      - neo4j_plugins:/plugins
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_db_tx__log_rotation_retention__policy=1 files
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
```

Then run:

```bash
# Start Neo4j with Docker Compose
docker-compose up -d

# Check if it's running
docker-compose ps
```

### 3. Verify Neo4j Installation

1. Open your browser and go to: http://localhost:7474
2. Login with:
   - Username: `neo4j`
   - Password: `password`
3. You should see the Neo4j Browser interface

### 4. Configure Environment Variables

Set your OpenAI API key:

```bash
# Option 1: Export in terminal
export OPENAI_API_KEY="your-openai-api-key-here"

# Option 2: Create .env file (recommended)
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
echo "NEO4J_URI=bolt://localhost:7687" >> .env
echo "NEO4J_USERNAME=neo4j" >> .env
echo "NEO4J_PASSWORD=password" >> .env
```

### 5. Prepare Your Data

- **Single file mode**: Place your text content in `corpus.txt`
- **Multiple files mode**: Place your documents in the `docs/` folder and change `source_text = "sync_folder"` in `params.py`

### 6. Configure Parameters

Edit `params.py` to customize your knowledge graph extraction:

```python
# Choose data source
source_text = "corpus"  # or "sync_folder"

# Define allowed entities and relationships
allowed_nodes = ["Person", "Country", "Organization"]
allowed_relationships = ["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"]

# Configure relationship tuples for precise control
allowed_relationships_tuples = [
    ("Person", "SPOUSE", "Person"),
    ("Person", "NATIONALITY", "Country"),
    ("Person", "WORKED_AT", "Organization"),
]
```

### 7. Run the Knowledge Graph Extraction

```bash
# Make sure Neo4j is running
docker-compose ps

# Make sure virtual environment is activated
source venv/bin/activate

# Run the extraction
python ingestion.py
```

## Usage

### Starting the System

```bash
# 1. Start Neo4j
docker-compose up -d

# 2. Activate Python environment
source venv/bin/activate

# 3. Run extraction
python ingestion.py
```

### Stopping the System

```bash
# Stop Neo4j (keeps data)
docker-compose stop

# Stop and remove containers (keeps volumes/data)
docker-compose down

# Stop and remove everything including data (CAUTION!)
docker-compose down -v
```

### Accessing Neo4j

- **Neo4j Browser**: http://localhost:7474
- **Bolt connection**: bolt://localhost:7687
- **Credentials**: neo4j/password

### Useful Neo4j Queries

```cypher
// View all nodes
MATCH (n) RETURN n LIMIT 25

// View all relationships
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25

// Count nodes by type
MATCH (n) RETURN labels(n) as NodeType, count(n) as Count

// Count relationships by type
MATCH ()-[r]->() RETURN type(r) as RelationType, count(r) as Count

// Find specific entities
MATCH (p:Person) WHERE p.name CONTAINS "Einstein" RETURN p

// Clear all data (CAUTION!)
MATCH (n) DETACH DELETE n
```

## Project Structure

```
.
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── params.py             # Configuration parameters
├── ingestion.py          # Main extraction script
├── corpus.txt            # Single text file (if using corpus mode)
├── docs/                 # Multiple documents folder (if using sync_folder mode)
└── docker-compose.yml    # Neo4j Docker configuration
```

## Troubleshooting

### Neo4j Connection Issues

```bash
# Check if Neo4j container is running
docker ps | grep neo4j

# Check Neo4j logs
docker logs neo4j-kg

# Restart Neo4j
docker-compose restart
```

### Python Environment Issues

```bash
# Recreate virtual environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### OpenAI API Issues

- Verify your API key is correct
- Check your OpenAI account has sufficient credits
- Ensure the API key has proper permissions

## Configuration Options

### Graph Schema Customization

In `params.py`, you can control:

- **Entity types**: What kinds of nodes to extract
- **Relationship types**: What kinds of connections to find
- **Relationship tuples**: Precise source→relationship→target constraints
- **Node properties**: Additional attributes to extract
- **Source tracking**: Whether to include document source information

### Performance Tuning

- Use `baseEntityLabel = True` for better query performance
- Adjust chunk sizes for large documents
- Consider using Neo4j Enterprise for production workloads

## License

This project is for educational purposes. Please ensure you comply with OpenAI's usage policies and Neo4j's licensing terms.

