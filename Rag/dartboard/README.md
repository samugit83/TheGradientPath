https://arxiv.org/html/2407.12101v2


# Dartboard RAG: Retrieval-Augmented Generation with Balanced Relevance and Diversity

A production-ready implementation of the Dartboard RAG algorithm that addresses redundancy in document retrieval by optimizing both relevance and diversity.

## Overview

The Dartboard RAG process addresses a common challenge in large knowledge bases: ensuring the retrieved information is both relevant and non-redundant. By explicitly optimizing a combined relevance-diversity scoring function, it prevents multiple documents from offering the same information.

This implementation is based on the paper: **"Better RAG using Relevant Information Gain"**

## Key Features

- **Relevance & Diversity Balance**: Combines document relevance to the query with diversity among selected documents
- **Configurable Weights**: Adjustable `RELEVANCE_WEIGHT` and `DIVERSITY_WEIGHT` for dynamic control
- **Production Ready**: Clean, modular code design for easy integration
- **Multiple Retrieval Modes**: Support for simple top-k and advanced dartboard retrieval

## Installation

1. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Quick Start

### 1. Run the Complete Example

```bash
python main.py
```

This will:
- Download sample data
- Create a vector store
- Demonstrate both simple and dartboard retrieval
- Show the effects of different weight configurations

### 2. Use Individual Modules

#### Document Ingestion

```python
from ingestion import DocumentIngestion

# Initialize ingestion system
ingestion = DocumentIngestion()

# Create vector store from PDF
vector_store = ingestion.encode_pdf(
    path="path/to/your/document.pdf",
    chunk_size=1000,
    chunk_overlap=200,
    density_multiplier=1  # Increase to simulate dense datasets
)

# Save for later use
ingestion.save_vector_store(vector_store, "my_vector_store")
```

#### Dartboard Retrieval

```python
from retrieval import DartboardRetrieval
from ingestion import DocumentIngestion

# Load existing vector store
ingestion = DocumentIngestion()
vector_store = ingestion.load_vector_store("my_vector_store")

# Initialize retrieval with custom weights
retrieval = DartboardRetrieval(
    vector_store=vector_store,
    diversity_weight=1.0,
    relevance_weight=1.0,
    sigma=0.1
)

# Perform dartboard retrieval
query = "What is climate change?"
texts, scores = retrieval.get_context_with_dartboard(
    query=query,
    num_results=5,
    oversampling_factor=3
)

# Compare with simple retrieval
retrieval.compare_retrievals(query, k=5)
```

## Algorithm Details

### Core Components

1. **Document Retrieval**: Initial candidate selection using similarity search
2. **Distance Calculation**: Compute distances between query-documents and document-document pairs
3. **Dartboard Selection**: Iteratively select documents balancing relevance and diversity
4. **Score Combination**: `combined_score = diversity_weight * diversity + relevance_weight * relevance`

### Key Parameters

- **`diversity_weight`**: Controls importance of diversity (default: 1.0)
- **`relevance_weight`**: Controls importance of relevance (default: 1.0)
- **`sigma`**: Smoothing parameter for probability conversion (default: 0.1)
- **`oversampling_factor`**: Multiplier for initial candidate retrieval (default: 3)

### When to Use Dartboard RAG

- **Dense Knowledge Bases**: When documents contain overlapping information
- **Comprehensive Answers**: When you need diverse perspectives on a topic
- **Avoiding Echo Chambers**: When simple top-k retrieval returns repetitive content

## File Structure

```
Rag/dartboard/
├── requirements.txt      # Python dependencies
├── ingestion.py         # Document processing and vector store creation
├── retrieval.py         # Dartboard retrieval algorithm
├── main.py     # Complete workflow demonstration
└── README.md           # This file
```

## Configuration Examples

### High Diversity (Explore Different Topics)
```python
retrieval.update_weights(
    diversity_weight=3.0,
    relevance_weight=1.0,
    sigma=0.1
)
```

### High Relevance (Focus on Query Match)
```python
retrieval.update_weights(
    diversity_weight=1.0,
    relevance_weight=3.0,
    sigma=0.1
)
```

### Balanced Approach
```python
retrieval.update_weights(
    diversity_weight=1.5,
    relevance_weight=1.5,
    sigma=0.15
)
```

## Performance Considerations

- **Oversampling Factor**: Higher values provide better diversity but increase computation
- **Vector Store Size**: Larger stores benefit more from dartboard selection
- **Query Complexity**: Complex queries may benefit from higher relevance weights

## Integration with Other Systems

The dartboard retrieval can be easily integrated with:

- **Hybrid Retrieval**: Combine dense and sparse (BM25) similarities
- **Cross-Encoders**: Use cross-encoder scores directly
- **Custom Embeddings**: Replace OpenAI embeddings with any embedding provider

## Troubleshooting

### Common Issues

1. **"Vector store not found"**: Run ingestion first or check the save path
2. **OpenAI API errors**: Verify your API key in the `.env` file
3. **Memory issues**: Reduce `oversampling_factor` or use smaller chunks

### Performance Tips

- Pre-compute and save vector stores for large documents
- Adjust chunk size based on your document type
- Use appropriate density_multiplier for testing vs production

## Contributing

Feel free to submit issues and pull requests. This implementation aims to be production-ready while maintaining clarity and ease of use.

## References

- Original paper: "Better RAG using Relevant Information Gain"
- Based on the official implementation but reorganized for production use
- LangChain integration for document processing and embeddings

## License

This implementation is provided as-is for educational and commercial use. 