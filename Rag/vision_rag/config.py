"""
Configuration parameters for Vision-RAG system with pgvector integration
"""

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class SemanticChunkerConfig:
    """Configuration for semantic text chunking"""
    
    # Breakpoint threshold type for semantic chunking
    breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile"] = "percentile"
    
    # Threshold value for breakpoints
    breakpoint_threshold_amount: float = 95.0
    
    # Minimum size for any chunk (in characters)
    min_chunk_size: int = 100
    
    # Maximum size for any chunk (in characters)
    max_chunk_size: int = 2000

@dataclass
class RetrievalConfig:
    """Configuration for retrieval settings with pgvector"""
    
    # Number of top text chunks to retrieve
    text_top_k: int = 0
    
    # Number of top images to retrieve
    image_top_k: int = 1
    
    # Similarity threshold for filtering results (note: pgvector uses cosine distance, converted to similarity)
    similarity_threshold: float = 0.1

@dataclass
class VisionRAGConfig:
    """Main configuration for Vision-RAG system with pgvector"""
    
    # Semantic chunker configuration
    semantic_chunker: SemanticChunkerConfig = field(default_factory=SemanticChunkerConfig)
    
    # Retrieval configuration
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # Answer provider: "openai" or "google"
    answer_provider: Literal["openai", "google"] = "google"
    
    # OpenAI model for text generation (vision-capable)
    openai_model: str = "gpt-4o-mini"
    
    # Google Gemini model for vision and text generation
    gemini_model: str = "gemini-2.5-flash"
    
    # Text embedding model (3072 dimensions for pgvector VECTOR(3072))
    text_embedding_model: str = "text-embedding-3-large"
    
    # Image embedding model (1536 dimensions for pgvector VECTOR(1536))
    image_embedding_model: str = "embed-v4.0"  # Cohere
    
    # Maximum tokens for OpenAI responses
    max_tokens: int = 1000
    
    # Temperature for OpenAI responses (not supported by reasoning models like o4-mini, o3-mini, o1, etc.)
    temperature: float = 1.0
    
    # Activate ingestion process
    activate_ingestion: bool = False
    
    # Activate query process
    activate_query: bool = True
    
    # Transform each PDF page as a whole image during ingestion (ignores text/image mix)
    each_pdf_page_as_image: bool = True

# Default configuration instance
DEFAULT_CONFIG = VisionRAGConfig() 