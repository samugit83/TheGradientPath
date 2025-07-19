-- Vision-RAG Database Initialization Script with pgvector
-- This script creates the necessary tables and indexes for the Vision-RAG system

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table for storing image embeddings with pgvector
CREATE TABLE IF NOT EXISTS image_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_name VARCHAR(255) NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    image_url VARCHAR(1000),
    source_file VARCHAR(500),
    source_type VARCHAR(50) DEFAULT 'image',
    base64_data TEXT,
    mime_type VARCHAR(50),
    embedding VECTOR(1536) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for storing text embeddings with pgvector
CREATE TABLE IF NOT EXISTS text_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text_content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    source_file VARCHAR(500) NOT NULL,
    source_type VARCHAR(50) DEFAULT 'pdf',
    embedding VECTOR(3072) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for image_embeddings using pgvector
CREATE INDEX IF NOT EXISTS idx_image_embeddings_embedding_cosine 
ON image_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_image_embeddings_name 
ON image_embeddings (image_name);

CREATE INDEX IF NOT EXISTS idx_image_embeddings_source 
ON image_embeddings (source_file);

CREATE INDEX IF NOT EXISTS idx_image_embeddings_created_at 
ON image_embeddings (created_at);

-- Create indexes for text_embeddings using pgvector
CREATE INDEX IF NOT EXISTS idx_text_embeddings_embedding_cosine 
ON text_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_text_embeddings_source 
ON text_embeddings (source_file);

CREATE INDEX IF NOT EXISTS idx_text_embeddings_chunk 
ON text_embeddings (chunk_index);

CREATE INDEX IF NOT EXISTS idx_text_embeddings_created_at 
ON text_embeddings (created_at);

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Vision-RAG database with pgvector initialized successfully!';
    RAISE NOTICE 'Created tables: image_embeddings, text_embeddings';
    RAISE NOTICE 'Created pgvector indexes for performance optimization';
    RAISE NOTICE 'Text embeddings: VECTOR(3072), Image embeddings: VECTOR(1536)';
END $$; 