#!/usr/bin/env python3
"""
Generate Vector Embeddings for Database Text Fields
This script generates embeddings for all text fields that have corresponding _embed columns
using OpenAI's text-embedding-3-small model and updates the database.
"""

import os
import sys
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
import time

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'books_db',
    'user': 'bookadmin',
    'password': 'bookpass123'
}

# OpenAI configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


class EmbeddingGenerator:
    """Generate and store embeddings for database text fields."""
    
    def __init__(self):
        """Initialize the embedding generator."""
        self.client = self._initialize_openai()
        self.connection = self._connect_to_database()
        self.total_embeddings = 0
        self.total_cost_estimate = 0.0
        self.tables_with_embeddings = self._discover_embedding_fields()
    
    def _initialize_openai(self) -> OpenAI:
        """Initialize OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        return OpenAI(api_key=api_key)
    
    def _connect_to_database(self):
        """Connect to PostgreSQL database."""
        try:
            return psycopg2.connect(**DB_CONFIG)
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            sys.exit(1)
    
    def _discover_embedding_fields(self) -> Dict[str, List[tuple]]:
        """
        Automatically discover all tables and fields with embeddings.
        
        Returns:
            Dict mapping table names to list of (text_field, embed_field, id_field) tuples
        """
        cursor = self.connection.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        tables_with_embeddings = {}
        
        for table in tables:
            # Get all columns for this table
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table,))
            
            columns = cursor.fetchall()
            
            # Find embedding fields (fields ending with _embed)
            embed_fields = [col[0] for col in columns if col[0].endswith('_embed')]
            
            if not embed_fields:
                continue
            
            # Get primary key for this table
            cursor.execute("""
                SELECT column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY'
                ORDER BY kcu.ordinal_position;
            """, (table,))
            
            pk_result = cursor.fetchone()
            if not pk_result:
                continue
            
            id_field = pk_result[0]
            
            # Map each embedding field to its source text field
            field_mappings = []
            for embed_field in embed_fields:
                # Remove _embed suffix to get text field name
                text_field = embed_field[:-6]  # Remove '_embed'
                
                # Verify text field exists
                if any(col[0] == text_field for col in columns):
                    field_mappings.append((text_field, embed_field, id_field))
            
            if field_mappings:
                tables_with_embeddings[table] = field_mappings
        
        cursor.close()
        return tables_with_embeddings
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using OpenAI API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            return None
        
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
                dimensions=EMBEDDING_DIMENSIONS
            )
            
            embedding = response.data[0].embedding
            
            # Estimate cost (text-embedding-3-small: $0.020 per 1M tokens)
            # Rough estimate: ~1 token per 4 characters
            tokens = len(text) / 4
            self.total_cost_estimate += (tokens / 1_000_000) * 0.020
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def get_rows_to_process(self, table_name: str, text_field: str, embed_field: str, id_field: str) -> List[Dict[str, Any]]:
        """
        Get all rows that need embeddings generated.
        
        Args:
            table_name: Name of the table
            text_field: Name of the text field
            embed_field: Name of the embedding field
            id_field: Name of the ID field
            
        Returns:
            List of dictionaries with row data
        """
        cursor = self.connection.cursor()
        
        # Get rows where text field is not null but embedding is null
        query = f"""
            SELECT {id_field}, {text_field}
            FROM {table_name}
            WHERE {text_field} IS NOT NULL 
              AND {embed_field} IS NULL
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        
        return [{'id': row[0], 'text': row[1]} for row in rows]
    
    def update_embedding(self, table_name: str, embed_field: str, id_field: str, row_id: int, embedding: List[float]) -> bool:
        """
        Update a single row with its embedding.
        
        Args:
            table_name: Name of the table
            embed_field: Name of the embedding field
            id_field: Name of the ID field
            row_id: ID of the row to update
            embedding: Embedding vector to save
            
        Returns:
            bool: True if successful
        """
        cursor = self.connection.cursor()
        
        try:
            # Convert embedding list to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            query = f"""
                UPDATE {table_name}
                SET {embed_field} = %s::vector
                WHERE {id_field} = %s
            """
            
            cursor.execute(query, (embedding_str, row_id))
            self.connection.commit()
            cursor.close()
            return True
            
        except Exception as e:
            print(f"Error updating embedding: {e}")
            self.connection.rollback()
            cursor.close()
            return False
    
    def process_table_field(self, table_name: str, text_field: str, embed_field: str, id_field: str):
        """
        Process all rows for a specific table and field combination.
        
        Args:
            table_name: Name of the table
            text_field: Name of the text field
            embed_field: Name of the embedding field
            id_field: Name of the ID field
        """
        print(f"\nProcessing {table_name}.{text_field} → {embed_field}")
        print("-" * 60)
        
        # Get rows to process
        rows = self.get_rows_to_process(table_name, text_field, embed_field, id_field)
        
        if not rows:
            print(f"  No rows to process (all embeddings already generated)")
            return
        
        print(f"  Found {len(rows)} rows to process")
        
        # Process each row
        for i, row in enumerate(rows, 1):
            # Generate embedding
            embedding = self.generate_embedding(row['text'])
            
            if embedding:
                # Update database
                success = self.update_embedding(table_name, embed_field, id_field, row['id'], embedding)
                
                if success:
                    self.total_embeddings += 1
                    print(f"  [{i}/{len(rows)}] Updated {id_field}={row['id']}")
                else:
                    print(f"  [{i}/{len(rows)}] Failed to update {id_field}={row['id']}")
            else:
                print(f"  [{i}/{len(rows)}] Skipped {id_field}={row['id']} (empty text or error)")
            
            # Small delay to avoid rate limits
            if i % 10 == 0:
                time.sleep(0.5)
    
    def process_all_tables(self):
        """Process all tables with embedding fields."""
        print("=" * 80)
        print("GENERATING EMBEDDINGS FOR ALL TABLES")
        print("=" * 80)
        print(f"Model: {EMBEDDING_MODEL}")
        print(f"Dimensions: {EMBEDDING_DIMENSIONS}")
        print("=" * 80)
        
        # Show discovered tables and fields
        print("\nDiscovered embedding fields:")
        for table_name, fields in self.tables_with_embeddings.items():
            print(f"  {table_name}: {len(fields)} field(s)")
            for text_field, embed_field, id_field in fields:
                print(f"    - {text_field} → {embed_field}")
        print("=" * 80)
        
        # Process each table
        for table_name, fields in self.tables_with_embeddings.items():
            for text_field, embed_field, id_field in fields:
                self.process_table_field(table_name, text_field, embed_field, id_field)
        
        # Summary
        print("\n" + "=" * 80)
        print("EMBEDDING GENERATION COMPLETE")
        print("=" * 80)
        print(f"Total embeddings generated: {self.total_embeddings}")
        print(f"Estimated cost: ${self.total_cost_estimate:.4f}")
        print("=" * 80)
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("\nDatabase connection closed.")


def main():
    """Main function to generate embeddings."""
    try:
        generator = EmbeddingGenerator()
        generator.process_all_tables()
        generator.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

