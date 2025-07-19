#!/usr/bin/env python3
"""
Database initialization script for Vision-RAG with pgvector
"""

import os
import psycopg2
import argparse
import logging
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_database_connection():
    """Check if database is accessible and properly configured"""
    try:
        connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
        if not connection_string:
            logger.error("❌ POSTGRES_CONNECTION_STRING not found in environment")
            return False
        
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Check if database is accessible
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        logger.info(f"✅ Connected to PostgreSQL: {version}")
        
        # Check if pgvector extension is installed
        cursor.execute("SELECT EXISTS (SELECT FROM pg_extension WHERE extname = 'vector');")
        vector_exists = cursor.fetchone()[0]
        
        if vector_exists:
            logger.info("✅ pgvector extension is installed")
        else:
            logger.error("❌ pgvector extension is not installed")
            return False
        
        # Check if tables exist
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name IN ('text_embeddings', 'image_embeddings')
            );
        """)
        
        tables_exist = cursor.fetchone()[0]
        
        if tables_exist:
            logger.info("✅ Vision-RAG tables are created")
            
            # Check table structures
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'text_embeddings' 
                AND column_name = 'embedding';
            """)
            
            text_embedding_info = cursor.fetchone()
            if text_embedding_info and 'vector' in text_embedding_info[1].lower():
                logger.info("✅ Text embeddings table uses VECTOR type")
            else:
                logger.warning("⚠️  Text embeddings table may not be using VECTOR type")
            
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'image_embeddings' 
                AND column_name = 'embedding';
            """)
            
            image_embedding_info = cursor.fetchone()
            if image_embedding_info and 'vector' in image_embedding_info[1].lower():
                logger.info("✅ Image embeddings table uses VECTOR type")
            else:
                logger.warning("⚠️  Image embeddings table may not be using VECTOR type")
        else:
            logger.error("❌ Vision-RAG tables are not created")
            return False
        
        # Check indexes
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename IN ('text_embeddings', 'image_embeddings')
            AND indexname LIKE '%embedding%cosine%';
        """)
        
        indexes = cursor.fetchall()
        if indexes:
            logger.info(f"✅ Found {len(indexes)} pgvector cosine indexes")
        else:
            logger.warning("⚠️  No pgvector cosine indexes found")
        
        cursor.close()
        conn.close()
        
        logger.info("🎉 Database health check passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database health check failed: {e}")
        return False


def setup_database():
    """Set up the database with pgvector and required tables"""
    try:
        connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
        if not connection_string:
            logger.error("❌ POSTGRES_CONNECTION_STRING not found in environment")
            return False
        
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        logger.info("🔧 Setting up Vision-RAG database with pgvector...")
        
        # Read and execute the initialization SQL
        init_sql_path = os.path.join(os.path.dirname(__file__), 'init_db.sql')
        
        if os.path.exists(init_sql_path):
            with open(init_sql_path, 'r') as f:
                init_sql = f.read()
            
            # Execute the initialization script
            cursor.execute(init_sql)
            conn.commit()
            
            logger.info("✅ Database initialization completed successfully")
        else:
            logger.error(f"❌ Initialization SQL file not found: {init_sql_path}")
            return False
        
        cursor.close()
        conn.close()
        
        # Verify the setup
        return check_database_connection()
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Vision-RAG Database Initialization")
    parser.add_argument("--setup", action="store_true", help="Set up the database")
    parser.add_argument("--check", action="store_true", help="Check database health")
    
    args = parser.parse_args()
    
    if not args.setup and not args.check:
        parser.print_help()
        sys.exit(1)
    
    success = True
    
    if args.setup:
        success = setup_database()
    
    if args.check:
        success = check_database_connection()
    
    if success:
        logger.info("🎉 All operations completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some operations failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 