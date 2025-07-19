import os
import io
import base64
import numpy as np
import psycopg2
import PIL.Image
import cohere
from typing import List, Dict, Optional
from dotenv import load_dotenv
import json
import glob
from pathlib import Path
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import pgvector.psycopg2

# Local imports
from pdf_processor import PDFProcessor
from config import DEFAULT_CONFIG

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Helper class for image processing utilities"""
    
    MAX_PIXELS = 1568 * 1568
    
    @staticmethod
    def resize_image(pil_image: PIL.Image.Image) -> None:
        """Resize image if it's too large"""
        org_width, org_height = pil_image.size
        
        if org_width * org_height > ImageProcessor.MAX_PIXELS:
            scale_factor = (ImageProcessor.MAX_PIXELS / (org_width * org_height)) ** 0.5
            new_width = int(org_width * scale_factor)
            new_height = int(org_height * scale_factor)
            pil_image.thumbnail((new_width, new_height))
    
    @staticmethod
    def base64_from_image(img_path: str) -> str:
        """Convert image to base64 string"""
        pil_image = PIL.Image.open(img_path)
        img_format = pil_image.format if pil_image.format else "PNG"
        
        ImageProcessor.resize_image(pil_image)
        
        with io.BytesIO() as img_buffer:
            pil_image.save(img_buffer, format=img_format)
            img_buffer.seek(0)
            img_data = f"data:image/{img_format.lower()};base64," + base64.b64encode(img_buffer.read()).decode("utf-8")
        
        return img_data


class TextIngestionPipe:
    """Handles text ingestion with semantic chunking"""
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
        
        # Initialize LangChain components - using text-embedding-3-large for 3072 dimensions
        self.chunker_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.semantic_chunker = SemanticChunker(
            embeddings=self.chunker_embeddings,
            breakpoint_threshold_type=self.config.semantic_chunker.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.config.semantic_chunker.breakpoint_threshold_amount
        )
        
        # Verify database connection
        self._verify_database_connection()
    
    def _verify_database_connection(self):
        """Verify that database connection is working"""
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Check if text_embeddings table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'text_embeddings'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logger.error("❌ text_embeddings table not found!")
                logger.info("💡 Please run database initialization first: docker compose up -d")
                raise Exception("Database not properly initialized")
            
            # Check if pgvector extension is installed
            cursor.execute("SELECT EXISTS (SELECT FROM pg_extension WHERE extname = 'vector');")
            vector_exists = cursor.fetchone()[0]
            
            if not vector_exists:
                logger.error("❌ pgvector extension not found!")
                logger.info("💡 Please ensure pgvector is installed in PostgreSQL")
                raise Exception("pgvector extension not available")
            
            cursor.close()
            conn.close()
            logger.info("✅ Text database connection and pgvector verified")
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text using semantic chunker"""
        try:
            if not text.strip():
                return []
            
            # Create documents from text
            documents = self.semantic_chunker.create_documents([text])
            
            # Extract text content from documents
            chunks = [doc.page_content for doc in documents]
            
            # Filter out very small chunks
            chunks = [chunk for chunk in chunks if len(chunk) >= self.config.semantic_chunker.min_chunk_size]
            
            logger.info(f"✅ Created {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error chunking text: {e}")
            return []
    
    def compute_text_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text using OpenAI embeddings (3072 dimensions)"""
        try:
            embedding = self.chunker_embeddings.embed_query(text)
            embedding_array = np.array(embedding)
            
            # Verify dimension
            if len(embedding_array) != 3072:
                logger.warning(f"⚠️  Expected 3072 dimensions, got {len(embedding_array)}")
            
            return embedding_array
        except Exception as e:
            logger.error(f"❌ Error computing text embedding: {e}")
            return np.array([])
    
    def store_text_embedding(self, text_content: str, chunk_index: int, source_file: str, 
                           embedding: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """Store text embedding in PostgreSQL using pgvector"""
        try:
            conn = psycopg2.connect(self.connection_string)
            pgvector.psycopg2.register_vector(conn)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO text_embeddings (text_content, chunk_index, source_file, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
            """, (text_content, chunk_index, source_file, embedding, json.dumps(metadata) if metadata else None))
            
            doc_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            return str(doc_id)
            
        except Exception as e:
            logger.error(f"❌ Error storing text embedding: {e}")
            return None
    
    def process_text(self, text: str, source_file: str) -> List[str]:
        """Process text by chunking and storing embeddings"""
        if not text.strip():
            logger.warning(f"⚠️  Empty text content for {source_file}")
            return []
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        if not chunks:
            logger.warning(f"⚠️  No chunks created for {source_file}")
            return []
        
        doc_ids = []
        
        for chunk_index, chunk in enumerate(chunks):
            try:
                # Compute embedding
                embedding = self.compute_text_embedding(chunk)
                
                if embedding.size == 0:
                    logger.warning(f"⚠️  Empty embedding for chunk {chunk_index}")
                    continue
                
                # Create metadata
                metadata = {
                    "chunk_length": len(chunk),
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                    "source_file": source_file
                }
                
                # Store in database
                doc_id = self.store_text_embedding(chunk, chunk_index, source_file, embedding, metadata)
                
                if doc_id:
                    doc_ids.append(doc_id)
                    logger.info(f"✅ Stored text chunk {chunk_index + 1}/{len(chunks)} from {source_file}")
                
            except Exception as e:
                logger.error(f"❌ Error processing chunk {chunk_index}: {e}")
        
        return doc_ids


class ImageIngestionPipe:
    """Handles image ingestion with Cohere embeddings"""
    
    def __init__(self, docs_folder: str = "docs"):
        self.cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        self.connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
        self.docs_folder = docs_folder
        
        # Verify database connection
        self._verify_database_connection()
    
    def _verify_database_connection(self):
        """Verify that database connection is working"""
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Check if image_embeddings table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'image_embeddings'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logger.error("❌ image_embeddings table not found!")
                logger.info("💡 Please run database initialization first: docker compose up -d")
                raise Exception("Database not properly initialized")
            
            # Check if pgvector extension is installed
            cursor.execute("SELECT EXISTS (SELECT FROM pg_extension WHERE extname = 'vector');")
            vector_exists = cursor.fetchone()[0]
            
            if not vector_exists:
                logger.error("❌ pgvector extension not found!")
                logger.info("💡 Please ensure pgvector is installed in PostgreSQL")
                raise Exception("pgvector extension not available")
            
            cursor.close()
            conn.close()
            logger.info("✅ Image database connection and pgvector verified")
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def compute_image_embedding(self, img_path: str) -> np.ndarray:
        """Compute embedding for an image using Cohere Embed v4 (1536 dimensions)"""
        try:
            base64_data = ImageProcessor.base64_from_image(img_path)
            
            # Use the 'images' parameter approach (working with current SDK version)
            api_response = self.cohere_client.embed(
                model="embed-v4.0",
                images=[base64_data],
                input_type="image",
                embedding_types=["float"],
            )
            
            embedding_array = np.asarray(api_response.embeddings.float[0])
            
            # Verify dimension
            if len(embedding_array) != 1536:
                logger.warning(f"⚠️  Expected 1536 dimensions, got {len(embedding_array)}")
            
            return embedding_array
        
        except Exception as e:
            logger.error(f"❌ Error computing image embedding: {e}")
            return np.array([])
    
    def store_image_embedding(self, image_name: str, image_path: str, source_file: str,
                            embedding: np.ndarray, base64_data: str, mime_type: str, 
                            metadata: Optional[Dict] = None) -> str:
        """Store image embedding with base64 data in PostgreSQL using pgvector"""
        try:
            conn = psycopg2.connect(self.connection_string)
            pgvector.psycopg2.register_vector(conn)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO image_embeddings (image_name, image_path, source_file, base64_data, mime_type, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (image_name, image_path, source_file, base64_data, mime_type, embedding, json.dumps(metadata) if metadata else None))
            
            doc_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            return str(doc_id)
            
        except Exception as e:
            logger.error(f"❌ Error storing image embedding: {e}")
            return None
    
    def process_image(self, image_path: str, source_file: str = None) -> str:
        """Process a single image and store its embedding with base64 data"""
        try:
            image_name = os.path.basename(image_path)
            
            # Read and encode image as base64
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            base64_data = base64.b64encode(img_bytes).decode()
            
            # Determine MIME type
            file_ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.tiff': 'image/tiff',
                '.webp': 'image/webp'
            }.get(file_ext, 'image/jpeg')
            
            # Compute embedding
            embedding = self.compute_image_embedding(image_path)
            
            if embedding.size == 0:
                logger.warning(f"⚠️  Empty embedding for {image_name}")
                return None
            
            # Create metadata
            metadata = {
                "file_size": os.path.getsize(image_path),
                "source_file": source_file or image_path,
                "image_name": image_name,
                "mime_type": mime_type
            }
            
            # Store in database with base64 data
            doc_id = self.store_image_embedding(
                image_name, image_path, source_file or image_path, 
                embedding, base64_data, mime_type, metadata
            )
            
            if doc_id:
                logger.info(f"✅ Processed image {image_name}")
                return doc_id
            
        except Exception as e:
            logger.error(f"❌ Error processing image {image_path}: {e}")
            
        return None
    
    def process_images(self, image_paths: List[str], source_file: str = None) -> List[str]:
        """Process multiple images"""
        doc_ids = []
        
        for image_path in image_paths:
            doc_id = self.process_image(image_path, source_file)
            if doc_id:
                doc_ids.append(doc_id)
        
        return doc_ids


class UnifiedIngestionPipe:
    """Unified ingestion pipeline for both PDFs and images"""
    
    def __init__(self, docs_folder: str = "docs", config=None):
        self.docs_folder = docs_folder
        self.config = config or DEFAULT_CONFIG
        
        # Initialize processors
        self.pdf_processor = PDFProcessor()
        self.text_ingestion = TextIngestionPipe(config)
        self.image_ingestion = ImageIngestionPipe(docs_folder)
        
        # Create extracted images directory
        self.extracted_images_dir = os.path.join(docs_folder, "extracted_images")
        os.makedirs(self.extracted_images_dir, exist_ok=True)
    
    def get_all_files_from_docs(self) -> Dict[str, List[str]]:
        """Get all files from the docs folder, separated by type"""
        if not os.path.exists(self.docs_folder):
            logger.info(f"Creating docs folder: {self.docs_folder}")
            os.makedirs(self.docs_folder, exist_ok=True)
            return {"pdfs": [], "images": []}
        
        # PDF files
        pdf_files = glob.glob(os.path.join(self.docs_folder, "**", "*.pdf"), recursive=True)
        
        # Image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        
        #SUbfolder search
        for extension in image_extensions:
            pattern = os.path.join(self.docs_folder, '**', extension)
            image_files.extend(glob.glob(pattern, recursive=True))
        
        # Filter out images from extracted_images directory to avoid duplicates
        image_files = [img for img in image_files if "extracted_images" not in img]
        
        logger.info(f"Found {len(pdf_files)} PDFs and {len(image_files)} images in {self.docs_folder}")
        
        return {
            "pdfs": pdf_files,
            "images": image_files
        }
    
    def process_all_files(self) -> Dict[str, List[str]]:
        """Process all files from the docs folder"""
        files = self.get_all_files_from_docs()
        
        text_doc_ids = []
        image_doc_ids = []
        
        # Process PDFs
        for pdf_path in files["pdfs"]:
            logger.info(f"🔄 Processing PDF: {pdf_path}")
            
            if self.config.each_pdf_page_as_image:
                # Process each PDF page as a whole image
                logger.info(f"📄➡️🖼️  Converting PDF pages to images: {pdf_path}")
                page_image_ids = self._process_pdf_as_page_images(pdf_path)
                image_doc_ids.extend(page_image_ids)
            else:
                # Standard processing: extract text and images separately
                pdf_result = self.pdf_processor.process_pdf(pdf_path, self.extracted_images_dir)
                
                # Process text
                if pdf_result["text_content"]:
                    text_ids = self.text_ingestion.process_text(pdf_result["text_content"], pdf_path)
                    text_doc_ids.extend(text_ids)
                
                # Process images
                if pdf_result["image_paths"]:
                    image_ids = self.image_ingestion.process_images(pdf_result["image_paths"], pdf_path)
                    image_doc_ids.extend(image_ids)
        
        # Process standalone images
        for image_path in files["images"]:
            logger.info(f"🔄 Processing image: {image_path}")
            image_id = self.image_ingestion.process_image(image_path)
            if image_id:
                image_doc_ids.append(image_id)
        
        return {
            "text_doc_ids": text_doc_ids,
            "image_doc_ids": image_doc_ids
        }
    
    def _process_pdf_as_page_images(self, pdf_path: str) -> List[str]:
        """
        Process PDF by converting each page to an image
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of document IDs for processed page images
        """
        try:
            from pdf2image import convert_from_path
            
            # Get PDF filename without extension
            pdf_name = Path(pdf_path).stem
            
            # Convert PDF pages to images
            pages = convert_from_path(
                pdf_path,
                dpi=200,  # Good quality for vision models
                fmt='PNG'
            )
            
            image_doc_ids = []
            
            for page_num, page in enumerate(pages):
                try:
                    # Save page as image
                    img_filename = f"{pdf_name}_page{page_num + 1}_full.png"
                    img_path = os.path.join(self.extracted_images_dir, img_filename)
                    
                    page.save(img_path, 'PNG')
                    
                    # Process the page image
                    doc_id = self.image_ingestion.process_image(img_path, pdf_path)
                    if doc_id:
                        image_doc_ids.append(doc_id)
                        logger.info(f"✅ Processed PDF page {page_num + 1}/{len(pages)} as image: {pdf_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing page {page_num + 1} of {pdf_path}: {e}")
            
            logger.info(f"✅ Converted {len(pages)} pages to images for {pdf_path}")
            return image_doc_ids
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF as page images {pdf_path}: {e}")
            return []
