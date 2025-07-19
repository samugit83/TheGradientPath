import os
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import cohere
from typing import List, Dict
from dotenv import load_dotenv
import logging
import base64
import io
from PIL import Image
import pgvector.psycopg2

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# Google GenAI imports
from google import genai

# Local imports
from config import DEFAULT_CONFIG

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RagQuery:
    """Handles querying with both text and image retrieval using LangChain and pgvector"""
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
        
        # Initialize LangChain components - using text-embedding-3-large for 3072 dimensions
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        
        # Initialize based on answer provider
        if self.config.answer_provider == "openai":
            # Initialize ChatOpenAI with proper parameters for the model
            self.llm = init_chat_model(self.config.openai_model, model_provider="openai")
        elif self.config.answer_provider == "google":
            # Initialize Google GenAI client
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when using Google provider")
            self.gemini_client = genai.Client(api_key=gemini_api_key)
    
    def search_similar_texts(self, question: str, top_k: int = 3) -> List[Dict]:
        """Search for similar text chunks using pgvector cosine similarity"""
        try:
            # Compute embedding for the query
            query_embedding = np.array(self.embeddings.embed_query(question))
            
            # Verify embedding dimension
            if len(query_embedding) != 3072:
                logger.warning(f"⚠️  Expected 3072 dimensions for text query, got {len(query_embedding)}")
            
            # Search in PostgreSQL using pgvector cosine distance
            conn = psycopg2.connect(self.connection_string)
            pgvector.psycopg2.register_vector(conn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT id, text_content, chunk_index, source_file, metadata,
                       (embedding <=> %s) as distance
                FROM text_embeddings
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not results:
                return []
            
            # Convert distance to similarity (1 - distance for cosine)
            similarities = []
            for result in results:
                similarity = 1 - result['distance']  # Convert cosine distance to similarity
                similarities.append({
                    'id': result['id'],
                    'text_content': result['text_content'],
                    'chunk_index': result['chunk_index'],
                    'source_file': result['source_file'],
                    'metadata': result['metadata'],
                    'similarity': similarity
                })
            
            logger.info(f"✅ Found {len(similarities)} similar text chunks using pgvector")
            return similarities
            
        except Exception as e:
            logger.error(f"❌ Error searching similar texts with pgvector: {e}")
            return []
    
    def search_similar_images(self, question: str, top_k: int = 2) -> List[Dict]:
        """Search for similar images using Cohere embeddings and pgvector cosine similarity"""
        try:
            # Use Cohere for image search
            cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
            
            api_response = cohere_client.embed(
                model="embed-v4.0",
                input_type="search_query",
                embedding_types=["float"],
                texts=[question],
            )
            
            query_embedding = np.asarray(api_response.embeddings.float[0])
            
            # Verify embedding dimension
            if len(query_embedding) != 1536:
                logger.warning(f"⚠️  Expected 1536 dimensions for image query, got {len(query_embedding)}")
            
            # Search in PostgreSQL using pgvector cosine distance
            conn = psycopg2.connect(self.connection_string)
            pgvector.psycopg2.register_vector(conn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT id, image_name, image_path, source_file, base64_data, mime_type, metadata,
                       (embedding <=> %s) as distance
                FROM image_embeddings
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not results:
                return []
            
            # Convert distance to similarity (1 - distance for cosine)
            similarities = []
            for result in results:
                similarity = 1 - result['distance']  # Convert cosine distance to similarity
                similarities.append({
                    'id': result['id'],
                    'image_name': result['image_name'],
                    'image_path': result['image_path'],
                    'source_file': result['source_file'],
                    'base64_data': result['base64_data'],
                    'mime_type': result['mime_type'],
                    'metadata': result['metadata'],
                    'similarity': similarity
                })
            
            logger.info(f"✅ Found {len(similarities)} similar images using pgvector")
            return similarities
            
        except Exception as e:
            logger.error(f"❌ Error searching similar images with pgvector: {e}")
            return []
    
    def _base64_to_pil_image(self, base64_data: str) -> Image.Image:
        """Convert base64 data to PIL Image"""
        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_data)
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            return image
        except Exception as e:
            logger.error(f"❌ Error converting base64 to PIL Image: {e}")
            return None
    
    def generate_answer_with_gemini(self, question: str, text_context: List[Dict], image_context: List[Dict]) -> str:
        """Generate answer using Google Gemini with both text and image context"""
        try:
            logger.info("🤖 Starting Gemini answer generation...")
            
            # Build the prompt content
            prompt_parts = []
            
            # Add text context
            if text_context:
                text_content = "Text Context:\n"
                for i, text_item in enumerate(text_context, 1):
                    text_content += f"Text {i} (from {text_item['source_file']}, similarity: {text_item['similarity']:.3f}):\n{text_item['text_content']}\n\n"
                prompt_parts.append(text_content)
            
            # Add the main prompt
            main_prompt = f"""Answer the question based on the provided text and image context.
Don't use markdown.
Please provide enough context for your answer.

Question: {question}"""
            
            prompt_parts.append(main_prompt)
            
            # Combine text parts
            prompt_text = "\n".join(prompt_parts)
            
            # Build the full prompt with images
            full_prompt = [prompt_text]
            
            # Add images from context
            if image_context:
                for img_item in image_context:
                    try:
                        # Convert base64 to PIL Image
                        pil_image = self._base64_to_pil_image(img_item['base64_data'])
                        if pil_image:
                            full_prompt.append(pil_image)
                            logger.info(f"✅ Added image {img_item['image_name']} (similarity: {img_item['similarity']:.3f}) to prompt")
                    except Exception as e:
                        logger.error(f"❌ Error processing image {img_item['image_name']}: {e}")
            
            logger.info("🔄 Calling Gemini API...")
            # Generate response using Gemini
            response = self.gemini_client.models.generate_content(
                model=self.config.gemini_model,
                contents=full_prompt
            )
            
            logger.info("✅ Gemini API call completed successfully")
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Error generating answer with Gemini: {e}")
            return "Sorry, I couldn't generate an answer for this question."
    
    def generate_answer_with_openai(self, question: str, text_context: List[Dict], image_context: List[Dict]) -> str:
        """Generate answer using OpenAI with text context (original method)"""
        try:
            logger.info("🤖 Starting OpenAI answer generation...")
            
            # Build the content for the human message
            content_parts = []
            
            # Add text context
            if text_context:
                text_content = "Text Context:\n"
                for i, text_item in enumerate(text_context, 1):
                    text_content += f"Text {i} (from {text_item['source_file']}, similarity: {text_item['similarity']:.3f}):\n{text_item['text_content']}\n\n"
                content_parts.append(text_content)
            
            # Add image context info (OpenAI vision models can handle images but this implementation shows text context)
            if image_context:
                image_info = "Image Context:\n"
                for i, img_item in enumerate(image_context, 1):
                    image_info += f"Image {i}: {img_item['image_name']} from {img_item['source_file']} (similarity: {img_item['similarity']:.3f})\n"
                content_parts.append(image_info)
            
            # Add the question
            content_parts.append(f"Question: {question}")
            
            # For now, we'll combine text content only (images require vision models)
            human_content = "\n".join(content_parts)
            
            # Create system message
            system_message = SystemMessage(
                content="You are a helpful assistant that answers questions based on provided context. "
                       "You have access to both text content and images. Use both sources to provide comprehensive answers."
            )
            
            # Create human message
            human_message = HumanMessage(content=human_content)
            
            logger.info("🔄 Calling OpenAI API...")
            # Generate response using ChatOpenAI with invoke method
            response = self.llm.invoke([system_message, human_message])
            logger.info("✅ OpenAI API call completed successfully")
            return response.content
            
        except Exception as e:
            logger.error(f"❌ Error generating answer with OpenAI: {e}")
            return "Sorry, I couldn't generate an answer for this question."
    
    def generate_answer(self, question: str, text_context: List[Dict], image_context: List[Dict]) -> str:
        """Generate answer using the configured provider (OpenAI or Google)"""
        if self.config.answer_provider == "google":
            return self.generate_answer_with_gemini(question, text_context, image_context)
        elif self.config.answer_provider == "openai":
            return self.generate_answer_with_openai(question, text_context, image_context)
        else:
            logger.error(f"❌ Unsupported answer provider: {self.config.answer_provider}")
            return "Sorry, unsupported answer provider configuration."
    
    def query(self, question: str) -> Dict:
        """Complete query flow: search both text and images, then generate answer"""
        logger.info(f"🔍 Query: {question}")
        
        # Search for similar texts and images using pgvector
        similar_texts = self.search_similar_texts(question, self.config.retrieval.text_top_k)
        similar_images = self.search_similar_images(question, self.config.retrieval.image_top_k)
        
        logger.info(f"📊 Found {len(similar_texts)} text chunks and {len(similar_images)} images using pgvector")
        
        if not similar_texts and not similar_images:
            return {
                "question": question,
                "answer": "No relevant content found.",
                "text_context": [],
                "image_context": []
            }
        
        # Generate answer
        answer = self.generate_answer(question, similar_texts, similar_images)
        
        return {
            "question": question,
            "answer": answer,
            "text_context": similar_texts,
            "image_context": similar_images
        }
