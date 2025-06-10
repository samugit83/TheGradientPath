"""
Dartboard RAG - Document Ingestion Module

This module handles the ingestion of PDF documents, chunking them into smaller pieces,
and creating vector stores using FAISS and OpenAI embeddings.
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class DocumentIngestion:
    """
    Handles document ingestion and vector store creation for the Dartboard RAG system.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the document ingestion system.
        
        Args:
            openai_api_key: OpenAI API key. If not provided, will load from environment or prompt user.
        """
        # Load environment variables
        load_dotenv()
        
        # Set up OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.getenv('OPENAI_API_KEY'):
            api_key = input("Please enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
    
    @staticmethod
    def replace_t_with_space(docs):
        """
        Replace tab characters with spaces in document content.
        
        Args:
            docs: List of documents
            
        Returns:
            List of documents with cleaned content
        """
        for doc in docs:
            doc.page_content = doc.page_content.replace('\t', ' ')
        return docs
    
    def encode_pdf(self, path: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                   density_multiplier: int = 1) -> FAISS:
        """
        Encodes a PDF document into a vector store using OpenAI embeddings.
        
        Args:
            path: The path to the PDF file
            chunk_size: The desired size of each text chunk
            chunk_overlap: The amount of overlap between consecutive chunks
            density_multiplier: Multiplier to simulate dense dataset (for testing)
            
        Returns:
            A FAISS vector store containing the encoded document content
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found at path: {path}")
        
        print(f"Loading PDF from: {path}")
        
        # Load PDF documents
        loader = PyPDFLoader(path)
        documents = loader.load()
        
        # Multiply documents to simulate dense dataset if needed
        if density_multiplier > 1:
            print(f"Multiplying documents by {density_multiplier} to simulate dense dataset")
            documents = documents * density_multiplier
        
        print(f"Loaded {len(documents)} document pages")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks")
        
        # Clean texts by replacing tabs with spaces
        cleaned_texts = self.replace_t_with_space(texts)
        
        # Create vector store
        print("Creating vector store with embeddings...")
        vectorstore = FAISS.from_documents(cleaned_texts, self.embeddings)
        
        print("Vector store created successfully!")
        return vectorstore
    
    def save_vector_store(self, vectorstore: FAISS, save_path: str):
        """
        Save the vector store to disk.
        
        Args:
            vectorstore: The FAISS vector store to save
            save_path: Path where to save the vector store
        """
        print(f"Saving vector store to: {save_path}")
        vectorstore.save_local(save_path)
        print("Vector store saved successfully!")
    
    def load_vector_store(self, load_path: str) -> FAISS:
        """
        Load a vector store from disk.
        
        Args:
            load_path: Path from where to load the vector store
            
        Returns:
            The loaded FAISS vector store
        """
        print(f"Loading vector store from: {load_path}")
        vectorstore = FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully!")
        return vectorstore


def main():
    """
    Example usage of the document ingestion system.
    """
    # Initialize ingestion system
    ingestion = DocumentIngestion()
    
    # Example: Encode a PDF file
    pdf_path = "data/Understanding_Climate_Change.pdf"
    
    try:
        # Create vector store
        vectorstore = ingestion.encode_pdf(
            path=pdf_path,
            chunk_size=1000,
            chunk_overlap=200,
            density_multiplier=5  # Simulate dense dataset
        )
        
        # Save vector store
        save_path = "vector_store"
        ingestion.save_vector_store(vectorstore, save_path)
        
        # Example: Load vector store
        loaded_vectorstore = ingestion.load_vector_store(save_path)
        
        print(f"Vector store contains {vectorstore.index.ntotal} vectors")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")


if __name__ == "__main__":
    main()
