"""
Dartboard RAG - Retrieval Module

This module implements the core dartboard retrieval algorithm that balances relevance and diversity
in document retrieval for Retrieval-Augmented Generation systems.

Based on the paper: "Better RAG using Relevant Information Gain"
"""

import os
import numpy as np
from typing import Tuple, List, Any, Optional
from scipy.special import logsumexp
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# OpenAI imports
from openai import OpenAI

class DartboardRetrieval:
    """
    Implements the Dartboard RAG retrieval algorithm that balances relevance and diversity.
    """
    
    def __init__(self, vector_store: FAISS, 
                 diversity_weight: float = 1.0, 
                 relevance_weight: float = 1.0, 
                 sigma: float = 0.1):
        """
        Initialize the Dartboard retrieval system.
        
        Args:
            vector_store: FAISS vector store containing document embeddings
            diversity_weight: Weight for diversity in document selection
            relevance_weight: Weight for relevance to query
            sigma: Smoothing parameter for probability distribution
        """
        self.vector_store = vector_store
        self.diversity_weight = diversity_weight
        self.relevance_weight = relevance_weight
        self.sigma = max(sigma, 1e-5)  # Avoid division by zero
        
        # Initialize embeddings
        load_dotenv()
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
    
    @staticmethod
    def lognorm(dist: np.ndarray, sigma: float) -> np.ndarray:
        """
        Calculate the log-normal probability for given distances and sigma.
        
        Args:
            dist: Array of distances
            sigma: Standard deviation parameter
            
        Returns:
            Log-normal probabilities
        """
        if sigma < 1e-9:
            return -np.inf * dist
        return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist**2 / (2 * sigma**2)
    
    def idx_to_text(self, idx: int) -> str:
        """
        Convert a vector store index to the corresponding text.
        
        Args:
            idx: Document index in the vector store
            
        Returns:
            Document text content
        """
        docstore_id = self.vector_store.index_to_docstore_id[idx]
        document = self.vector_store.docstore.search(docstore_id)
        return document.page_content
    
    def greedy_dartboard_search(self, 
                               query_distances: np.ndarray,
                               document_distances: np.ndarray,
                               documents: List[str],
                               num_results: int) -> Tuple[List[str], List[float]]:
        """
        Perform greedy dartboard search to select top k documents balancing relevance and diversity.
        
        Args:
            query_distances: Distance between query and each document
            document_distances: Pairwise distances between documents
            documents: List of document texts
            num_results: Number of documents to return
        
        Returns:
            Tuple containing:
            - List of selected document texts
            - List of selection scores for each document
        """
        # Convert distances to probability distributions
        query_probabilities = self.lognorm(query_distances, self.sigma)
        document_probabilities = self.lognorm(document_distances, self.sigma)
        
        # Initialize with most relevant document
        most_relevant_idx = np.argmax(query_probabilities)
        selected_indices = np.array([most_relevant_idx])
        selection_scores = [1.0]  # dummy score for the first document
        
        # Get initial distances from the first selected document
        max_distances = document_probabilities[most_relevant_idx]
        
        # Select remaining documents
        while len(selected_indices) < num_results:
            # Update maximum distances considering new document
            updated_distances = np.maximum(max_distances, document_probabilities)
            
            # Calculate combined diversity and relevance scores
            combined_scores = (
                updated_distances * self.diversity_weight +
                query_probabilities * self.relevance_weight
            )
            
            # Normalize scores and mask already selected documents
            normalized_scores = logsumexp(combined_scores, axis=1)
            normalized_scores[selected_indices] = -np.inf
            
            # Select best remaining document
            best_idx = np.argmax(normalized_scores)
            best_score = np.max(normalized_scores)
            
            # Update tracking variables
            max_distances = updated_distances[best_idx]
            selected_indices = np.append(selected_indices, best_idx)
            selection_scores.append(best_score)
        
        # Return selected documents and their scores
        selected_documents = [documents[i] for i in selected_indices]
        return selected_documents, selection_scores
    
    def get_context_simple(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve top k context items for a query using simple top-k retrieval.
        
        Args:
            query: Search query string
            k: Number of documents to retrieve
            
        Returns:
            List of document texts
        """
        # Get query embedding
        query_embedding = self.vector_store.embedding_function.embed_documents([query])
        
        # Search vector store
        _, indices = self.vector_store.index.search(np.array(query_embedding), k=k)
        
        # Convert indices to texts
        texts = [self.idx_to_text(i) for i in indices[0]]
        return texts
    
    def get_context_with_dartboard(self, 
                                  query: str,
                                  num_results: int = 5,
                                  oversampling_factor: int = 3) -> Tuple[List[str], List[float]]:
        """
        Retrieve most relevant and diverse context items for a query using the dartboard algorithm.
        
        Args:
            query: The search query string
            num_results: Number of context items to return
            oversampling_factor: Factor to oversample initial results for better diversity
        
        Returns:
            Tuple containing:
            - List of selected context texts
            - List of selection scores
        """
        # Embed query and retrieve initial candidates
        query_embedding = self.vector_store.embedding_function.embed_documents([query])
        query_embedding = np.array(query_embedding)
        
        # Get more candidates than needed for better diversity
        num_candidates = num_results * oversampling_factor
        _, candidate_indices = self.vector_store.index.search(query_embedding, k=num_candidates)
        
        # Get document vectors and texts for candidates
        candidate_vectors = np.array(
            self.vector_store.index.reconstruct_batch(candidate_indices[0])
        )
        candidate_texts = [self.idx_to_text(idx) for idx in candidate_indices[0]]
        
        # Calculate distance matrices
        # Using 1 - cosine_similarity as distance metric
        document_distances = 1 - np.dot(candidate_vectors, candidate_vectors.T)
        query_distances = 1 - np.dot(query_embedding, candidate_vectors.T).flatten()
        
        # Apply dartboard selection algorithm
        selected_texts, selection_scores = self.greedy_dartboard_search(
            query_distances,
            document_distances,
            candidate_texts,
            num_results
        )
        
        return selected_texts, selection_scores
    
    def show_context(self, texts: List[str], scores: Optional[List[float]] = None):
        """
        Display context texts in a formatted way.
        
        Args:
            texts: List of document texts to display
            scores: Optional list of scores for each text
        """
        for i, text in enumerate(texts):
            score_info = f" (Score: {scores[i]:.4f})" if scores else ""
            print(f"\nContext {i+1}{score_info}:")
            print(text[:500] + ("..." if len(text) > 500 else ""))
            print("-" * 80)
    
    def generate_answer(self, query: str, context_texts: List[str]) -> str:
        """
        Generate an answer using OpenAI based on the query and retrieved context.
        
        Args:
            query: The user's question
            context_texts: List of retrieved context documents
            
        Returns:
            Generated answer from OpenAI
        """
        # Combine context texts
        combined_context = "\n\n".join([f"Context {i+1}:\n{text}" for i, text in enumerate(context_texts)])
        
        # Create prompt
        prompt = f"""Based on the following context documents, please answer the question. Your answer must be well-articulated and include all the pertinent information from the context. Use only the information provided in the context to answer the question. If the context doesn't contain enough information to answer the question, please say so.

Context:
{combined_context}

Question: {query}

Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def compare_retrievals(self, query: str, k: int = 3):
        """
        Compare simple retrieval vs dartboard retrieval for the same query.
        
        Args:
            query: Search query string
            k: Number of documents to retrieve
        """
        print("=== SIMPLE RETRIEVAL ===")
        simple_texts = self.get_context_simple(query, k)
        self.show_context(simple_texts)
        
        # Generate answer for simple retrieval
        print("\n--- SIMPLE RETRIEVAL ANSWER ---")
        simple_answer = self.generate_answer(query, simple_texts)
        print(f"Answer: {simple_answer}")
        
        print("\n=== DARTBOARD RETRIEVAL ===")
        dartboard_texts, scores = self.get_context_with_dartboard(query, k)
        self.show_context(dartboard_texts, scores)
        
        # Generate answer for dartboard retrieval
        print("\n--- DARTBOARD RETRIEVAL ANSWER ---")
        dartboard_answer = self.generate_answer(query, dartboard_texts)
        print(f"Answer: {dartboard_answer}")
    
    def update_weights(self, diversity_weight: float, relevance_weight: float, sigma: float):
        """
        Update the weights and parameters for the dartboard algorithm.
        
        Args:
            diversity_weight: New diversity weight
            relevance_weight: New relevance weight
            sigma: New sigma parameter
        """
        self.diversity_weight = diversity_weight
        self.relevance_weight = relevance_weight
        self.sigma = max(sigma, 1e-5)
        print(f"Updated weights - Diversity: {diversity_weight}, Relevance: {relevance_weight}, Sigma: {sigma}")


def main():
    """
    Example usage of the Dartboard retrieval system.
    """
    # Load vector store (assuming it exists from ingestion)
    try:
        from ingestion import DocumentIngestion
        
        ingestion = DocumentIngestion()
        vector_store = ingestion.load_vector_store("vector_store")
        
        # Initialize dartboard retrieval
        retrieval = DartboardRetrieval(vector_store)
        
        # Example query
        test_query = "What is the main cause of climate change?"
        
        print(f"Query: {test_query}\n")
        
        # Compare simple vs dartboard retrieval
        retrieval.compare_retrievals(test_query, k=3)
        
        # Example of adjusting weights
        print("\n=== ADJUSTING WEIGHTS ===")
        retrieval.update_weights(diversity_weight=2.0, relevance_weight=1.0, sigma=0.2)
        
        dartboard_texts, scores = retrieval.get_context_with_dartboard(test_query, num_results=3)
        print("Results with higher diversity weight:")
        retrieval.show_context(dartboard_texts, scores)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have run the ingestion script first to create the vector store.")


if __name__ == "__main__":
    main()
