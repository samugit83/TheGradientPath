"""
Example Usage of Dartboard RAG System

This script demonstrates how to use the ingestion and retrieval modules together
to create a complete Dartboard RAG system.
"""

import os
from ingestion import DocumentIngestion
from retrieval import DartboardRetrieval

def download_sample_data():
    """
    Download sample PDF data for testing.
    """
    import urllib.request
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download sample PDF
    url = "https://cdn.openai.com/papers/gpt-4-system-card.pdf"
    pdf_path = "data/gpt-4-system-card.pdf"
    
    if not os.path.exists(pdf_path):
        print("Downloading sample PDF...")
        urllib.request.urlretrieve(url, pdf_path)
        print(f"Downloaded to {pdf_path}")
    else:
        print(f"PDF already exists at {pdf_path}")
    
    return pdf_path

def main():
    """
    Complete example of the Dartboard RAG workflow.
    """
    print("=== Dartboard RAG Example ===\n")
    
    # Step 1: Download or prepare data
    pdf_path = download_sample_data()
    
    # Step 2: Initialize ingestion system
    print("Initializing document ingestion...")
    ingestion = DocumentIngestion()
    
    # Step 3: Create vector store
    print("Creating vector store...")
    vector_store = ingestion.encode_pdf(
        path=pdf_path,
        chunk_size=1000,
        chunk_overlap=200,
        density_multiplier=5  # Simulate dense dataset to show dartboard benefits
    )
    
    # Step 4: Save vector store for future use
    save_path = "vector_store"
    ingestion.save_vector_store(vector_store, save_path)
    
    # Step 5: Initialize retrieval system
    print("\nInitializing Dartboard retrieval...")
    retrieval = DartboardRetrieval(vector_store)
    
    # Step 6: Test queries
    test_queries = [
        "How do the two-stage training process limitations connect to the specific red teaming evaluation methods used, and what role did external expert selection bias play in identifying deployment risks?",
        "What is the relationship between GPT-4's improved coherence capabilities, the increased believability of hallucinations, and the specific content policy violations found during quantitative evaluations?",
        "How do the cybersecurity risks identified through expert red teaming relate to both the economic impact concerns and the autonomous replication evaluation conducted by the Alignment Research Center?",
        "What connections exist between the model's tendency toward sycophancy during training, the representation biases found in evaluations, and the specific harmful content mitigation strategies implemented between GPT-4-early and GPT-4-launch?",
        "How do the limitations of current safety mitigations relate to the anticipatory planning needs, the iterative deployment strategy, and the specific areas where expert red teamers had educational and geographic biases?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}: {query}")
        print('='*60)
        
        # Compare simple vs dartboard retrieval
        retrieval.compare_retrievals(query, k=3)
        
        if i < len(test_queries):
            input("\nPress Enter to continue to next query...")
    
    # Step 7: Demonstrate weight adjustment
    print(f"\n{'='*60}")
    print("DEMONSTRATING WEIGHT ADJUSTMENT")
    print('='*60)
    
    query = test_queries[0]
    print(f"Query: {query}\n")
    
    # Default weights
    print("--- Default Weights (Diversity=1.0, Relevance=1.0) ---")
    texts1, scores1 = retrieval.get_context_with_dartboard(query, num_results=3)
    retrieval.show_context(texts1, scores1)
    
    # Higher diversity weight
    print("\n--- Higher Diversity Weight (Diversity=3.0, Relevance=1.0) ---")
    retrieval.update_weights(diversity_weight=3.0, relevance_weight=1.0, sigma=0.1)
    texts2, scores2 = retrieval.get_context_with_dartboard(query, num_results=3)
    retrieval.show_context(texts2, scores2)
    
    # Higher relevance weight
    print("\n--- Higher Relevance Weight (Diversity=1.0, Relevance=3.0) ---")
    retrieval.update_weights(diversity_weight=1.0, relevance_weight=3.0, sigma=0.1)
    texts3, scores3 = retrieval.get_context_with_dartboard(query, num_results=3)
    retrieval.show_context(texts3, scores3)
    
    print("\n" + "="*60)
    print("Example completed! The vector store has been saved for future use.")
    print("You can now use the ingestion and retrieval modules independently.")
    print("="*60)

if __name__ == "__main__":
    main() 