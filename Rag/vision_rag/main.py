import logging
from dotenv import load_dotenv

# Local imports
from ingestion import UnifiedIngestionPipe
from query import RagQuery
from config import DEFAULT_CONFIG

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to initialize and run the Vision-RAG system"""
    
    # Initialize components
    logger.info("🚀 Initializing Vision-RAG system...")
    
    unified_ingestion = UnifiedIngestionPipe()
    rag_query = RagQuery()
    
    text_count = 0
    image_count = 0
    
    # Process all files from docs folder (if activated)
    if DEFAULT_CONFIG.activate_ingestion:
        logger.info("🔄 Processing all files from docs folder...")
        result = unified_ingestion.process_all_files()
        
        text_count = len(result["text_doc_ids"])
        image_count = len(result["image_doc_ids"])
        
        logger.info(f"✅ Processing completed: {text_count} text chunks, {image_count} images")
    else:
        logger.info("⏭️  Ingestion process is disabled in config")
    
    # Example queries (if activated)
    if DEFAULT_CONFIG.activate_query:
        logger.info("\n🔍 Running example query...")
        query_1 = """Find me a floor plan that includes a tea store and a gaming space. 
        What are the adjacent or connected rooms? Expand the search step by step, describing each adjacent node."""
        query_2 = "Describe a floor plan of an apartment that includes Álvaro Siza’s dining chairs."
        query_3 = "There’s a photo showing a kitchen with red chairs... I can’t remember what color the sofa is, could you help me?"
        query_4 = "What would Tesla's net profit be without interest expenses?"
        query_result = rag_query.query(query_4)
        print(f"\nAnswer: {query_result['answer']}")
        
        # Show context information
        if query_result['text_context']:
            print(f"\nText sources used: {len(query_result['text_context'])} chunks")
            for i, text_ctx in enumerate(query_result['text_context'], 1):
                print(f"  {i}. {text_ctx['source_file']} (similarity: {text_ctx['similarity']:.3f})")
        
        if query_result['image_context']:
            print(f"\nImage sources used: {len(query_result['image_context'])} images")
            for i, img_ctx in enumerate(query_result['image_context'], 1):
                print(f"  {i}. {img_ctx['image_name']} from {img_ctx['source_file']} (similarity: {img_ctx['similarity']:.3f})")
    else:
        logger.info("⏭️  Query process is disabled in config")


if __name__ == "__main__":
    main()
