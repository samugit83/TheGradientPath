import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from params import (
    model, 
    VECTOR_SEARCH_CONFIGURATIONS, 
    TOP_K_INITIAL, 
    TOP_K_TRAVERSAL, 
    ACTIVATE_INITIAL_VECTOR_SEARCH,
    GRAPH_TRAVERSAL_METHOD
)
import json
import logging
from prompts import FINAL_ANSWER_SYSTEM_PROMPT

# Import the traversal classes
from traversal.context_to_cypher import ContextToCypher
from traversal.khop_limited_bfs import KhopLimitedBFS
from traversal.khop_limited_bfs_pred_llm import KhopLimitedBFSWithLLM
from traversal.depth_limited_dfs import DepthLimitedDFS
from traversal.depth_limited_dfs_pred_llm import DepthLimitedDFSWithLLM
from traversal.uniform_cost_search_ucs import UniformCostSearchUCS
from traversal.uniform_cost_search_ucs_pred_llm import UniformCostSearchUCSWithLLM
from traversal.astar_search_heuristic import AStarSearchHeuristic
from traversal.astar_search_heuristic_pred_llm import AStarSearchHeuristicWithLLM
from traversal.beam_search_over_the_graph import BeamSearchOverGraph
from traversal.beam_search_over_the_graph_pred_llm import BeamSearchOverGraphWithLLM

logger = logging.getLogger(__name__)

class HybridRAGQuery:
    """
    A singleton class for performing vector similarity search on Document nodes in Neo4j
    to retrieve relevant context based on user queries.
    Initialized only once when the first instance is created.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Ensure only one instance of HybridRAGQuery exists."""
        if cls._instance is None:
            cls._instance = super(HybridRAGQuery, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the context retrieval system - only runs once."""
        if not HybridRAGQuery._initialized:
            print("🔧 Initializing HybridRAGQuery singleton...")
            
            # Initialize all services immediately
            self._setup_environment()
            self._initialize_embedding_service()
            self._initialize_neo4j_connection()
            
            HybridRAGQuery._initialized = True
            print("✅ HybridRAGQuery singleton initialized successfully")
    
    def _setup_environment(self) -> None:
        """Load environment variables and validate required credentials."""
        print("🔧 Setting up environment...")
        load_dotenv()
        
        # OpenAI API Key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Neo4j Credentials validation
        self._validate_neo4j_credentials()
        print("✅ Environment setup complete")
    
    def _validate_neo4j_credentials(self) -> None:
        """Validate Neo4j connection credentials."""
        required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def _initialize_embedding_service(self) -> None:
        """Initialize embedding service for query processing."""
        print("🤖 Initializing embedding service...")
        try:
            self.embeddings = OpenAIEmbeddings()
            print("✅ Embedding service initialized")
        except Exception as e:
            print(f"❌ Failed to initialize embedding service: {e}")
            raise
    
    def _initialize_neo4j_connection(self) -> None:
        """Initialize Neo4j graph connection."""
        print("🔗 Initializing Neo4j connection...")
        
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            refresh_schema=False,
        )
        
        print("✅ Neo4j connection established")
    
    def list_all_vector_indexes(self) -> None:
        """List all available vector indexes in the Neo4j database."""
        print("\n📋 Available Vector Indexes:")
        print("-" * 40)
        
        try:
            # Query to get all vector indexes
            result = self.graph.query("""
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties, state
                WHERE type = 'VECTOR'
                RETURN name, labelsOrTypes, properties, state
                ORDER BY name
            """)
            
            if not result:
                print("⚠️  No vector indexes found in the database")
                return
            
            for idx, index_info in enumerate(result, 1):
                name = index_info.get('name', 'Unknown')
                labels = index_info.get('labelsOrTypes', [])
                properties = index_info.get('properties', [])
                state = index_info.get('state', 'Unknown')
                
                print(f"{idx}. Index: {name}")
                print(f"   Labels: {labels}")
                print(f"   Properties: {properties}")
                print(f"   State: {state}")
                print()
            
            print(f"📊 Total vector indexes found: {len(result)}")
            
        except Exception as e:
            print(f"⚠️  Error listing vector indexes: {e}")
    
    def get_available_labels_for_property(self, embedding_property: str) -> List[str]:
        """Get all node labels that have the specified embedding property."""
        try:
            result = self.graph.query(f"""
                MATCH (n)
                WHERE n.embedding_{embedding_property} IS NOT NULL
                RETURN DISTINCT labels(n) as node_labels
            """)
            
            labels = []
            for item in result:
                node_labels = item.get('node_labels', [])
                labels.extend(node_labels)
            
            # Remove duplicates and return
            return list(set(labels))
            
        except Exception as e:
            print(f"⚠️  Error getting labels for property embedding_{embedding_property}: {e}")
            return []
    
    def _search_by_index(
        self, 
        query_embedding: List[float], 
        embedding_property: str, 
        top_k: int,
        target_label: str = "Document"
    ) -> List[Dict[str, Any]]:
        """
        Search nodes using a specific embedding property and label.
        
        Args:
            query_embedding: The query vector embedding
            embedding_property: The embedding property to search against (e.g., 'text', 'hyp_queries')
            top_k: Maximum number of results to return
            target_label: The node label to search in (default: 'Document')
            
        Returns:
            List of matching nodes with similarity scores
        """
        try:
            # Construct the proper index name using both property and label
            index_name = f"embedding_{embedding_property}_{target_label.lower()}_index"
            
            # Check if vector index exists for this property and label combination
            if not self._check_vector_index_exists_with_label(embedding_property, target_label):
                print(f"⚠️  Vector index '{index_name}' does not exist, skipping")
                return []
            
            # Perform vector similarity search using Neo4j's vector index
            cypher_query = f"""
                CALL db.index.vector.queryNodes('{index_name}', $top_k, $query_embedding)
                YIELD node, score
                WHERE node:{target_label}
                RETURN 
                    elementId(node) as node_id,
                    node.text as text,
                    node.source as source,
                    node.chunk_index as chunk_index,
                    node.total_chunks as total_chunks,
                    score,
                    'embedding_{embedding_property}' as search_type,
                    labels(node) as labels,
                    keys(node) as properties
                ORDER BY score DESC
                LIMIT $top_k
            """
            
            result = self.graph.query(cypher_query, {
                "query_embedding": query_embedding,
                "top_k": top_k
            })
            
            print(f"📊 Found {len(result)} results using {index_name}")
            return result
            
        except Exception as e:
            print(f"⚠️  Error searching by {index_name}: {e}")
            return []
    
    def _check_vector_index_exists_with_label(self, embedding_property: str, target_label: str) -> bool:
        """
        Check if a vector index exists for the given embedding property and label combination.
        
        Args:
            embedding_property: The embedding property name (without 'embedding_' prefix)
            target_label: The node label to check for
            
        Returns:
            True if index exists, False otherwise
        """
        try:
            index_name = f"embedding_{embedding_property}_{target_label.lower()}_index"
            
            # Query to check if the index exists
            result = self.graph.query("""
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties
                WHERE name = $index_name AND type = 'VECTOR'
                RETURN count(*) as index_count
            """, {"index_name": index_name})
            
            exists = result[0]['index_count'] > 0 if result else False
            
            if not exists:
                print(f"ℹ️  Vector index '{index_name}' not found")
            
            return exists
            
        except Exception as e:
            print(f"⚠️  Error checking vector index existence: {e}")
            return False
    
    def search_similar_documents(self, user_query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for Document nodes similar to the user query using vector similarity.
        Uses the configured search properties and labels from VECTOR_SEARCH_CONFIGURATIONS.
        
        Args:
            user_query: The user's search query
            top_k: Maximum number of documents to return
            
        Returns:
            List of Document nodes with similarity scores and metadata
        """
        print(f"🔍 Searching for documents similar to: '{user_query}'")
        
        try:
            # Convert user query to embedding
            query_embedding = self.embeddings.embed_query(user_query)
            
            # Store the query embedding for potential use in graph traversal
            self.current_query_embedding = query_embedding
            
            # Search using all configured embedding properties and labels
            similar_docs = []
            
            for embedding_property, target_label in VECTOR_SEARCH_CONFIGURATIONS:
                print(f"🔍 Searching using embedding_{embedding_property} on {target_label} nodes...")
                
                search_results = self._search_by_index(
                    query_embedding, embedding_property, top_k, target_label
                )
                
                if search_results:
                    similar_docs.extend(search_results)
                    print(f"   ✅ Found {len(search_results)} results from embedding_{embedding_property}")
                else:
                    print(f"   ⚠️  No results from embedding_{embedding_property} on {target_label}")
            
            # Remove duplicates and sort by similarity score
            unique_docs = self._deduplicate_and_sort_results(similar_docs, top_k)
            
            print(f"✅ Found {len(unique_docs)} similar document(s) total")
            return unique_docs
            
        except Exception as e:
            print(f"⚠️  Error searching for similar documents: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _deduplicate_and_sort_results(
        self, 
        results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate documents and sort by similarity score.
        
        Args:
            results: List of search results from different embedding properties
            top_k: Maximum number of results to return
            
        Returns:
            Deduplicated and sorted list of document results
        """
        # Use node_id to deduplicate, keeping the result with highest score
        unique_results = {}
        
        for result in results:
            node_id = result.get('node_id')
            if not node_id:
                continue
            
            # Keep the result with the highest similarity score
            if node_id not in unique_results or result['score'] > unique_results[node_id]['score']:
                unique_results[node_id] = result
        
        # Sort by similarity score (highest first) and limit results
        sorted_results = sorted(
            unique_results.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )[:top_k]
        
        return sorted_results
    
    def get_document_context(self, user_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Main method to get relevant document context for a user query.
        This is the primary interface for the class.
        
        Args:
            user_query: The user's search query
            top_k: Maximum number of documents to return (default: 5)
            
        Returns:
            List of relevant Document nodes with metadata
        """
        try:
            similar_docs = self.search_similar_documents(user_query, top_k)
            
            if similar_docs:
                print(f"📝 Retrieved {len(similar_docs)} relevant document(s)")
                self._display_results_summary(similar_docs)
            else:
                print("⚠️  No relevant documents found")
            
            return similar_docs
            
        except Exception as e:
            print(f"❌ Context retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _display_results_summary(self, results: List[Dict[str, Any]]) -> None:
        """Display a summary of the search results."""
        if not results:
            return
        
        print(f"\n📊 Search Results Summary:")
        print("-" * 30)
        
        for i, result in enumerate(results[:5], 1):  # Show top 5 results
            source = result.get('source', 'Unknown')
            score = result.get('score', 0)
            search_type = result.get('search_type', 'unknown')
            chunk_info = ""
            
            if result.get('chunk_index') is not None and result.get('total_chunks') is not None:
                chunk_info = f" (chunk {result['chunk_index']+1}/{result['total_chunks']})"
            
            print(f"{i}. {source}{chunk_info}")
            print(f"   Similarity: {score:.4f} (via {search_type})")
            
            # Show a preview of the text
            text = result.get('text', '')
            if text:
                preview = text[:150] + "..." if len(text) > 150 else text
                print(f"   Preview: {preview}")
            print()

    def generate_final_answer(self, context: List[Dict[str, Any]], user_query: str) -> str:
        """
        Generate a final answer to the user's query based on the provided context.
        
        Args:
            context: List of relevant Document nodes with metadata
            user_query: The user's search query
            
        Returns:
            Formatted final answer to the user's query
        """
        print(f"🤖 Generating final answer based on {len(context)} context documents...")
        
        try:
            # Initialize ChatOpenAI
            chat = ChatOpenAI(
                model=model
            )
            
            # Format the context with all properties
            formatted_context = self._format_context_for_llm(context)
            
            # Create a comprehensive system prompt
            system_prompt = FINAL_ANSWER_SYSTEM_PROMPT
            
            # Create the user prompt with context and query
            user_prompt = f"""Context Information:
{formatted_context}

User Query: {user_query}

Please provide a comprehensive answer based on the context provided above."""
            
            # Generate response using LangChain
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = chat.invoke(messages)
            
            print("✅ Final answer generated successfully")
            return response.content
            
        except Exception as e:
            print(f"⚠️  Error generating final answer: {e}")
            import traceback
            traceback.print_exc()
            return "I'm sorry, but I couldn't generate a response to your query due to a technical error."
    
    def _format_context_for_llm(self, context: List[Dict[str, Any]]) -> str:
        """
        Format context documents for LLM consumption as JSON string.
        
        Args:
            context: List of context documents
            
        Returns:
            JSON stringified context
        """

        if not context:
            return "No context documents available."
        
        initial_docs = []
        added_docs = []
        
        for doc in context:
            labels = doc.get('labels', [])
            
            # Remove these keys from all documents
            doc.pop('node_id', None)
            doc.pop('score', None)
            doc.pop('search_type', None)
            doc.pop('discovered_from_node_id', None)
            doc.pop('hop_level', None)
            doc.pop('depth', None)
            doc.pop('total_cost', None)
            doc.pop('edge_cost', None)
            doc.pop('g_score', None)
            doc.pop('h_score', None)
            doc.pop('f_score', None)
            doc.pop('level', None)
            

            if labels == ["Document"]:
                # Keep only the 'text' key for Document nodes
                text_content = doc.get('text', '')
                doc.clear()
                doc['text'] = text_content
                initial_docs.append(doc)
            else:
                added_docs.append(doc)

        formatted = {
            "initial_documents": initial_docs,
            "additional_nodes_from_graph_traversal": added_docs
        }
        print('formatted', json.dumps(formatted, indent=2, default=str))
        return json.dumps(formatted, indent=2, ensure_ascii=False)


def _deduplicate_contexts(initial_docs: List[Dict[str, Any]], additional_docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Deduplicate documents from initial and additional contexts based on node_id.
    
    Args:
        initial_docs: Documents from initial vector search
        additional_docs: Documents from graph traversal
        top_k: Maximum number of results to return
        
    Returns:
        Combined list with duplicates removed, preserving initial docs when duplicates exist
    """
    # Use node_id as the key for deduplication
    unique_docs = {}
    
    # First add initial documents (they get priority)
    for doc in initial_docs:
        node_id = doc.get('node_id')
        if node_id:
            unique_docs[node_id] = doc
    
    # Then add additional documents (only if not already present)
    for doc in additional_docs:
        node_id = doc.get('node_id')
        if node_id and node_id not in unique_docs:
            unique_docs[node_id] = doc
    
    # Convert back to list, maintaining order (initial docs first, then additional)
    result = []
    
    # Add initial docs in original order
    for doc in initial_docs:
        node_id = doc.get('node_id')
        if node_id in unique_docs:
            result.append(unique_docs[node_id])
            del unique_docs[node_id]  # Remove to avoid adding again
    
    # Add remaining additional docs
    result.extend(unique_docs.values())
    
    # Sort by similarity score (highest first) and limit results
    sorted_results = sorted(
        result, 
        key=lambda x: x['score'], 
        reverse=True
    )[:top_k]
    
    return sorted_results


def main():
    """Example usage of the HybridRAGQuery singleton class."""
    print("🚀 Testing HybridRAGQuery Singleton")
    print("=" * 50)
    
    # First call - will initialize the singleton and all services
    print("Creating HybridRAGQuery instance:")
    context_retriever = HybridRAGQuery()
    # List all available vector indexes before the first query
    #context_retriever.list_all_vector_indexes()


    
    query_1 = "Construct a timeline (year-by-year) of every birth and death mentioned for Albert Einstein's immediate relatives – parents, siblings, spouse(s), children and grandchildren – then highlight which of those people were still alive on April 18 1955 (Einstein's death). For each person give the relationship to Albert Einstein."
    query_2 = "Two of Albert Einstein's children pursued medical studies but met very different ends. Identify them, specify where each studied, summarize the health condition that shaped each life, and state whether Einstein ever reunited with them after leaving Europe."
    query_3 = "Describe the full corporate journey of Einstein & Cie.: give its founding city and year, the founders' respective roles, the industries it served, the reason for its move to Italy, the city to which it moved, and the ultimate fate of the enterprise."
    query_4 = "List all six of Hermann Einstein's siblings in birth order, note the sex of each, and say which of them became the parent of Albert's future second wife."
    query = "Draw (or describe textually) a three-generation family graph starting with Abraham Einstein and Helene Moos at the top and ending with Bernhard Caesar Einstein's five children at the bottom. Include spouses where named"
    query_6 = "Which blood relatives of Albert Einstein ever lived with him at 112 Mercer Street in Princeton? For each, give the years of residence, their reason for staying, and the health status that influenced the arrangement."
    query_7 = "Identify every member of the Einstein family who emigrated to the United States before, during, or just after World War II, give the year of arrival, and state the push or pull factor that brought each person."
    query_8 = "Within Ludwigsvorstadt-Isarvorstadt, match each quarter (St. Paul, Ludwigsvorstadt/Kliniken, Am Schlachthof, Dreimühlenviertel, Am alten Südfriedhof, Glockenbachviertel, Gärtnerplatzviertel, Am Deutschen Museum) to the U-Bahn or S-Bahn stations and tram lines that serve it, noting any quarters that currently lack a subway stop."
    query_9 = "Rank the following Einstein relatives by the severity and type of personal tragedy they experienced (business bankruptcy, disabling illness, forced migration, early death, institutionalization): Hermann, Eduard, Lieserl, Hans Albert, Pauline. Justify the ranking with one-sentence explanations using information from the text."
    query_10 = "Trace Pauline Koch Einstein's residential moves chronologically from her marriage in 1876 to her death in 1920, identifying the city for each move, the primary family or economic reason, and any significant event in Albert's life that coincided with each relocation."   


    print("\n" + "🔄 PHASE 1: VECTOR SEARCH" + "\n" + "=" * 50)
    if ACTIVATE_INITIAL_VECTOR_SEARCH:
        print("🔍 Performing initial vector search...")
        initial_nodes = context_retriever.get_document_context(query, top_k=TOP_K_INITIAL)
    else:
        print("⏭️  Skipping initial vector search (ACTIVATE_INITIAL_VECTOR_SEARCH = False)")
        initial_nodes = []
    
    print("\n" + "🔄 PHASE 2: GRAPH TRAVERSAL ANALYSIS" + "\n" + "=" * 50)

    
    try:
        # Initialize graph traversal method based on configuration
        if GRAPH_TRAVERSAL_METHOD == "kop_limited_bfs":
            print("🔧 Initializing k-hop limited BFS traversal...")
            traversal_method = KhopLimitedBFS()
        elif GRAPH_TRAVERSAL_METHOD == "kop_limited_bfs_pred_llm":
            print("🔧 Initializing Predicate Constrained BFS traversal...")
            traversal_method = KhopLimitedBFSWithLLM()
        elif GRAPH_TRAVERSAL_METHOD == "depth_limited_dfs":
            print("🔧 Initializing Depth Limited DFS traversal...")
            traversal_method = DepthLimitedDFS()
        elif GRAPH_TRAVERSAL_METHOD == "depth_limited_dfs_pred_llm":
            print("🔧 Initializing Predicate Constrained DFS traversal...")
            traversal_method = DepthLimitedDFSWithLLM()
        elif GRAPH_TRAVERSAL_METHOD == "uniform_cost_search_ucs":
            print("🔧 Initializing Uniform Cost Search UCS traversal...")
            traversal_method = UniformCostSearchUCS()
        elif GRAPH_TRAVERSAL_METHOD == "uniform_cost_search_ucs_pred_llm":
            print("🔧 Initializing Uniform Cost Search UCS with Predicate LLM traversal...")
            traversal_method = UniformCostSearchUCSWithLLM()
        elif GRAPH_TRAVERSAL_METHOD == "astar_search_heuristic":
            print("🔧 Initializing A* Search with Heuristic traversal...")
            query_embedding = getattr(context_retriever, 'current_query_embedding', None)
            traversal_method = AStarSearchHeuristic(query_embedding=query_embedding)
        elif GRAPH_TRAVERSAL_METHOD == "astar_search_heuristic_pred_llm":
            print("🔧 Initializing A* Search with Heuristic Predicate LLM traversal...")
            query_embedding = getattr(context_retriever, 'current_query_embedding', None)
            traversal_method = AStarSearchHeuristicWithLLM(query_embedding=query_embedding)
        elif GRAPH_TRAVERSAL_METHOD == "beam_search_over_the_graph":
            print("🔧 Initializing Beam Search Over Graph traversal...")
            traversal_method = BeamSearchOverGraph()
        elif GRAPH_TRAVERSAL_METHOD == "beam_search_over_the_graph_pred_llm":
            print("🔧 Initializing Beam Search Over Graph with Predicate LLM traversal...")
            traversal_method = BeamSearchOverGraphWithLLM()
        else:
            print("🔧 Initializing ContextToCypher...")
            traversal_method = ContextToCypher()
        
        # Get additional context through graph traversal
        additional_context = traversal_method.traverse_graph(initial_nodes, query)
        #print('additional_context', json.dumps(additional_context, indent=2, default=str))

        # Combine initial and additional context with deduplication and apply TOP_K limit
        all_context = _deduplicate_contexts(initial_nodes, additional_context, TOP_K_TRAVERSAL)
       # print('all_context', json.dumps(all_context, indent=2, default=str))
        
        print(f"\n📊 FINAL CONTEXT SUMMARY:")
        print(f"Initial documents: {len(initial_nodes)}")
        print(f"Additional documents from graph traversal: {len(additional_context)}")
        print(f"Total context documents (after deduplication and TOP_K limit): {len(all_context)}")
        
    except Exception as e:
        print(f"⚠️  Error during graph traversal: {e}")
        print("Continuing with initial context only...")
        all_context = initial_nodes[:TOP_K_TRAVERSAL]  # Apply TOP_K limit even for initial context only
    
    print("\n" + "🔄 PHASE 3: ANSWER GENERATION" + "\n" + "=" * 50)
    
    # Generate final answer using all collected context
    if all_context:
        final_answer = context_retriever.generate_final_answer(all_context, query)
        print("\n" + "🎯 FINAL ANSWER" + "\n" + "=" * 50)
        print(final_answer)
    else:
        print("⚠️  No context available to generate an answer.")
        print("Please check your vector search configuration and graph traversal setup.")
    
    print("\n" + "=" * 50)
    print("🏁 Query processing completed!")


if __name__ == "__main__":
    main()
