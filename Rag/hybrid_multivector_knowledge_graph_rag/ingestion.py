# main.py

import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from getpass import getpass
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
import params


class KnowledgeGraphIngestion:
    """
    A comprehensive class for ingesting documents into a Neo4j knowledge graph
    with optional vector embeddings and indexes.
    """
    
    def __init__(self):
        """Initialize the ingestion system with all services."""
        # Core services
        self.graph: Optional[Neo4jGraph] = None
        self.llm: Optional[ChatOpenAI] = None
        self.transformer: Optional[LLMGraphTransformer] = None

        # Embedding services
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.chunker_embeddings: Optional[OpenAIEmbeddings] = None
        self.semantic_chunker: Optional[SemanticChunker] = None
        
        # Data storage
        self.documents: List[Document] = []
        
        # Track newly ingested data for this session
        self.current_session_sources: set = set()
        self.current_session_nodes: List[str] = []
        self.current_session_relationships: List[str] = []
    
    def _initialize_embedding_services(self) -> None:
        """Initialize embedding services."""
        # Initialize embeddings for general use (vector indexing)
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize separate embeddings instance for chunking to avoid conflicts
        self.chunker_embeddings = OpenAIEmbeddings()
        
        # Initialize semantic chunker with configurable parameters
        self.semantic_chunker = SemanticChunker(
            embeddings=self.chunker_embeddings,                         # Embedding model to compute semantic similarity
            breakpoint_threshold_type=params.semantic_chunker_breakpoint_type,     # Method for determining breakpoints
            breakpoint_threshold_amount=params.semantic_chunker_breakpoint_threshold,   # Threshold value for breakpoints
            min_chunk_size=params.semantic_chunker_min_chunk_size       # Minimum size for any chunk
        )
    
    def _initialize_llm(self) -> None:
        """Initialize LLM service."""
        self.llm = ChatOpenAI(model_name=params.model)
    
    def _ensure_services_initialized(self) -> None:
        """Ensure all services are properly initialized, reinitialize if needed."""
        if self.embeddings is None or self.chunker_embeddings is None or self.semantic_chunker is None:
            print("🔧 Initializing embedding services...")
            self._initialize_embedding_services()
        
        if self.llm is None:
            print("🔧 Initializing LLM...")
            self._initialize_llm()
    
    def setup_environment(self) -> None:
        """Load environment variables and validate required credentials."""
        print("🔧 Setting up environment...")
        load_dotenv()
        
        # OpenAI API Key
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")
        
        # Neo4j Credentials validation
        self._validate_neo4j_credentials()
        print("✅ Environment setup complete")
    
    def _validate_neo4j_credentials(self) -> None:
        """Validate Neo4j connection credentials."""
        required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def initialize_neo4j_connection(self) -> None:
        """Initialize Neo4j graph connection."""
        print("🔗 Initializing Neo4j connection...")
        
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            refresh_schema=False,
        )
        
        # Check Neo4j version for compatibility
        self._check_neo4j_version()
        
        print("✅ Neo4j connection established")
    
    def _check_neo4j_version(self) -> None:
        """Check Neo4j version and log compatibility information."""
        try:
            version_result = self.graph.query("CALL dbms.components() YIELD name, versions, edition WHERE name = 'Neo4j Kernel' RETURN versions[0] as version")
            if version_result:
                version = version_result[0]['version']
                print(f"ℹ️  Neo4j version: {version}")
                
                # Parse version to determine vector index syntax
                major_version = int(version.split('.')[0])
                if major_version >= 5:
                    print("ℹ️  Using Neo4j 5+ vector index syntax")
                    # Test if relationship vector indexes are supported
                    self._test_relationship_vector_support()
                else:
                    print("ℹ️  Using legacy Neo4j vector index syntax")
                    print("⚠️  Relationship vector indexes may not be supported in this version")
            else:
                print("ℹ️  Could not determine Neo4j version")
        except Exception as e:
            print(f"ℹ️  Could not check Neo4j version: {e}")
    
    def _test_relationship_vector_support(self) -> None:
        """Test if relationship vector indexes are supported in this Neo4j instance."""
        try:
            # Try to create a test relationship vector index with configurable parameters
            test_result = self.graph.query(f"""
                CREATE VECTOR INDEX test_rel_vector_index
                IF NOT EXISTS
                FOR ()-[r:TEST_REL]-() ON (r.test_embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {params.vector_embedding_dimensions},
                        `vector.similarity_function`: '{params.vector_similarity_function}'
                    }}
                }}
            """)
            
            # If successful, drop the test index
            self.graph.query("DROP INDEX test_rel_vector_index IF EXISTS")
            print("✅ Relationship vector indexes are supported")
            
        except Exception as e:
            print(f"⚠️  Relationship vector indexes may not be supported: {e}")
            print("ℹ️  This could be due to Neo4j version, edition, or configuration")
    
    def setup_llm_and_transformer(self) -> None:
        """Setup LLM and configure the graph transformer with parameters."""
        print("🤖 Setting up LLM and graph transformer...")
        
        # Ensure all services are initialized
        self._ensure_services_initialized()
        
        # Build transformer with pre-initialized LLM
        transformer_params = self._build_transformer_parameters()
        self.transformer = LLMGraphTransformer(**transformer_params)
        
        print("✅ LLM and transformer setup complete")
    
    def _build_transformer_parameters(self) -> Dict[str, Any]:
        """Build transformer parameters based on configuration."""
        transformer_params = {"llm": self.llm}
        
        if params.use_custom_prompt and params.custom_prompt:
            # Use custom prompt configuration
            custom_chat_prompt = ChatPromptTemplate.from_messages([
                ("system", params.custom_prompt),
                ("human", "{input}")
            ])
            transformer_params["prompt"] = custom_chat_prompt
        else:
            # Use schema constraints
            self._add_schema_constraints(transformer_params)
        
        # Add property configurations
        self._add_property_configurations(transformer_params)
        
        # Add additional instructions
        if params.use_additional_instructions and params.additional_instructions:
            transformer_params["additional_instructions"] = params.additional_instructions
        
        return transformer_params
    
    def _add_schema_constraints(self, transformer_params: Dict[str, Any]) -> None:
        """Add schema constraints to transformer parameters."""
        if params.use_allowed_nodes:
            transformer_params["allowed_nodes"] = params.allowed_nodes
            
        if params.use_allowed_relationships:
            transformer_params["allowed_relationships"] = (
                params.allowed_relationships_tuples 
                if params.use_relationships_tuples 
                else params.allowed_relationships
            )
    
    def _add_property_configurations(self, transformer_params: Dict[str, Any]) -> None:
        """Add property configurations to transformer parameters."""
        if params.node_properties:
            transformer_params["node_properties"] = params.node_properties
            
        if params.relationship_properties:
            transformer_params["relationship_properties"] = params.relationship_properties
    
    def load_documents(self) -> None:
        """Load documents from configured source."""
        print("📄 Loading documents...")
        self._load_from_docs_folder()
        
        if not self.documents:
            raise ValueError("No documents found. Please check your data source.")
        
        print(f"✅ Loaded {len(self.documents)} document(s)")
    
    
    def _chunk_document(self, content: str, source_path: str, text_splitter: SemanticChunker) -> List[Document]:
        """
        Chunk a document's content into semantic chunks and return Document objects.
        
        Args:
            content: The document content to chunk
            source_path: Path to the source document
            text_splitter: SemanticChunker instance to use for chunking
            
        Returns:
            List of Document objects representing the chunks
        """
        chunks = text_splitter.split_text(content)
        chunked_documents = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "source": source_path,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            chunked_documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        return chunked_documents
    
    def _load_from_docs_folder(self) -> None:
        """Load documents from docs/ folder and apply semantic chunking."""
        self.documents = []
        
        # Ensure semantic chunker is initialized
        if self.semantic_chunker is None:
            print("🔧 Semantic chunker not initialized, reinitializing...")
            self._initialize_embedding_services()
        
        if self.semantic_chunker is None:
            raise ValueError("Failed to initialize semantic chunker. Please check your OpenAI API key.")
        
        text_splitter = self.semantic_chunker
        
        # Load documents from configurable source path
        for path in glob.glob(params.documents_source_path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Chunk the document and add all chunks to documents list
                chunked_docs = self._chunk_document(content, path, text_splitter)
                self.documents.extend(chunked_docs)
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not load {path}: {e}")
    
    def extract_graph_documents(self) -> List[Any]:
        """Extract graph structures from documents."""
        print("🔍 Extracting graph structures...")
        
        graph_docs = self.transformer.convert_to_graph_documents(self.documents)
        
        # Display first document's extracted elements
        if graph_docs:
            print(f"Nodes: {graph_docs[0].nodes}")
            print(f"Relationships: {graph_docs[0].relationships}")
        
        print(f"✅ Extracted graph structures from {len(graph_docs)} document(s)")
        return graph_docs
    
    def ingest_to_neo4j(self, graph_docs: List[Any]) -> None:
        """Ingest graph documents into Neo4j and track newly created nodes and relationships."""
        print("💾 Ingesting data into Neo4j...")
        
        # Track sources being processed in this session
        for doc in self.documents:
            source = doc.metadata.get('source', 'unknown')
            self.current_session_sources.add(source)
        
        print(f"📄 Processing sources: {sorted(self.current_session_sources)}")
        
        # Get existing nodes and relationships count before ingestion
        existing_nodes_count = self.graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        existing_rels_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        
        # Ingest the graph documents
        self.graph.add_graph_documents(
            graph_docs,
            baseEntityLabel=params.baseEntityLabel,
            include_source=params.include_source,
        )
        
        # Get newly created nodes and relationships for this session
        self._collect_newly_created_elements()
        
        print(f"✅ Successfully ingested {len(graph_docs)} document(s) into Neo4j")
        print(f"📊 New nodes: {len(self.current_session_nodes)}, New relationships: {len(self.current_session_relationships)}")
    
    def _collect_newly_created_elements(self) -> None:
        """Collect IDs of newly created nodes and relationships from current session sources."""
        try:
            # Get nodes created from current session sources (Document nodes)
            if self.current_session_sources:
                sources_list = list(self.current_session_sources)
                
                # Get Document nodes related to current session sources
                document_nodes_result = self.graph.query("""
                    MATCH (n)
                    WHERE any(source IN $sources WHERE n.source CONTAINS source OR n.source = source)
                    RETURN elementId(n) as node_id
                """, {"sources": sources_list})
                
                document_node_ids = [item['node_id'] for item in document_nodes_result]
                
                # Get ALL nodes that are connected to these Document nodes (Person, Concept, etc.)
                if document_node_ids:
                    # Get all nodes connected to Document nodes through MENTIONS relationships
                    all_connected_nodes_result = self.graph.query("""
                        MATCH (doc)-[:MENTIONS]->(entity)
                        WHERE elementId(doc) IN $document_node_ids
                        RETURN DISTINCT elementId(entity) as node_id
                    """, {"document_node_ids": document_node_ids})
                    
                    connected_node_ids = [item['node_id'] for item in all_connected_nodes_result]
                    
                    # Combine Document nodes and connected entity nodes
                    self.current_session_nodes = document_node_ids + connected_node_ids
                else:
                    self.current_session_nodes = document_node_ids
                
                # Get relationships connected to current session nodes
                if self.current_session_nodes:
                    rels_result = self.graph.query("""
                        MATCH (n1)-[r]->(n2)
                        WHERE elementId(n1) IN $node_ids OR elementId(n2) IN $node_ids
                        RETURN elementId(r) as rel_id
                    """, {"node_ids": self.current_session_nodes})
                    
                    self.current_session_relationships = [item['rel_id'] for item in rels_result]
                
                print(f"🔍 Identified {len(self.current_session_nodes)} nodes and {len(self.current_session_relationships)} relationships for current session")
            
        except Exception as e:
            print(f"⚠️  Error collecting newly created elements: {e}")
            # Fallback: use empty lists to avoid processing entire database
            self.current_session_nodes = []
            self.current_session_relationships = []
    
    def create_vector_embeddings(self) -> None:
        """Create vector embeddings for graph properties if enabled."""
        if not params.add_vector_index:
            print("ℹ️  Vector embeddings disabled in configuration")
            return
        
        print("🔍 Creating vector embeddings...")
        
        try:
            # Ensure all services are initialized
            self._ensure_services_initialized()
            
            # Process nodes and relationships in parallel
            self._parallel_process_embeddings()
            
            # Create vector indexes in parallel
            self._parallel_create_vector_indexes()
            
            # Verify embeddings were created
            self._verify_embeddings()
            
            print("✅ Vector embeddings creation complete")
            
        except Exception as e:
            print(f"⚠️  Error creating vector embeddings: {e}")
            import traceback
            traceback.print_exc()
    
    def _parallel_process_embeddings(self) -> None:
        """Process node and relationship embeddings in parallel."""
        
        def process_node_embeddings():
            """Process embeddings for node properties."""
            print("📋 Processing node embeddings...")
            
            nodes_data = self._get_nodes_data()
            if not nodes_data:
                print("⚠️  No nodes found for embedding")
                return []
            
            properties_to_embed = self._collect_node_properties_to_embed(nodes_data)
            
            if properties_to_embed:
                self._parallel_create_embeddings_batch(properties_to_embed, "node")
                print(f"✅ Created embeddings for {len(properties_to_embed)} node properties")
                return properties_to_embed
            else:
                print("⚠️  No suitable node properties found for embedding")
                return []
        
        def process_relationship_embeddings():
            """Process embeddings for relationship properties."""
            print("📋 Processing relationship embeddings...")
            
            relationships_data = self._get_relationships_data()
            if not relationships_data:
                print("⚠️  No relationships found for embedding")
                return []
            
            properties_to_embed = self._collect_relationship_properties_to_embed(relationships_data)
            
            if properties_to_embed:
                self._parallel_create_embeddings_batch(properties_to_embed, "relationship")
                print(f"✅ Created embeddings for {len(properties_to_embed)} relationship properties")
                return properties_to_embed
            else:
                print("⚠️  No suitable relationship properties found for embedding")
                return []
        
        # Process nodes and relationships in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            node_future = executor.submit(process_node_embeddings)
            rel_future = executor.submit(process_relationship_embeddings)
            
            # Wait for both to complete
            node_properties = node_future.result()
            rel_properties = rel_future.result()
    
    def _parallel_create_vector_indexes(self) -> None:
        """Create vector indexes in parallel."""
        print("🔍 Creating vector indexes...")
        
        properties_to_index = self._get_properties_to_index()
        
        if not properties_to_index:
            print("⚠️  No properties specified for indexing")
            return
        
        print(f"Creating indexes for properties: {sorted(list(properties_to_index))}")
        
        def create_property_indexes_parallel(prop: str) -> Tuple[str, int]:
            """Create vector indexes for a specific property in parallel."""
            return prop, self._create_property_indexes(prop)
        
        # Create indexes for all properties in parallel
        max_workers = min(4, len(properties_to_index))
        total_created = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prop = {
                executor.submit(create_property_indexes_parallel, prop): prop
                for prop in properties_to_index
            }
            
            for future in as_completed(future_to_prop):
                prop, created = future.result()
                if created:
                    total_created += 1
        
        print(f"✅ Successfully created vector indexes for {total_created} properties")
    
    def _get_nodes_data(self) -> List[Dict[str, Any]]:
        """Retrieve nodes data from Neo4j for current session only."""
        if not self.current_session_nodes:
            print("ℹ️  No current session nodes to process")
            return []
        
        print(f"📋 Getting data for {len(self.current_session_nodes)} current session nodes")
        return self.graph.query("""
            MATCH (n) 
            WHERE elementId(n) IN $node_ids
            RETURN labels(n) as node_labels, keys(n) as properties, elementId(n) as node_id
        """, {"node_ids": self.current_session_nodes})
    
    def _get_relationships_data(self) -> List[Dict[str, Any]]:
        """Retrieve relationships data from Neo4j for current session only."""
        if not self.current_session_relationships:
            print("⚠️  No current session relationships found - falling back to all relationships")
            print("🔍 Getting all relationships from database...")
            
            # Fallback: get all non-MENTIONS relationships
            result = self.graph.query("""
                MATCH ()-[r]->() 
                RETURN type(r) as rel_type, keys(r) as properties, elementId(r) as rel_id
                LIMIT 1000
            """)
            
            return result
        
        print(f"📋 Getting data for {len(self.current_session_relationships)} current session relationships (excluding MENTIONS)")
        result = self.graph.query("""
            MATCH ()-[r]->() 
            WHERE elementId(r) IN $rel_ids AND type(r) <> 'MENTIONS'
            RETURN type(r) as rel_type, keys(r) as properties, elementId(r) as rel_id
        """, {"rel_ids": self.current_session_relationships})
        
        return result
    
    def _collect_node_properties_to_embed(self, nodes_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect node properties that should be embedded."""
        properties_to_embed = []
        
        for node_info in nodes_data:
            if not self._should_process_node(node_info):
                continue
            
            node_data = self._get_node_data(node_info['node_id'])
            if not node_data:
                continue
            
            properties_to_embed.extend(
                self._extract_node_property_embeddings(node_info, node_data)
            )
        
        return properties_to_embed
    
    def _collect_relationship_properties_to_embed(self, relationships_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect relationship properties that should be embedded."""
        properties_to_embed = []
        
        for rel_info in relationships_data:
            if not self._should_process_relationship(rel_info):
                continue
            
            # Get relationship data from Neo4j
            rel_data = self._get_relationship_data(rel_info['rel_id'])
            if not rel_data:
                continue
            
            # Extract properties for embedding
            extracted_props = self._extract_relationship_property_embeddings(rel_info, rel_data)
            if extracted_props:
                properties_to_embed.extend(extracted_props)
        
        return properties_to_embed
    
    def _should_process_node(self, node_info: Dict[str, Any]) -> bool:
        """Check if node should be processed based on filters."""
        if not params.filter_node_labels_to_index:
            return True
        
        # Check for "ALL" parameter
        if "ALL" in params.filter_node_labels_to_index:
            return True
        
        node_labels = node_info.get('node_labels', [])
        return any(label in params.filter_node_labels_to_index for label in node_labels)
    
    def _should_process_relationship(self, rel_info: Dict[str, Any]) -> bool:
        """Check if relationship should be processed based on filters."""
        rel_type = rel_info.get('rel_type', '')
        
        if not params.filter_rels_labels_to_index:
            return True
        
        # Check for "ALL" parameter
        if "ALL" in params.filter_rels_labels_to_index:
            return True
        
        result = rel_type in params.filter_rels_labels_to_index
        return result
    
    def _get_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data by ID."""
        result = self.graph.query("""
            MATCH (n) 
            WHERE elementId(n) = $node_id 
            RETURN n
        """, {"node_id": node_id})
        
        return result[0]['n'] if result else None
    
    def _get_relationship_data(self, rel_id: str) -> Optional[Dict[str, Any]]:
        """Get relationship data by ID."""
        result = self.graph.query("""
            MATCH ()-[r]->() 
            WHERE elementId(r) = $rel_id 
            RETURN r, properties(r) as all_props
        """, {"rel_id": rel_id})
        
        return result[0] if result else None
    
    def _extract_node_property_embeddings(self, node_info: Dict[str, Any], node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract embeddings for node properties."""
        properties_to_embed = []
        node_labels = node_info.get('node_labels', [])
        properties = node_info.get('properties', [])
        
        for prop in properties:
            if not self._should_embed_node_property(prop):
                continue
            
            if prop in node_data and node_data[prop] is not None:
                value = str(node_data[prop])
                if self._is_valid_embedding_value(value, prop):
                    context_text = f"Node type: {', '.join(node_labels) if node_labels else 'Unknown'} | Property {prop}: {value}"
                    
                    properties_to_embed.append({
                        'node_id': node_info['node_id'],
                        'property_name': prop,
                        'text': context_text,
                        'embedding_name': f'embedding_{prop}',
                        'type': 'node'
                    })
        
        return properties_to_embed
    
    def _extract_relationship_property_embeddings(self, rel_info: Dict[str, Any], rel_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relationship properties that should be embedded."""
        properties_to_embed = []
        rel_type = rel_info.get('rel_type', 'Unknown')
        properties = rel_info.get('properties', [])
        all_props = rel_data.get('all_props', {})
        
        for prop in properties:
            if not self._should_embed_relationship_property(prop):
                continue
            
            if prop not in all_props:
                continue
                
            if all_props[prop] is None:
                continue
            
            value = str(all_props[prop])
            
            if self._is_valid_embedding_value(value, prop):
                context_text = f"Relationship type: {rel_type} | Property {prop}: {value}"
                
                properties_to_embed.append({
                    'rel_id': rel_info['rel_id'],
                    'property_name': prop,
                    'text': context_text,
                    'embedding_name': f'embedding_{prop}',
                    'type': 'relationship'
                })
        
        return properties_to_embed
    
    def _should_embed_node_property(self, prop: str) -> bool:
        """Check if node property should be embedded."""
        if not params.filter_node_properties_to_index:
            return True
        
        # Check for "ALL" parameter
        if "ALL" in params.filter_node_properties_to_index:
            return True
        
        return prop in params.filter_node_properties_to_index
    
    def _should_embed_relationship_property(self, prop: str) -> bool:
        """Check if relationship property should be embedded."""
        if not params.filter_rels_properties_to_index:
            return True
        
        # Check for "ALL" parameter
        if "ALL" in params.filter_rels_properties_to_index:
            return True
        
        result = prop in params.filter_rels_properties_to_index
        return result
    
    def _should_embed_document_property(self, prop: str) -> bool:
        """Check if document property should be embedded based on node property filters."""
        if not params.filter_node_properties_to_index:
            return True
        
        # Check for "ALL" parameter
        if "ALL" in params.filter_node_properties_to_index:
            return True
        
        return prop in params.filter_node_properties_to_index
    
    def _is_valid_embedding_value(self, value: str, prop: str) -> bool:
        """Check if value is suitable for embedding."""
        return (len(value) < params.max_embedding_text_length and 
                not value.startswith(params.embedding_exclusion_prefix) and 
                not prop.startswith('embedding'))
    
    def _create_embeddings_batch(self, properties_to_embed: List[Dict[str, Any]], embed_type: str) -> None:
        """Create embeddings in batches with parallel processing."""
        if not properties_to_embed:
            return
            
        # Use the parallel version for better performance
        self._parallel_create_embeddings_batch(properties_to_embed, embed_type)
    
    def _parallel_create_embeddings_batch(self, properties_to_embed: List[Dict[str, Any]], embed_type: str) -> None:
        """Create embeddings in parallel batches."""
        if not properties_to_embed:
            return
        
        def create_embedding_batch(batch_info: Tuple[int, List[Dict]]) -> Tuple[int, List[List[float]]]:
            """Create embeddings for a single batch."""
            batch_idx, batch = batch_info
            texts = [item['text'] for item in batch]
            embeddings = self.embeddings.embed_documents(texts)
            return batch_idx, embeddings
        
        # Split into batches
        batch_size = params.embedding_batch_size
        batches = [
            properties_to_embed[i:i+batch_size] 
            for i in range(0, len(properties_to_embed), batch_size)
        ]
        
        total_batches = len(batches)
        print(f"🔍 Processing {total_batches} embedding batches in parallel...")
        
        # Process batches in parallel
        max_workers = min(4, total_batches, os.cpu_count() or 2)  # Limit to avoid API rate limits
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch tasks
            batch_tasks = [(i, batch) for i, batch in enumerate(batches)]
            future_to_batch = {
                executor.submit(create_embedding_batch, batch_info): batch_info
                for batch_info in batch_tasks
            }
            
            # Collect results maintaining order
            batch_results = [None] * total_batches
            completed = 0
            
            for future in as_completed(future_to_batch):
                try:
                    batch_idx, embeddings = future.result()
                    batch_results[batch_idx] = embeddings
                    completed += 1
                    print(f"{embed_type.title()} properties batch {completed}/{total_batches} - Embedded {len(embeddings)} properties")
                    
                except Exception as batch_error:
                    batch_info = future_to_batch[future]
                    print(f"⚠️  Error processing {embed_type} properties batch {batch_info[0] + 1}: {batch_error}")
                    batch_results[batch_info[0]] = []
        
        # Store embeddings in order
        for batch_idx, (batch, embeddings) in enumerate(zip(batches, batch_results)):
            if embeddings:
                for item, embedding in zip(batch, embeddings):
                    self._store_embedding(item, embedding, embed_type)
    
    def _store_embedding(self, item: Dict[str, Any], embedding: List[float], embed_type: str) -> None:
        """Store embedding in Neo4j."""
        if embed_type == 'node' or embed_type == 'document_node':
            self.graph.query(f"""
                MATCH (n) 
                WHERE elementId(n) = $node_id 
                SET n.{item['embedding_name']} = $embedding
            """, {
                "node_id": item['node_id'], 
                "embedding": embedding
            })
        else:  # relationship
            self.graph.query(f"""
                MATCH ()-[r]->() 
                WHERE elementId(r) = $rel_id 
                SET r.{item['embedding_name']} = $embedding
            """, {
                "rel_id": item['rel_id'], 
                "embedding": embedding
            })
    
    def _get_properties_to_index(self) -> set:
        """Get set of properties that should be indexed."""
        properties_to_index = set()
        
        if params.filter_node_properties_to_index:
            properties_to_index.update(params.filter_node_properties_to_index)
        
        if params.filter_rels_properties_to_index:
            properties_to_index.update(params.filter_rels_properties_to_index)
        
        return properties_to_index
    
    def _create_property_indexes(self, prop: str) -> bool:
        """Create vector indexes for a specific property on current session nodes and relationships only."""
        if not self.current_session_nodes and not self.current_session_relationships:
            print(f"ℹ️  No current session data to index for property {prop}")
            return False
        
        success_count = 0
        
        try:
            # Get unique node labels that have this embedding property in current session
            if self.current_session_nodes:
                node_labels_result = self.graph.query(f"""
                    MATCH (n) 
                    WHERE elementId(n) IN $node_ids AND n.embedding_{prop} IS NOT NULL 
                    RETURN DISTINCT labels(n) as node_labels
                """, {"node_ids": self.current_session_nodes})
                
                # Create indexes for each node label that has this property
                for label_info in node_labels_result:
                    node_labels = label_info['node_labels']
                    for label in node_labels:
                        try:
                            index_name = f"embedding_{prop}_{label.lower()}_index"
                            self.graph.query(f"""
                                CREATE VECTOR INDEX {index_name}
                                IF NOT EXISTS
                                FOR (n:{label}) ON (n.embedding_{prop})
                                OPTIONS {{
                                    indexConfig: {{
                                        `vector.dimensions`: {params.vector_embedding_dimensions},
                                        `vector.similarity_function`: '{params.vector_similarity_function}'
                                    }}
                                }}
                            """)
                            print(f"✅ Created node vector index for {label}.embedding_{prop}")
                            success_count += 1
                            
                        except Exception as node_error:
                            print(f"⚠️  Could not create node index for {label}.embedding_{prop}: {node_error}")
            
            # Get unique relationship types that have this embedding property in current session
            if self.current_session_relationships:
                rel_types_result = self.graph.query(f"""
                    MATCH ()-[r]->() 
                    WHERE elementId(r) IN $rel_ids AND r.embedding_{prop} IS NOT NULL 
                    RETURN DISTINCT type(r) as rel_type
                """, {"rel_ids": self.current_session_relationships})
                
                # Create indexes for each relationship type that has this property
                for rel_info in rel_types_result:
                    rel_type = rel_info['rel_type']
                    try:
                        index_name = f"embedding_{prop}_{rel_type.lower()}_rel_index"
                        
                        # Use configurable vector parameters for relationship indexes
                        self.graph.query(f"""
                            CREATE VECTOR INDEX {index_name}
                            IF NOT EXISTS
                            FOR ()-[r:{rel_type}]-() ON (r.embedding_{prop})
                            OPTIONS {{
                                indexConfig: {{
                                    `vector.dimensions`: {params.vector_embedding_dimensions},
                                    `vector.similarity_function`: '{params.vector_similarity_function}'
                                }}
                            }}
                        """)
                        
                        print(f"✅ Created relationship vector index for {rel_type}.embedding_{prop}")
                        success_count += 1
                        
                    except Exception as rel_error:
                        print(f"⚠️  Could not create relationship index for {rel_type}.embedding_{prop}: {rel_error}")
                        
                        # Let's also check what Neo4j version we're dealing with
                        print(f"ℹ️  This might be due to Neo4j version compatibility.")
                        print(f"ℹ️  Relationship vector indexes require Neo4j 5.0+ with specific configurations.")
            
            return success_count > 0
            
        except Exception as e:
            print(f"⚠️  Error creating indexes for embedding_{prop}: {e}")
            return False
    
    def _verify_embeddings(self) -> None:
        """Verify that embeddings were successfully created for current session only."""
        print("\n📊 Verifying embeddings for current session...")
        
        if not self.current_session_nodes and not self.current_session_relationships:
            print("ℹ️  No current session data to verify")
            return
        
        try:
            # Check node embeddings for current session
            properties_to_check = self._get_properties_to_index()
            
            for prop in sorted(properties_to_check):
                node_count = 0
                rel_count = 0
                
                # Count current session nodes with this embedding
                if self.current_session_nodes:
                    node_count_result = self.graph.query(f"""
                        MATCH (n) 
                        WHERE elementId(n) IN $node_ids AND n.embedding_{prop} IS NOT NULL 
                        RETURN count(n) as count
                    """, {"node_ids": self.current_session_nodes})
                    
                    node_count = node_count_result[0]['count'] if node_count_result else 0
                
                # Count current session relationships with this embedding
                if self.current_session_relationships:
                    rel_count_result = self.graph.query(f"""
                        MATCH ()-[r]->() 
                        WHERE elementId(r) IN $rel_ids AND r.embedding_{prop} IS NOT NULL 
                        RETURN count(r) as count
                    """, {"rel_ids": self.current_session_relationships})
                    
                    rel_count = rel_count_result[0]['count'] if rel_count_result else 0
                
                if node_count > 0 or rel_count > 0:
                    print(f"✅ {prop}: {node_count} nodes, {rel_count} relationships")
                else:
                    print(f"⚠️  {prop}: No embeddings found in current session")
            
            # Overall summary for current session
            total_node_embeddings = 0
            total_rel_embeddings = 0
            
            for prop in properties_to_check:
                if self.current_session_nodes:
                    node_result = self.graph.query(f"""
                        MATCH (n) 
                        WHERE elementId(n) IN $node_ids AND n.embedding_{prop} IS NOT NULL 
                        RETURN count(n) as count
                    """, {"node_ids": self.current_session_nodes})
                    total_node_embeddings += node_result[0]['count'] if node_result else 0
                
                if self.current_session_relationships:
                    rel_result = self.graph.query(f"""
                        MATCH ()-[r]->() 
                        WHERE elementId(r) IN $rel_ids AND r.embedding_{prop} IS NOT NULL 
                        RETURN count(r) as count
                    """, {"rel_ids": self.current_session_relationships})
                    total_rel_embeddings += rel_result[0]['count'] if rel_result else 0
            
            print(f"\n🎉 Total embeddings created for current session: {total_node_embeddings + total_rel_embeddings}")
            print(f"   - Node embeddings: {total_node_embeddings}")
            print(f"   - Relationship embeddings: {total_rel_embeddings}")
            
        except Exception as e:
            print(f"⚠️  Error verifying embeddings: {e}")
    
    def add_multivectors_to_document_nodes(self) -> None:
        """Add custom properties to Document nodes from current session only."""
        if not hasattr(params, 'document_multi_vector_properties') or not params.document_multi_vector_properties:
            print("ℹ️  No document node properties to add")
            return
        
        if not self.current_session_sources:
            print("ℹ️  No current session sources to process")
            return
        
        print("🔧 Adding custom properties to Document nodes from current session...")
        
        try:
            # Get Document nodes from current session sources only
            sources_list = list(self.current_session_sources)
            document_nodes = self.graph.query("""
                MATCH (d:Document) 
                WHERE any(source IN $sources WHERE d.source CONTAINS source OR d.source = source)
                RETURN elementId(d) as node_id, d.text as text, d.source as source
            """, {"sources": sources_list})
            
            if not document_nodes:
                print("⚠️  No Document nodes found in current session")
                return
            
            print(f"📄 Found {len(document_nodes)} Document nodes from current session to process")
            
            # Parallel processing of document properties
            added_properties, properties_to_embed = self._parallel_process_document_properties(document_nodes)
            
            print(f"✅ Successfully processed multi-vector properties for {len(document_nodes)} Document nodes")
            
            # Create vector embeddings for the new properties
            if params.add_vector_index and properties_to_embed:
                print("🔍 Creating vector embeddings for new document properties...")
                
                # Show which properties will get embeddings
                properties_with_embeddings = {item['property_name'] for item in properties_to_embed}
                print(f"ℹ️  Creating embeddings for properties: {sorted(properties_with_embeddings)}")
                
                # Ensure embedding services are initialized
                self._ensure_services_initialized()
                
                # Create embeddings in parallel batches
                self._parallel_create_embeddings_batch(properties_to_embed, "document_node")
                
                # Create vector indexes for the new properties
                self._parallel_create_document_property_indexes(added_properties)
                
                # Verify the embeddings were created
                self._verify_document_property_embeddings(added_properties)
                
                print("✅ Vector embeddings for document properties created successfully")
            else:
                if not params.add_vector_index:
                    print("ℹ️  Vector embeddings disabled in configuration")
                elif not properties_to_embed:
                    print("ℹ️  No valid properties found for embedding")
                else:
                    print("ℹ️  No properties to embed")
            
        except Exception as e:
            print(f"⚠️  Error adding properties to Document nodes: {e}")
            import traceback
            traceback.print_exc()
    
    def _parallel_process_document_properties(self, document_nodes: List[Dict]) -> Tuple[set, List[Dict]]:
        """Process document properties in parallel using ThreadPoolExecutor."""
        
        def process_single_property(doc_node: Dict, prop_config: Dict) -> Dict:
            """Process a single document-property combination."""
            node_id = doc_node['node_id']
            document_text = doc_node['text']
            source = doc_node.get('source', 'unknown')
            property_name = prop_config.get('property_name')
            prompt = prop_config.get('prompt')
            
            if not property_name or not prompt:
                return {'error': f"Invalid property config: {prop_config}"}
            
            try:
                # Create the full prompt with document text
                full_prompt = f"{prompt}\n\nDocument text:\n{document_text}"
                
                # Call LLM to generate the property value
                response = self.llm.invoke(full_prompt)
                property_value = response.content.strip()
                
                return {
                    'node_id': node_id,
                    'property_name': property_name,
                    'property_value': property_value,
                    'document_text': document_text,
                    'source': source,
                    'success': True
                }
                
            except Exception as e:
                return {
                    'error': f"Error processing {property_name} for {source}: {e}",
                    'node_id': node_id,
                    'property_name': property_name,
                    'success': False
                }
        
        # Create all combinations of documents and properties
        tasks = []
        for doc_node in document_nodes:
            for prop_config in params.document_multi_vector_properties:
                tasks.append((doc_node, prop_config))
        
        print(f"🚀 Processing {len(tasks)} document-property combinations in parallel...")
        
        # Process in parallel with optimal number of workers
        max_workers = min(8, len(tasks), os.cpu_count() or 4)
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_single_property, doc_node, prop_config): (doc_node, prop_config)
                for doc_node, prop_config in tasks
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"📝 Completed {completed}/{len(tasks)} property generations")
        
        # Process results and update Neo4j in batches
        return self._batch_update_document_properties(results)
    
    def _batch_update_document_properties(self, results: List[Dict]) -> Tuple[set, List[Dict]]:
        """Update document properties in Neo4j using batch operations."""
        added_properties = set()
        properties_to_embed = []
        successful_updates = []
        
        # Separate successful and failed results
        for result in results:
            if result.get('success'):
                successful_updates.append(result)
            else:
                if 'error' in result:
                    print(f"⚠️  {result['error']}")
        
        if not successful_updates:
            print("⚠️  No successful property generations to update")
            return added_properties, properties_to_embed
        
        # Group updates by property name for batch processing
        property_groups = {}
        for result in successful_updates:
            prop_name = result['property_name']
            if prop_name not in property_groups:
                property_groups[prop_name] = []
            property_groups[prop_name].append(result)
        
        # Batch update each property type
        for property_name, group in property_groups.items():
            print(f"📝 Batch updating {len(group)} nodes with property '{property_name}'")
            
            try:
                # Prepare batch update query
                batch_params = []
                for result in group:
                    batch_params.append({
                        'node_id': result['node_id'],
                        'property_value': result['property_value']
                    })
                
                # Execute batch update
                self.graph.query(f"""
                    UNWIND $batch_params as param
                    MATCH (d) 
                    WHERE elementId(d) = param.node_id 
                    SET d.{property_name} = param.property_value
                """, {'batch_params': batch_params})
                
                added_properties.add(property_name)
                
                # Prepare for embedding creation
                for result in group:
                    if self._is_valid_embedding_value(result['property_value'], property_name):
                        context_text = f"Node type: Document | Property {property_name}: {result['property_value']}"
                        
                        properties_to_embed.append({
                            'node_id': result['node_id'],
                            'property_name': property_name,
                            'text': context_text,
                            'embedding_name': f'embedding_{property_name}',
                            'type': 'node'
                        })
                
                print(f"✅ Successfully updated {len(group)} nodes with property '{property_name}'")
                
            except Exception as e:
                print(f"⚠️  Error batch updating property '{property_name}': {e}")
        
        return added_properties, properties_to_embed
    
    def _parallel_create_document_property_indexes(self, added_properties: set) -> None:
        """Create vector indexes for document properties in parallel."""
        if not added_properties:
            print("ℹ️  No document properties to index")
            return
        
        print("🔍 Creating vector indexes for new document properties...")
        print(f"Creating indexes for document properties: {sorted(added_properties)}")
        
        def create_single_index(prop: str) -> Tuple[str, bool, str]:
            """Create a single vector index."""
            try:
                index_name = f"embedding_{prop}_document_index"
                self.graph.query(f"""
                    CREATE VECTOR INDEX {index_name}
                    IF NOT EXISTS
                    FOR (n:Document) ON (n.embedding_{prop})
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {params.vector_embedding_dimensions},
                            `vector.similarity_function`: '{params.vector_similarity_function}'
                        }}
                    }}
                """)
                return prop, True, f"Created vector index for Document.embedding_{prop}"
                
            except Exception as e:
                return prop, False, f"Could not create vector index for Document.embedding_{prop}: {e}"
        
        # Create indexes in parallel
        max_workers = min(4, len(added_properties))
        created_indexes = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prop = {
                executor.submit(create_single_index, prop): prop
                for prop in added_properties
            }
            
            for future in as_completed(future_to_prop):
                prop, success, message = future.result()
                if success:
                    print(f"✅ {message}")
                    created_indexes += 1
                else:
                    print(f"⚠️  {message}")
        
        print(f"✅ Successfully created {created_indexes} vector indexes for document properties")
    
    def _verify_document_property_embeddings(self, added_properties: set) -> None:
        """Verify that embeddings were successfully created for document properties in current session only."""
        print("\n📊 Verifying document property embeddings for current session...")
        
        if not added_properties:
            print("ℹ️  No document properties to verify")
            return
        
        if not self.current_session_sources:
            print("ℹ️  No current session sources to verify")
            return
        
        try:
            total_embeddings = 0
            sources_list = list(self.current_session_sources)
            
            for prop in sorted(added_properties):
                # Count Document nodes from current session with this embedding
                count_result = self.graph.query(f"""
                    MATCH (d:Document) 
                    WHERE any(source IN $sources WHERE d.source CONTAINS source OR d.source = source)
                    AND d.embedding_{prop} IS NOT NULL 
                    RETURN count(d) as count
                """, {"sources": sources_list})
                
                count = count_result[0]['count'] if count_result else 0
                total_embeddings += count
                
                if count > 0:
                    print(f"✅ embedding_{prop}: {count} Document nodes from current session")
                else:
                    print(f"⚠️  embedding_{prop}: No embeddings found in current session Document nodes")
            
            print(f"\n🎉 Total document property embeddings created for current session: {total_embeddings}")
            
        except Exception as e:
            print(f"⚠️  Error verifying document property embeddings: {e}")
    
    def _reset_session_tracking(self) -> None:
        """Reset session tracking for a new ingestion run."""
        self.current_session_sources.clear()
        self.current_session_nodes.clear()
        self.current_session_relationships.clear()
        print("🔄 Reset session tracking for new ingestion run")
    
    def run_ingestion(self) -> None:
        """Execute the complete ingestion process."""
        print("🚀 Starting Knowledge Graph Ingestion Process")
        print("=" * 50)
        
        try:
            # Reset session tracking for this run
            self._reset_session_tracking()
            
            # Setup phase
            self.setup_environment()
            self.initialize_neo4j_connection()
            self.setup_llm_and_transformer()
            
            # Data processing phase
            self.load_documents()
            graph_docs = self.extract_graph_documents()
            
            # Ingestion phase
            self.ingest_to_neo4j(graph_docs)

            # Vector embeddings phase (optional) - only for current session
            self.create_vector_embeddings()
            
            # Add custom properties to Document nodes - only for current session
            self.add_multivectors_to_document_nodes()
            
            print("=" * 50)
            print("🎉 Knowledge Graph Ingestion Process Completed Successfully!")
            print(f"📊 Processed {len(self.current_session_sources)} source files")
            print(f"📊 Created embeddings for {len(self.current_session_nodes)} nodes and {len(self.current_session_relationships)} relationships")
            
        except Exception as e:
            print(f"❌ Ingestion process failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function to run the ingestion process."""
    ingestion = KnowledgeGraphIngestion()
    ingestion.run_ingestion()


if __name__ == "__main__":
    main()
