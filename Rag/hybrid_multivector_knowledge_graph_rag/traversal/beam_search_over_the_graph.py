import os
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from sklearn.metrics.pairwise import cosine_similarity

class BeamSearchOverGraph:
    """
    A graph traversal class that implements Beam Search over the graph.
    This approach explores the graph in layers (like BFS), but at each layer
    only keeps the top-w most promising nodes based on cosine similarity scores.
    """
    
    def __init__(
        self, 
        beam_width: int = 10,
        max_depth: int = 3,
        max_total_nodes: int = 100,
        remove_mentions_nodes: bool = True,
        rel_type_filter: Optional[List[str]] = None
    ):
        """
        Initialize the Beam Search traversal.
        
        Args:
            beam_width: Number of top candidates to keep at each layer (default: 10)
            max_depth: Maximum depth to explore (default: 3)
            max_total_nodes: Maximum total nodes to return (default: 100)
            remove_mentions_nodes: If True, filter out nodes discovered through MENTIONS relationships (default: True)
            rel_type_filter: List of relationship types to filter by (default: None = no filtering)
        """
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.max_total_nodes = max_total_nodes
        self.remove_mentions_nodes = remove_mentions_nodes
        self.rel_type_filter = rel_type_filter
        self._setup_environment()
        self._initialize_neo4j_connection()
    
    def _setup_environment(self) -> None:
        """Load environment variables and validate required credentials."""
        load_dotenv()
        
        # Neo4j Credentials validation
        required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def _initialize_neo4j_connection(self) -> None:
        """Initialize Neo4j graph connection."""
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            refresh_schema=False,
        )
    
    def traverse_graph(self, relevant_docs: List[Dict[str, Any]], user_query: str) -> List[Dict[str, Any]]:
        """
        Retrieve additional context through Beam Search traversal.
        
        This method:
        1. Extracts node IDs from initial documents
        2. Performs beam search to find connected nodes in layers
        3. Returns additional context nodes with proper formatting
        
        Args:
            relevant_docs: List of initially retrieved documents
            user_query: The user's question
            
        Returns:
            List of additional context nodes with embedding properties removed
        """
        try:
            # Extract starting node IDs from relevant documents
            starting_node_ids = self._extract_node_ids(relevant_docs)
            
            if not starting_node_ids:
                print("No starting nodes found from initial documents")
                return []
            
            print(f"🔍 Starting Beam Search from {len(starting_node_ids)} initial nodes")
            print(f"📊 Parameters: beam_width={self.beam_width}, max_depth={self.max_depth}")
            
            # Perform beam search traversal
            additional_nodes = self._perform_beam_search(starting_node_ids)
            
            if additional_nodes:
                print(f"✅ Retrieved {len(additional_nodes)} additional context nodes via Beam Search")
                
                # Filter embedding properties from results
                filtered_items = self._filter_properties(additional_nodes)
                
                return filtered_items
            else:
                print("No additional context found through Beam Search traversal")
                return []
                
        except Exception as e:
            print(f"⚠️ Error in Beam Search traversal: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_node_ids(self, relevant_docs: List[Dict[str, Any]]) -> List[str]:
        """Extract node IDs from relevant documents."""
        node_ids = []
        
        for doc in relevant_docs:
            if 'node_id' in doc:
                node_ids.append(doc['node_id'])
        
        return list(set(node_ids))  # Remove duplicates
    
    def _perform_beam_search(self, starting_node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Perform Beam Search traversal starting from the given node IDs.
        
        Args:
            starting_node_ids: List of node IDs to start traversal from
            
        Returns:
            List of discovered nodes with metadata
        """
        # Initialize beam search data structures
        visited: Set[str] = set()
        discovered_nodes: List[Dict[str, Any]] = []
        
        # Current beam (frontier) - starts with seed nodes
        current_beam: List[Tuple[str, float, int, Optional[str]]] = []  # (node_id, score, depth, parent_node_id)
        
        # Track node scores for final results
        node_scores: Dict[str, float] = {}
        
        # Initialize beam with starting nodes (score 1.0 for seeds, depth 0)
        for node_id in starting_node_ids:
            current_beam.append((node_id, 1.0, 0, None))
            visited.add(node_id)
            node_scores[node_id] = 1.0
        
        print(f"🚀 Starting Beam Search traversal from {len(starting_node_ids)} nodes")
        
        # Get embeddings for all starting nodes
        starting_embeddings = self._get_node_embeddings_batch(starting_node_ids)
        
        # Beam search traversal - layer by layer
        for depth in range(self.max_depth):
            if not current_beam:
                break
                
            print(f"🔄 Layer {depth + 1}: Expanding {len(current_beam)} nodes in current beam")
            
            # Collect all candidates for next layer
            next_layer_candidates: List[Tuple[float, str, int, str, Dict[str, Any]]] = []  # (score, node_id, depth, parent_id, metadata)
            
            # Process each node in current beam
            for beam_node_id, beam_score, beam_depth, beam_parent in current_beam:
                # Add current beam node to discovered nodes (if not already added)
                if beam_depth > 0:  # Skip seed nodes as they're already in relevant_docs
                    # Get node metadata with actual score
                    node_metadata = self._get_node_metadata(beam_node_id, beam_parent, beam_depth, beam_score)
                    if node_metadata:
                        # Only add if it's not a MENTIONS relationship or if we allow MENTIONS
                        if node_metadata.get('relationship_type') != 'MENTIONS' or not self.remove_mentions_nodes:
                            discovered_nodes.append(node_metadata)
                
                # Get current node embedding
                current_embedding = starting_embeddings.get(beam_node_id)
                if current_embedding is None:
                    current_embedding = self._get_node_embedding(beam_node_id)
                
                if current_embedding is None:
                    print(f"⚠️ Skipping node {beam_node_id[:8]}... - no embedding found")
                    continue
                
                # Get neighbors of current node
                neighbors = self._get_node_neighbors(beam_node_id, beam_depth)
                
                # Get embeddings for all neighbors
                neighbor_ids = [neighbor['node_id'] for neighbor in neighbors]
                neighbor_embeddings = self._get_node_embeddings_batch(neighbor_ids)
                
                # Calculate scores for each neighbor
                for neighbor in neighbors:
                    neighbor_id = neighbor['node_id']
                    
                    # Skip if already visited
                    if neighbor_id in visited:
                        continue
                    
                    # Skip MENTIONS relationships if configured and depth > 0
                    if neighbor.get('relationship_type') == 'MENTIONS' and self.remove_mentions_nodes and beam_depth > 0:
                        continue
                    
                    # Get neighbor embedding
                    neighbor_embedding = neighbor_embeddings.get(neighbor_id)
                    if neighbor_embedding is None:
                        continue
                    
                    # Calculate similarity score (higher = better)
                    similarity_score = self._calculate_cosine_similarity(current_embedding, neighbor_embedding)
                    
                    # Add to candidates for next layer
                    next_layer_candidates.append((
                        similarity_score,
                        neighbor_id,
                        beam_depth + 1,
                        beam_node_id,
                        neighbor
                    ))
            
            # Sort candidates by score (descending - higher is better)
            next_layer_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Keep only top beam_width candidates
            top_candidates = next_layer_candidates[:self.beam_width]
            
            # Prepare next beam
            current_beam = []
            for score, node_id, depth, parent_id, metadata in top_candidates:
                if node_id not in visited:
                    visited.add(node_id)
                    current_beam.append((node_id, score, depth, parent_id))
                    node_scores[node_id] = score  # Store the actual similarity score
            
            print(f"📊 Layer {depth + 1}: Selected {len(current_beam)} candidates for next beam (from {len(next_layer_candidates)} total)")
            
            # Check if we've reached the limit
            if len(discovered_nodes) >= self.max_total_nodes:
                print(f"🛑 Reached maximum node limit ({self.max_total_nodes})")
                break
        
        # Add any remaining beam nodes to discovered nodes
        for beam_node_id, beam_score, beam_depth, beam_parent in current_beam:
            if beam_depth > 0 and len(discovered_nodes) < self.max_total_nodes:
                node_metadata = self._get_node_metadata(beam_node_id, beam_parent, beam_depth, beam_score)
                if node_metadata:
                    # Only add if it's not a MENTIONS relationship or if we allow MENTIONS
                    if node_metadata.get('relationship_type') != 'MENTIONS' or not self.remove_mentions_nodes:
                        discovered_nodes.append(node_metadata)
        
        print(f"🎯 Beam Search traversal completed: {len(discovered_nodes)} nodes discovered")
        return discovered_nodes[:self.max_total_nodes]
    
    def _get_node_neighbors(self, node_id: str, depth: int = 0) -> List[Dict[str, Any]]:
        """
        Get neighbors of a node (both incoming and outgoing relationships).
        
        Args:
            node_id: ID of the node to get neighbors for
            depth: Current depth in the search
            
        Returns:
            List of neighbor nodes with metadata
        """
        try:
            # Apply relationship filtering if specified
            if self.rel_type_filter:
                # Filter by specific relationship types
                cypher_query = """
                MATCH (central)-[r]-(neighbor)
                WHERE elementId(central) = $node_id AND type(r) IN $rel_types
                WITH neighbor, r, central
                RETURN DISTINCT
                    elementId(neighbor) as node_id,
                    labels(neighbor) as labels,
                    properties(neighbor) as properties,
                    type(r) as relationship_type,
                    1.0 as score,
                    'beam_search_neighbor' as search_type
                """
                
                result = self.graph.query(cypher_query, {
                    "node_id": node_id,
                    "rel_types": self.rel_type_filter
                })
                
                print(f"📊 Depth {depth}: Found {len(result)} neighbors for node {node_id[:8]}... using filtered relationships")
                
            else:
                # No filtering - get all neighbors
                cypher_query = """
                MATCH (central)-[r]-(neighbor)
                WHERE elementId(central) = $node_id
                WITH neighbor, r, central
                RETURN DISTINCT
                    elementId(neighbor) as node_id,
                    labels(neighbor) as labels,
                    properties(neighbor) as properties,
                    type(r) as relationship_type,
                    1.0 as score,
                    'beam_search_neighbor' as search_type
                """
                
                result = self.graph.query(cypher_query, {
                    "node_id": node_id
                })
                
                print(f"📊 Depth {depth}: Found {len(result)} neighbors for node {node_id[:8]}... using all relationships")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error getting neighbors for node {node_id}: {e}")
            return []
    
    def _get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        Get the merged embedding for a node by concatenating all its embedding properties.
        Handles dimensional consistency by padding shorter embeddings.
        
        Args:
            node_id: ID of the node to get embedding for
            
        Returns:
            Merged embedding as numpy array, or None if no embeddings found
        """
        try:
            cypher_query = """
            MATCH (n)
            WHERE elementId(n) = $node_id
            RETURN properties(n) as all_properties
            """
            
            result = self.graph.query(cypher_query, {"node_id": node_id})
            
            if not result or len(result) == 0:
                return None
                
            properties = result[0]['all_properties']
            if not properties:
                return None
            
            # Find all embedding properties
            embedding_properties = []
            for key, value in properties.items():
                if key.startswith('embedding_') and isinstance(value, list):
                    try:
                        embedding_array = np.array(value, dtype=np.float32)
                        if embedding_array.size > 0:
                            embedding_properties.append(embedding_array)
                    except (ValueError, TypeError):
                        continue
            
            if not embedding_properties:
                return None
            
            # Get the maximum expected dimension based on your configuration
            # From params.py: you have text, hyp_queries, italian_translation (potentially 3-4 embeddings)
            max_expected_embeddings = 4  # Adjust based on your document_multi_vector_properties
            base_dimension = 1536  # From your vector_embedding_dimensions in params.py
            target_dimension = max_expected_embeddings * base_dimension
            
            # Concatenate all available embeddings
            concatenated_embedding = np.concatenate(embedding_properties, axis=0)
            
            # Pad with zeros if the concatenated embedding is shorter than target
            if concatenated_embedding.shape[0] < target_dimension:
                padding_size = target_dimension - concatenated_embedding.shape[0]
                concatenated_embedding = np.pad(concatenated_embedding, (0, padding_size), mode='constant', constant_values=0)
            
            # Truncate if longer than target (shouldn't happen but safety check)
            elif concatenated_embedding.shape[0] > target_dimension:
                concatenated_embedding = concatenated_embedding[:target_dimension]
            
            # L2 normalize the final embedding for better cosine similarity
            norm = np.linalg.norm(concatenated_embedding)
            if norm > 0:
                concatenated_embedding = concatenated_embedding / norm
            
            return concatenated_embedding
            
        except Exception as e:
            print(f"⚠️ Error getting embedding for node {node_id}: {e}")
            return None
    
    def _get_node_embeddings_batch(self, node_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple nodes in a batch operation.
        
        Args:
            node_ids: List of node IDs to get embeddings for
            
        Returns:
            Dictionary mapping node_id to merged embedding
        """
        embeddings = {}
        
        for node_id in node_ids:
            embedding = self._get_node_embedding(node_id)
            if embedding is not None:
                embeddings[node_id] = embedding
        
        return embeddings
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity, range [-1, 1] where higher values indicate more similarity
        """
        try:
            # Reshape for sklearn cosine_similarity
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            # Ensure similarity is in valid range
            return max(-1.0, min(1.0, similarity))
            
        except Exception as e:
            print(f"⚠️ Error calculating cosine similarity: {e}")
            # Return a default low similarity if calculation fails
            return 0.0
    
    def _get_node_metadata(self, node_id: str, parent_id: Optional[str], depth: int, score: float) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a node including its relationship to the parent.
        
        Args:
            node_id: ID of the node to get metadata for
            parent_id: ID of the parent node (if any)
            depth: Current depth in the search
            score: The similarity score of the node
            
        Returns:
            Dictionary with node metadata, or None if not found
        """
        try:
            if parent_id:
                # Get relationship information
                cypher_query = """
                MATCH (parent)-[r]-(node)
                WHERE elementId(parent) = $parent_id AND elementId(node) = $node_id
                WITH node, r, parent
                RETURN DISTINCT
                    elementId(node) as node_id,
                    labels(node) as labels,
                    properties(node) as properties,
                    type(r) as relationship_type,
                    $score as score,
                    'beam_search_result' as search_type
                LIMIT 1
                """
                
                result = self.graph.query(cypher_query, {
                    "parent_id": parent_id,
                    "node_id": node_id,
                    "score": score
                })
                
                if result and len(result) > 0:
                    metadata = result[0].copy()
                    metadata['depth'] = depth
                    metadata['discovered_from_node_id'] = parent_id
                    
                    # Get parent node name
                    parent_name = self._get_node_name_id_property(parent_id)
                    
                    # Set related_to_node_name conditionally based on relationship type
                    if metadata.get('relationship_type') == 'MENTIONS':
                        metadata['related_to_node_name'] = 'Document'
                    else:
                        metadata['related_to_node_name'] = parent_name
                    
                    return metadata
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error getting metadata for node {node_id}: {e}")
            return None
    
    def _get_node_name_id_property(self, node_id: str) -> str:
        """
        Get the "id" property value from a node.
        
        Args:
            node_id: ID of the node to get the "id" property from
            
        Returns:
            The "id" property value, or the node_id if property doesn't exist
        """
        try:
            cypher_query = """
            MATCH (n)
            WHERE elementId(n) = $node_id
            RETURN n.id as id_property, properties(n) as all_properties
            """
            
            result = self.graph.query(cypher_query, {"node_id": node_id})
            
            if result and len(result) > 0:
                id_property = result[0]['id_property']
                all_properties = result[0]['all_properties']
                
                if id_property is not None:
                    return str(id_property)
                else:
                    if 'name' in all_properties:
                        return str(all_properties['name'])
                    else:
                        return node_id
            else:
                return node_id
                
        except Exception as e:
            print(f"⚠️ Error getting id property for node {node_id}: {e}")
            return node_id
    
    def _filter_properties(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out embedding properties from the results.
        
        Args:
            items: List of items to filter
            
        Returns:
            List of items with embedding properties removed
        """
        filtered_items = []
        
        for item in items:
            filtered_item = item.copy()
            
            # Filter properties if they exist
            if 'properties' in filtered_item and filtered_item['properties']:
                filtered_properties = {}
                for key, value in filtered_item['properties'].items():
                    # Skip embedding properties
                    if not key.startswith('embedding_'):
                        filtered_properties[key] = value
                filtered_item['properties'] = filtered_properties
            
            filtered_items.append(filtered_item)
        
        return filtered_items
    
    def get_traversal_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the traversal configuration.
        
        Returns:
            Dictionary containing traversal statistics
        """
        return {
            "method": "beam_search",
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "max_total_nodes": self.max_total_nodes,
            "remove_mentions_nodes": self.remove_mentions_nodes,
            "rel_type_filter": self.rel_type_filter
        } 