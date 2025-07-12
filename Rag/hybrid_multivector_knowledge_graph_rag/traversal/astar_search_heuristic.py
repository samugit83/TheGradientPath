import os
import heapq
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from sklearn.metrics.pairwise import cosine_similarity

class AStarSearchHeuristic:
    """
    A graph traversal class that implements A* Search (Best-First Search with Heuristic).
    This approach finds nodes using f(n) = g(n) + h(n), where:
    - g(n) is the actual cost from seed nodes (cosine distance between node embeddings)
    - h(n) is a heuristic estimate of remaining cost to the query (cosine distance to query embedding)
    
    The heuristic h(n) = 1 - cos(embed(n), embed(query)) is admissible and guides the search
    toward nodes that are both structurally close (low g) and semantically close to the query (low h).
    """
    
    def __init__(
        self, 
        max_total_nodes: int = 100,
        remove_mentions_nodes: bool = True,
        rel_type_filter: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None
    ):
        """
        Initialize the A* Search traversal.
        
        Args:
            max_total_nodes: Maximum total nodes to return (default: 100)
            remove_mentions_nodes: If True, filter out nodes discovered through MENTIONS relationships (default: True)
            rel_type_filter: List of relationship types to filter by (default: None = no filtering)
            query_embedding: Pre-computed query embedding for heuristic calculation (default: None)
        """
        self.max_total_nodes = max_total_nodes
        self.remove_mentions_nodes = remove_mentions_nodes
        self.rel_type_filter = rel_type_filter
        self.query_embedding = query_embedding
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
        Retrieve additional context through A* Search traversal.
        
        This method:
        1. Uses the pre-computed query embedding for the heuristic function
        2. Extracts node IDs from initial documents
        3. Performs A* search to find connected nodes using f(n) = g(n) + h(n)
        4. Returns additional context nodes with proper formatting
        
        Args:
            relevant_docs: List of initially retrieved documents
            user_query: The user's question (for logging purposes)
            
        Returns:
            List of additional context nodes with embedding properties removed
        """
        try:
            # Use the pre-computed query embedding for heuristic function
            query_embedding = self._prepare_query_embedding()
            if query_embedding is None:
                print("⚠️ Could not prepare query embedding, falling back to UCS behavior")
                query_embedding = None
            
            # Extract starting node IDs from relevant documents
            starting_node_ids = self._extract_node_ids(relevant_docs)
            
            if not starting_node_ids:
                print("No starting nodes found from initial documents")
                return []
            
            print(f"🔍 Starting A* Search from {len(starting_node_ids)} initial nodes")
            print(f"📊 Parameters: max_total_nodes={self.max_total_nodes}")
            print(f"🎯 Query embedding available: {'✅' if query_embedding is not None else '❌'}")
            
            # Perform A* traversal
            additional_nodes = self._perform_astar(starting_node_ids, query_embedding)
            
            if additional_nodes:
                print(f"✅ Retrieved {len(additional_nodes)} additional context nodes via A*")
                
                # Filter embedding properties from results
                filtered_items = self._filter_properties(additional_nodes)
                
                return filtered_items
            else:
                print("No additional context found through A* traversal")
                return []
                
        except Exception as e:
            print(f"⚠️ Error in A* traversal: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _prepare_query_embedding(self) -> Optional[np.ndarray]:
        """
        Prepare the query embedding for use in A* Search.
        
        Returns:
            Query embedding as numpy array, or None if preparation fails
        """
        try:
            if self.query_embedding is None:
                print("⚠️ No query embedding provided - A* will fall back to UCS behavior")
                return None
            
            # Convert query embedding to numpy array
            query_embedding = np.array(self.query_embedding, dtype=np.float32)
            
            # Ensure the embedding has the expected dimension
            # From params.py: base_dimension = 1536, max_expected_embeddings = 4
            max_expected_embeddings = 4
            base_dimension = 1536
            target_dimension = max_expected_embeddings * base_dimension
            
            # Pad with zeros if the query embedding is shorter than target
            if query_embedding.shape[0] < target_dimension:
                padding_size = target_dimension - query_embedding.shape[0]
                query_embedding = np.pad(query_embedding, (0, padding_size), mode='constant', constant_values=0)
            
            # Truncate if longer than target
            elif query_embedding.shape[0] > target_dimension:
                query_embedding = query_embedding[:target_dimension]
            
            # L2 normalize the final embedding for better cosine similarity
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            
            return query_embedding
            
        except Exception as e:
            print(f"⚠️ Error preparing query embedding: {e}")
            return None
    
    def _extract_node_ids(self, relevant_docs: List[Dict[str, Any]]) -> List[str]:
        """Extract node IDs from relevant documents."""
        node_ids = []
        
        for doc in relevant_docs:
            if 'node_id' in doc:
                node_ids.append(doc['node_id'])
        
        return list(set(node_ids))  # Remove duplicates
    
    def _perform_astar(self, starting_node_ids: List[str], query_embedding: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Perform A* Search traversal starting from the given node IDs.
        
        Args:
            starting_node_ids: List of node IDs to start traversal from
            query_embedding: Query embedding for heuristic calculation (None falls back to UCS)
            
        Returns:
            List of discovered nodes with metadata
        """
        # Initialize A* data structures
        open_set: List[Tuple[float, str, int]] = []  # (f_score, node_id, level)
        closed_set: Set[str] = set()  # Nodes that have been fully expanded
        discovered_nodes: List[Dict[str, Any]] = []
        g_score: Dict[str, float] = {}  # Best known cost from start to each node
        node_metadata: Dict[str, Dict[str, Any]] = {}  # Store metadata for nodes
        
        # Get embeddings for all starting nodes
        starting_embeddings = self._get_node_embeddings_batch(starting_node_ids)
        
        # Initialize open set with starting nodes
        for node_id in starting_node_ids:
            g_score[node_id] = 0.0
            
            # Calculate heuristic for starting node
            h_score = 0.0
            if query_embedding is not None and node_id in starting_embeddings:
                h_score = self._calculate_heuristic(starting_embeddings[node_id], query_embedding)
            
            f_score = g_score[node_id] + h_score
            heapq.heappush(open_set, (f_score, node_id, 0))
            
            # Store metadata for starting nodes
            node_metadata[node_id] = {
                'node_id': node_id,
                'g_score': 0.0,
                'h_score': h_score,
                'f_score': f_score,
                'edge_cost': 0.0,
                'discovered_from_node_id': None,
                'related_to_node_name': 'Starting Node',
                'level': 0
            }
        
        print(f"🚀 Starting A* traversal from {len(starting_node_ids)} nodes")
        
        # A* traversal
        while open_set and len(discovered_nodes) < self.max_total_nodes:
            current_f_score, current_node_id, current_level = heapq.heappop(open_set)
            
            # Skip if already fully expanded
            if current_node_id in closed_set:
                continue
            
            # Mark as fully expanded
            closed_set.add(current_node_id)
            
            # Add current node to discovered nodes
            if current_node_id in node_metadata:
                node_data = node_metadata[current_node_id]
                # Only add to discovered nodes if it's not a MENTIONS relationship or if we allow MENTIONS
                if node_data.get('relationship_type') != 'MENTIONS' or not self.remove_mentions_nodes:
                    discovered_nodes.append(node_data)
            
            # Get current node properties to extract the "id" property
            current_node_name = self._get_node_name_id_property(current_node_id)
            
            # Get current node embedding
            current_embedding = starting_embeddings.get(current_node_id)
            if current_embedding is None:
                current_embedding = self._get_node_embedding(current_node_id)
            
            if current_embedding is None:
                print(f"⚠️ Skipping node {current_node_id[:8]}... - no embedding found")
                continue
            
            # Get neighbors of current node
            neighbors = self._get_node_neighbors(current_node_id)
            
            # Get embeddings for all neighbors
            neighbor_ids = [neighbor['node_id'] for neighbor in neighbors]
            neighbor_embeddings = self._get_node_embeddings_batch(neighbor_ids)
            
            # Process each neighbor
            for neighbor in neighbors:
                neighbor_id = neighbor['node_id']
                
                # Skip MENTIONS relationships only if we're beyond the initial level
                if neighbor.get('relationship_type') == 'MENTIONS' and self.remove_mentions_nodes and current_level > 0:
                    continue
                
                # Skip if already fully expanded
                if neighbor_id in closed_set:
                    continue
                
                # Get neighbor embedding
                neighbor_embedding = neighbor_embeddings.get(neighbor_id)
                if neighbor_embedding is None:
                    print(f"⚠️ Skipping neighbor {neighbor_id[:8]}... - no embedding found")
                    continue
                
                # Calculate edge cost using cosine distance
                edge_cost = self._calculate_cosine_distance(current_embedding, neighbor_embedding)
                tentative_g_score = g_score[current_node_id] + edge_cost
                
                # Update if this is a better path to the neighbor
                if tentative_g_score < g_score.get(neighbor_id, float('inf')):
                    g_score[neighbor_id] = tentative_g_score
                    
                    # Calculate heuristic for neighbor
                    h_score = 0.0
                    if query_embedding is not None:
                        h_score = self._calculate_heuristic(neighbor_embedding, query_embedding)
                    
                    f_score = tentative_g_score + h_score
                    heapq.heappush(open_set, (f_score, neighbor_id, current_level + 1))
                    
                    # Store neighbor metadata
                    neighbor_info = neighbor.copy()
                    neighbor_info['g_score'] = tentative_g_score
                    neighbor_info['h_score'] = h_score
                    neighbor_info['f_score'] = f_score
                    neighbor_info['edge_cost'] = edge_cost
                    neighbor_info['discovered_from_node_id'] = current_node_id
                    neighbor_info['level'] = current_level + 1
                    
                    # Set related_to_node_name conditionally based on relationship type
                    if neighbor_info.get('relationship_type') == 'MENTIONS':
                        neighbor_info['related_to_node_name'] = 'Document'
                    else:
                        neighbor_info['related_to_node_name'] = current_node_name
                    
                    # Store metadata for when the node is actually visited
                    node_metadata[neighbor_id] = neighbor_info
            
            # Log progress periodically
            if len(discovered_nodes) % 50 == 0 and len(discovered_nodes) > 0:
                print(f"📊 Discovered {len(discovered_nodes)} nodes, current f-score: {current_f_score:.4f}")
        
        # Sort discovered nodes by f-score (A* priority order)
        discovered_nodes.sort(key=lambda x: x.get('f_score', x.get('g_score', 0.0)))
        
        print(f"🎯 A* traversal completed: {len(discovered_nodes)} nodes discovered")
        return discovered_nodes[:self.max_total_nodes]
    
    def _calculate_heuristic(self, node_embedding: np.ndarray, query_embedding: np.ndarray) -> float:
        """
        Calculate heuristic h(n) = 1 - cos(embed(n), embed(query)).
        
        This heuristic is admissible because it never overestimates the true cost
        to reach a highly relevant node - any path through the graph can only
        add more distance.
        
        Args:
            node_embedding: Embedding of the node
            query_embedding: Embedding of the query
            
        Returns:
            Heuristic cost estimate (lower = more similar to query)
        """
        try:
            # Calculate cosine distance between node and query embeddings
            return self._calculate_cosine_distance(node_embedding, query_embedding)
        except Exception as e:
            print(f"⚠️ Error calculating heuristic: {e}")
            # Return neutral heuristic if calculation fails
            return 0.0
    
    def _get_node_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get neighbors of a node (both incoming and outgoing relationships).
        
        Args:
            node_id: ID of the node to get neighbors for
            
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
                    'ucs_neighbor' as search_type
                """
                
                result = self.graph.query(cypher_query, {
                    "node_id": node_id,
                    "rel_types": self.rel_type_filter
                })
                
                print(f"📊 Found {len(result)} neighbors for node {node_id[:8]}... using filtered relationships")
                
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
                    'ucs_neighbor' as search_type
                """
                
                result = self.graph.query(cypher_query, {
                    "node_id": node_id
                })
                
                print(f"📊 Found {len(result)} neighbors for node {node_id[:8]}... using all relationships")
            
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
    
    def _calculate_cosine_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine distance between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine distance (1 - cosine_similarity), range [0, 2]
        """
        try:
            # Reshape for sklearn cosine_similarity
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            # Convert to distance (lower = more similar)
            distance = 1.0 - similarity
            
            # Ensure non-negative distance
            return max(0.0, distance)
            
        except Exception as e:
            print(f"⚠️ Error calculating cosine distance: {e}")
            # Return a default high cost if calculation fails
            return 1.0
    
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
            "method": "astar_search",
            "max_total_nodes": self.max_total_nodes,
            "remove_mentions_nodes": self.remove_mentions_nodes,
            "rel_type_filter": self.rel_type_filter,
            "heuristic": "cosine_distance_to_query"
        }
