import os
import heapq
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from sklearn.metrics.pairwise import cosine_similarity

class UniformCostSearchUCS:
    """
    A graph traversal class that implements Uniform-Cost Search (UCS).
    This approach finds nodes in order of increasing total cost from seed nodes,
    where cost is calculated using cosine distance between node embeddings.
    """
    
    def __init__(
        self, 
        max_total_nodes: int = 100,
        remove_mentions_nodes: bool = True,
        rel_type_filter: Optional[List[str]] = None
    ):
        """
        Initialize the Uniform-Cost Search traversal.
        
        Args:
            max_total_nodes: Maximum total nodes to return (default: 100)
            remove_mentions_nodes: If True, filter out nodes discovered through MENTIONS relationships (default: True)
            rel_type_filter: List of relationship types to filter by (default: None = no filtering)
        """
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
        Retrieve additional context through Uniform-Cost Search traversal.
        
        This method:
        1. Extracts node IDs from initial documents
        2. Performs UCS to find connected nodes in order of increasing cost
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
            
            print(f"🔍 Starting Uniform-Cost Search from {len(starting_node_ids)} initial nodes")
            print(f"📊 Parameters: max_total_nodes={self.max_total_nodes}")
            
            # Perform UCS traversal
            additional_nodes = self._perform_ucs(starting_node_ids)
            
            if additional_nodes:
                print(f"✅ Retrieved {len(additional_nodes)} additional context nodes via UCS")
                
                # Filter embedding properties from results
                filtered_items = self._filter_properties(additional_nodes)
                
                return filtered_items
            else:
                print("No additional context found through UCS traversal")
                return []
                
        except Exception as e:
            print(f"⚠️ Error in UCS traversal: {str(e)}")
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
    
    def _perform_ucs(self, starting_node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Perform Uniform Cost Search (UCS) traversal starting from the given node IDs.
        
        Args:
            starting_node_ids: List of node IDs to start traversal from
            
        Returns:
            List of discovered nodes with metadata
        """
        # Initialize UCS data structures
        visited: Set[str] = set()
        priority_queue: List[Tuple[float, str, int]] = []  # (cost, node_id, level)
        discovered_nodes: List[Dict[str, Any]] = []
        best_cost: Dict[str, float] = {}
        node_metadata: Dict[str, Dict[str, Any]] = {}  # Store metadata for nodes
        
        # Initialize priority queue with starting nodes (cost 0, level 0)
        for node_id in starting_node_ids:
            heapq.heappush(priority_queue, (0.0, node_id, 0))
            best_cost[node_id] = 0.0
            # Store metadata for starting nodes
            node_metadata[node_id] = {
                'node_id': node_id,
                'total_cost': 0.0,
                'edge_cost': 0.0,
                'discovered_from_node_id': None,
                'related_to_node_name': 'Starting Node',
                'level': 0
            }
        
        print(f"🚀 Starting UCS traversal from {len(starting_node_ids)} nodes")
        
        # Get embeddings for all starting nodes
        starting_embeddings = self._get_node_embeddings_batch(starting_node_ids)
        
        # UCS traversal
        while priority_queue and len(discovered_nodes) < self.max_total_nodes:
            current_cost, current_node_id, current_level = heapq.heappop(priority_queue)
            
            # Skip if already visited with better cost
            if current_node_id in visited:
                continue
            
            # Mark as visited
            visited.add(current_node_id)
            
            # Add current node to discovered nodes (only when actually visited)
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
                
                # Skip MENTIONS relationships only if we're beyond the initial level (current_level > 0)
                # This allows traversal from initial documents (which connect via MENTIONS)
                # but prevents further MENTIONS propagation at deeper levels
                if neighbor.get('relationship_type') == 'MENTIONS' and self.remove_mentions_nodes and current_level > 0:
                    continue
                
                # Skip if already visited
                if neighbor_id in visited:
                    continue
                
                # Get neighbor embedding
                neighbor_embedding = neighbor_embeddings.get(neighbor_id)
                if neighbor_embedding is None:
                    print(f"⚠️ Skipping neighbor {neighbor_id[:8]}... - no embedding found")
                    continue
                
                # Calculate edge cost using cosine distance
                edge_cost = self._calculate_cosine_distance(current_embedding, neighbor_embedding)
                new_cost = current_cost + edge_cost
                
                # Update if this is a better path to the neighbor
                if new_cost < best_cost.get(neighbor_id, float('inf')):
                    best_cost[neighbor_id] = new_cost
                    heapq.heappush(priority_queue, (new_cost, neighbor_id, current_level + 1))
                    
                    # Store neighbor metadata (but don't add to discovered_nodes yet)
                    neighbor_info = neighbor.copy()
                    neighbor_info['total_cost'] = new_cost
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
            if len(discovered_nodes) % 100 == 0 and len(discovered_nodes) > 0:
                print(f"📊 Discovered {len(discovered_nodes)} nodes, current cost: {current_cost:.4f}")
        
        # Sort discovered nodes by total cost (UCS guarantee)
        discovered_nodes.sort(key=lambda x: x['total_cost'])
        
        print(f"🎯 UCS traversal completed: {len(discovered_nodes)} nodes discovered")
        return discovered_nodes[:self.max_total_nodes]
    
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
            "method": "uniform_cost_search",
            "max_total_nodes": self.max_total_nodes,
            "remove_mentions_nodes": self.remove_mentions_nodes,
            "rel_type_filter": self.rel_type_filter
        }
