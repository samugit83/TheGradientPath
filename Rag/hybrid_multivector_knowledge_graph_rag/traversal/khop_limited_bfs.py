import os
from typing import List, Dict, Any, Optional, Set
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from collections import deque

class KhopLimitedBFS:
    """
    A graph traversal class that implements k-hop limited Breadth-First Search (BFS).
    This approach expands the frontier level by level, but stops after k hops (often 1-3)
    and caps the number of neighbors per node to avoid combinatorial blow-ups.
    """
    
    def __init__(
        self, 
        max_hops: int = 3, 
        max_neighbors_per_node: int = 10, 
        max_total_nodes: int = 100, 
        remove_mentions_nodes: bool = True,
        rel_type_filter: Optional[List[str]] = None
    ):
        """
        Initialize the k-hop limited BFS traversal.
        
        Args:
            max_hops: Maximum number of hops to traverse (default: 3)
            max_neighbors_per_node: Maximum number of neighbors to consider per node (default: 10)
            max_total_nodes: Maximum total nodes to return (default: 100)
            remove_mentions_nodes: If True, filter out nodes discovered through MENTIONS relationships (default: True)
            rel_type_filter: List of relationship types to filter by (default: None = no filtering)
        """
        self.max_hops = max_hops # Maximum number of hops to traverse (default: 3)
        self.max_neighbors_per_node = max_neighbors_per_node # Maximum number of neighbors to consider per node (default: 10)
        self.max_total_nodes = max_total_nodes # Maximum total nodes to return (default: 100)
        self.remove_mentions_nodes = remove_mentions_nodes # Filter out MENTIONS nodes
        self.rel_type_filter = rel_type_filter # List of relationship types to filter by
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
        Retrieve additional context through k-hop limited BFS traversal.
        
        This method:
        1. Extracts node IDs from initial documents
        2. Performs k-hop limited BFS to find connected nodes
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
            
            print(f"🔍 Starting k-hop limited BFS from {len(starting_node_ids)} initial nodes")
            print(f"📊 Parameters: max_hops={self.max_hops}, max_neighbors_per_node={self.max_neighbors_per_node}")
            
            # Perform k-hop limited BFS traversal
            additional_nodes = self._perform_khop_bfs(starting_node_ids)
            
            if additional_nodes:
                print(f"✅ Retrieved {len(additional_nodes)} additional context nodes via k-hop BFS")
                
                # Filter embedding properties from results
                filtered_items = self._filter_properties(additional_nodes)
                
                return filtered_items
            else:
                print("No additional context found through k-hop BFS traversal")
                return []
                
        except Exception as e:
            print(f"⚠️ Error in k-hop BFS traversal: {str(e)}")
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
    
    def _perform_khop_bfs(self, starting_node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Perform k-hop limited BFS traversal from starting nodes.
        
        Args:
            starting_node_ids: List of node IDs to start traversal from
            
        Returns:
            List of discovered nodes with metadata
        """
        # Initialize BFS data structures
        visited: Set[str] = set()
        queue: deque = deque()
        discovered_nodes: List[Dict[str, Any]] = []
        
        # Initialize queue with starting nodes (hop 0)
        for node_id in starting_node_ids:
            queue.append((node_id, 0))  # (node_id, hop_level)
            visited.add(node_id)
        
        print(f"🚀 Starting BFS traversal from {len(starting_node_ids)} nodes")
        
        # BFS traversal
        while queue and len(discovered_nodes) < self.max_total_nodes:
            current_node_id, hop_level = queue.popleft()
            
            # Skip if we've reached maximum hops
            if hop_level >= self.max_hops:
                continue
            
            # Get current node properties to extract the "id" property
            current_node_name = self._get_node_name_id_property(current_node_id)
            
            # Get neighbors of current node
            neighbors = self._get_node_neighbors(current_node_id, hop_level)
            
            # Limit neighbors per node to avoid combinatorial explosion
            neighbors = neighbors[:self.max_neighbors_per_node]
            
            # Add neighbors to queue for next hop
            for neighbor in neighbors:
                neighbor_id = neighbor['node_id']
                
                # Skip MENTIONS relationships only if we're beyond the initial level (hop_level > 0)
                # This allows traversal from initial documents (which connect via MENTIONS)
                # but prevents further MENTIONS propagation at deeper levels
                if neighbor.get('relationship_type') == 'MENTIONS' and self.remove_mentions_nodes and hop_level > 0:
                    continue
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, hop_level + 1))
                    
                    # Add neighbor to discovered nodes
                    neighbor['hop_level'] = hop_level + 1
                    neighbor['discovered_from_node_id'] = current_node_id
                    
                    # Set related_to_node_name conditionally based on relationship type
                    if neighbor.get('relationship_type') == 'MENTIONS':
                        neighbor['related_to_node_name'] = 'Document'
                    else:
                        neighbor['related_to_node_name'] = current_node_name
                    
                    # Only add to discovered nodes if it's not a MENTIONS relationship or if we allow MENTIONS
                    if neighbor.get('relationship_type') != 'MENTIONS' or not self.remove_mentions_nodes:
                        discovered_nodes.append(neighbor)
                    
                    # Check if we've reached the limit
                    if len(discovered_nodes) >= self.max_total_nodes:
                        break
            
            # Log progress periodically
            if len(discovered_nodes) % 100 == 0 and len(discovered_nodes) > 0:
                print(f"📊 Discovered {len(discovered_nodes)} nodes, current hop level: {hop_level}")
        
        print(f"🎯 BFS traversal completed: {len(discovered_nodes)} nodes discovered")
        return discovered_nodes
    
    def _get_node_neighbors(self, node_id: str, hop_level: int = 0) -> List[Dict[str, Any]]:
        """
        Get neighbors of a node (both incoming and outgoing relationships).
        
        Args:
            node_id: ID of the node to get neighbors for
            hop_level: Current hop level (0 = first hop, 1+ = subsequent hops)
            
        Returns:
            List of neighbor nodes with metadata
        """
        try:
            # Apply relationship filtering only for hop levels > 0
            use_filtering = self.rel_type_filter and hop_level > 0
            
            # Query to get both incoming and outgoing neighbors
            # We use a bidirectional approach to capture the full neighborhood
            if use_filtering:

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
                    'khop_bfs_neighbor' as search_type
                LIMIT $limit
                """
                
                result = self.graph.query(cypher_query, {
                    "node_id": node_id,
                    "rel_types": self.rel_type_filter,
                    "limit": self.max_neighbors_per_node
                })
                
                print(f"📊 Hop {hop_level}: Found {len(result)} neighbors for node {node_id[:8]}... using filtered relationships")
                
            else:
                # No filtering - use original query (for hop 0 or when no filter is set)
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
                    'khop_bfs_neighbor' as search_type
                LIMIT $limit
                """
                
                result = self.graph.query(cypher_query, {
                    "node_id": node_id,
                    "limit": self.max_neighbors_per_node
                })
                
                filter_reason = "first hop (no filtering)" if hop_level == 0 else "no filter set"
                print(f"📊 Hop {hop_level}: Found {len(result)} neighbors for node {node_id[:8]}... using all relationships ({filter_reason})")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error getting neighbors for node {node_id}: {e}")
            return []
    
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
            "method": "k-hop_limited_bfs",
            "max_hops": self.max_hops,
            "max_neighbors_per_node": self.max_neighbors_per_node,
            "max_total_nodes": self.max_total_nodes,
            "remove_mentions_nodes": self.remove_mentions_nodes,
            "rel_type_filter": self.rel_type_filter
        } 