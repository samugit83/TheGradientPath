import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from params import model
from prompts import CONTEXT_TO_CYPHER_SYSTEM_PROMPT, CONTEXT_TO_CYPHER_HUMAN_PROMPT
import logging

logger = logging.getLogger(__name__)

class ContextToCypher:
    """
    A class that analyzes relevant documents and user queries to generate
    Cypher queries for additional graph traversal when initial context is insufficient.
    """
    
    def __init__(self):
        """Initialize the ContextToCypher with LLM and Neo4j connection."""
        self._setup_environment()
        self._initialize_llm()
        self._initialize_neo4j_connection()
        self._initialize_prompt_template()
    
    def _setup_environment(self) -> None:
        """Load environment variables and validate required credentials."""
        load_dotenv()
        
        # OpenAI API Key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Neo4j Credentials validation
        required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def _initialize_llm(self) -> None:
        """Initialize the OpenAI LLM for Cypher generation."""
        self.llm = ChatOpenAI(
            model=model
        )
    
    def _initialize_neo4j_connection(self) -> None:
        """Initialize Neo4j graph connection."""
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            refresh_schema=False,
        )
    
    def _initialize_prompt_template(self) -> None:
        """Initialize the prompt template for Cypher generation."""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", CONTEXT_TO_CYPHER_SYSTEM_PROMPT),
            ("human", CONTEXT_TO_CYPHER_HUMAN_PROMPT)
        ])
    
    def _sanitize_schema_for_template(self, schema_str: str) -> str:
        """Sanitize schema string to avoid template formatting issues."""
        # Replace curly braces to avoid template formatting conflicts
        sanitized = schema_str.replace('{', '{{').replace('}', '}}')
        return sanitized
    
    def _get_graph_schema(self) -> str:
        """Retrieve comprehensive Neo4j graph schema including vector indexes."""
        try:
            # Get basic schema information
            basic_schema = self._get_basic_graph_structure()
            
            # Get detailed vector index information
            vector_indexes_info = self._get_detailed_vector_indexes()
            
            # Combine all schema information - using regular string concatenation to avoid f-string issues with Cypher curly braces
            comprehensive_schema = """
=== NEO4J GRAPH SCHEMA ===

""" + basic_schema + """

=== VECTOR INDEXES FOR SIMILARITY SEARCH ===

""" + vector_indexes_info + """

=== INDEX NAMING CONVENTIONS ===

Vector indexes follow these naming patterns:
1. NODE INDEXES: "embedding_property_label_index"
   - Example: "embedding_text_document_index" = searches embedding_text property on Document nodes
   - Example: "embedding_birth_date_person_index" = searches embedding_birth_date property on Person nodes

2. RELATIONSHIP INDEXES: "embedding_property_type_rel_index" 
   - Example: "embedding_description_knows_rel_index" = searches embedding_description property on KNOWS relationships
   - Example: "embedding_description_worked_at_rel_index" = searches embedding_description property on WORKED_AT relationships

3. DOCUMENT-SPECIFIC INDEXES: "embedding_property_document_index"
   - Example: "embedding_hyp_queries_document_index" = searches embedding_hyp_queries property on Document nodes
   - Example: "embedding_italian_translation_document_index" = searches embedding_italian_translation property on Document nodes

=== CYPHER QUERY PATTERNS FOR VECTOR SEARCH ===

⚠️  IMPORTANT: The patterns below are EXAMPLES to demonstrate techniques and possibilities. 
The actual Cypher query you generate must be dynamically created by analyzing:
- The specific GRAPH STRUCTURE & AVAILABLE RESOURCES provided in the schema
- The content and gaps in the initial documents
- The specific user query requirements
- The available vector indexes and their properties
- The actual node labels and relationship types in the database

Do NOT copy these examples directly - use them as inspiration to craft a custom query 
that fits the specific context, schema, and information needs of the current request.

Use these patterns in your Cypher queries:

1. BASIC VECTOR SIMILARITY SEARCH ON NODES:
   CALL db.index.vector.queryNodes('index_name', top_k, $query_vector)
   YIELD node, score
   RETURN elementId(node) as node_id, 
          score, 
          'vector_node_search' as search_type,
          properties(node) as properties,
          labels(node) as labels

2. BASIC VECTOR SIMILARITY SEARCH ON RELATIONSHIPS:
   CALL db.index.vector.queryRelationships('index_name', top_k, $query_vector)
   YIELD relationship, score
   RETURN elementId(relationship) as node_id, 
          score, 
          'vector_relationship_search' as search_type,
          properties(relationship) as properties,
          type(relationship) as relationship_type

3. MULTI-INDEX VECTOR SEARCH (Query multiple vector indexes simultaneously):
   CALL db.index.vector.queryNodes('index1', 5, $query_vector) YIELD node as n1, score as s1
   WITH collect({node: n1, score: s1, index: 'index1'}) as results1
   CALL db.index.vector.queryNodes('index2', 5, $query_vector) YIELD node as n2, score as s2
   WITH results1 + collect({node: n2, score: s2, index: 'index2'}) as all_results
   UNWIND all_results as result
   RETURN elementId(result.node) as node_id,
          result.score as score,
          'multi_index_vector_search' as search_type,
          result.index as source_index,
          properties(result.node) as properties,
          labels(result.node) as labels

4. VECTOR SEARCH WITH GRAPH EXPANSION (Find similar nodes then expand to neighbors):
   CALL db.index.vector.queryNodes('index_name', 10, $query_vector)
   YIELD node as seed, score as seed_score
   MATCH (seed)-[r]-(neighbor)
   RETURN elementId(neighbor) as node_id,
          seed_score * 0.8 as score,
          'vector_expansion' as search_type,
          type(r) as relationship_type,
          properties(neighbor) as properties,
          labels(neighbor) as labels,
          properties(seed) as seed_properties,
          labels(seed) as seed_labels,
          elementId(seed) as seed_node_id

5. BIDIRECTIONAL VECTOR EXPANSION (Expand in both directions):
   CALL db.index.vector.queryNodes('index_name', 5, $query_vector)
   YIELD node as center, score
   MATCH (left)-[r1]-(center)-[r2]-(right)
   WHERE elementId(left) <> elementId(right)
   RETURN elementId(left) as node_id,
          score * 0.7 as score,
          'bidirectional_expansion' as search_type,
          type(r1) as relationship_type,
          properties(left) as properties,
          labels(left) as labels,
          elementId(center) as center_node_id,
          properties(center) as center_properties
   UNION
   RETURN elementId(right) as node_id,
          score * 0.7 as score,
          'bidirectional_expansion' as search_type,
          type(r2) as relationship_type,
          properties(right) as properties,
          labels(right) as labels,
          elementId(center) as center_node_id,
          properties(center) as center_properties

6. MULTI-HOP VECTOR TRAVERSAL (Find similar nodes then traverse multiple hops):
   CALL db.index.vector.queryNodes('index_name', 5, $query_vector)
   YIELD node as start, score as initial_score
   MATCH path = (start)-[*1..3]-(end)
   WHERE elementId(start) <> elementId(end)
   RETURN elementId(end) as node_id,
          initial_score / length(path) as score,
          'multi_hop_traversal' as search_type,
          length(path) as hop_distance,
          properties(end) as properties,
          labels(end) as labels,
          [rel in relationships(path) | type(rel)] as relationship_path,
          elementId(start) as start_node_id,
          properties(start) as start_properties

7. PATTERN-BASED VECTOR SEARCH (Find specific graph patterns around similar nodes):
   CALL db.index.vector.queryNodes('index_name', 8, $query_vector)
   YIELD node as doc, score
   MATCH (doc)-[r1:MENTIONS]->(entity)-[r2:RELATED_TO]->(related_entity)
   RETURN elementId(related_entity) as node_id,
          score * 0.9 as score,
          'pattern_based_search' as search_type,
          properties(related_entity) as properties,
          labels(related_entity) as labels,
          properties(entity) as intermediate_entity,
          type(r1) as first_relationship_type,
          type(r2) as second_relationship_type,
          elementId(doc) as source_node_id

8. AGGREGATED VECTOR SEARCH (Group results by relationship types):
   CALL db.index.vector.queryNodes('index_name', 15, $query_vector)
   YIELD node as source, score
   MATCH (source)-[r]->(target)
   WITH type(r) as rel_type, collect({target: target, score: score, source: source}) as targets
   UNWIND targets as t
   RETURN elementId(t.target) as node_id,
          avg(t.score) as score,
          'aggregated_vector_search' as search_type,
          rel_type as relationship_type,
          properties(t.target) as properties,
          labels(t.target) as labels,
          elementId(t.source) as source_node_id,
          properties(t.source) as source_properties

9. HYBRID VECTOR-PROPERTY SEARCH (Combine vector similarity with property filtering):
   CALL db.index.vector.queryNodes('index_name', 20, $query_vector)
   YIELD node, score
   WHERE any(prop in keys(properties(node)) WHERE properties(node)[prop] =~ '(?i).*search_term.*')
   RETURN elementId(node) as node_id,
          score + 0.1 as score,
          'hybrid_vector_property' as search_type,
          properties(node) as properties,
          labels(node) as labels

10. TEMPORAL VECTOR SEARCH (Vector search with temporal/sequential relationships):
    CALL db.index.vector.queryNodes('index_name', 10, $query_vector)
    YIELD node as current, score
    MATCH (previous)-[r1:PRECEDES|FOLLOWS|NEXT|AFTER*1..2]-(current)
    RETURN elementId(previous) as node_id,
           score * 0.85 as score,
           'temporal_vector_search' as search_type,
           properties(previous) as properties,
           labels(previous) as labels,
           [rel in r1 | type(rel)] as relationship_path,
           elementId(current) as current_node_id
    UNION
    MATCH (current)-[r2:PRECEDES|FOLLOWS|NEXT|AFTER*1..2]-(next)
    RETURN elementId(next) as node_id,
           score * 0.85 as score,
           'temporal_vector_search' as search_type,
           properties(next) as properties,
           labels(next) as labels,
           [rel in r2 | type(rel)] as relationship_path,
           elementId(current) as current_node_id

11. FAMILY RELATIONSHIP DISCOVERY (Specifically for family/relationship queries):
    CALL db.index.vector.queryNodes('index_name', 10, $query_vector)
    YIELD node as person, score
    MATCH (person)-[r]-(family_member)
    WHERE type(r) IN ['FATHER_OF', 'MOTHER_OF', 'CHILD_OF', 'SIBLING_OF', 'SPOUSE_OF', 'PARENT_OF', 'SON_OF', 'DAUGHTER_OF', 'MARRIED_TO', 'FAMILY_OF']
    RETURN elementId(family_member) as node_id,
           score * 0.9 as score,
           'family_relationship_search' as search_type,
           type(r) as relationship_type,
           properties(family_member) as properties,
           labels(family_member) as labels,
           elementId(person) as central_person_id,
           properties(person) as central_person_properties

12. RELATIONSHIP-FOCUSED PATTERN SEARCH (Find all connected nodes with relationship details):
    CALL db.index.vector.queryNodes('index_name', 8, $query_vector)
    YIELD node as center, score
    MATCH (center)-[r]-(connected)
    RETURN elementId(connected) as node_id,
           score * 0.8 as score,
           'relationship_focused_search' as search_type,
           type(r) as relationship_type,
           properties(connected) as properties,
           labels(connected) as labels,
           elementId(center) as center_node_id,
           properties(center) as center_properties,
           labels(center) as center_labels,
           properties(r) as relationship_properties

13. EXTENDED FAMILY NETWORK SEARCH (Multi-hop family relationships):
    CALL db.index.vector.queryNodes('index_name', 5, $query_vector)
    YIELD node as center, score
    MATCH path = (center)-[r*1..3]-(family_member)
    WHERE all(rel in relationships(path) WHERE type(rel) IN ['FATHER_OF', 'MOTHER_OF', 'CHILD_OF', 'SIBLING_OF', 'SPOUSE_OF', 'PARENT_OF', 'SON_OF', 'DAUGHTER_OF', 'MARRIED_TO', 'FAMILY_OF'])
    RETURN elementId(family_member) as node_id,
           score / length(path) as score,
           'extended_family_search' as search_type,
           [rel in relationships(path) | type(rel)] as relationship_path,
           length(path) as relationship_distance,
           properties(family_member) as properties,
           labels(family_member) as labels,
           elementId(center) as center_node_id,
           properties(center) as center_properties

14. BIOGRAPHICAL RELATIONSHIP SEARCH (For biographical queries):
    CALL db.index.vector.queryNodes('index_name', 12, $query_vector)
    YIELD node as person, score
    MATCH (person)-[r]-(related)
    WHERE type(r) IN ['LIVED_WITH', 'WORKED_WITH', 'STUDIED_WITH', 'COLLABORATED_WITH', 'KNEW', 'FRIEND_OF', 'COLLEAGUE_OF', 'TAUGHT_BY', 'STUDENT_OF']
    RETURN elementId(related) as node_id,
           score * 0.85 as score,
           'biographical_relationship_search' as search_type,
           type(r) as relationship_type,
           properties(related) as properties,
           labels(related) as labels,
           elementId(person) as person_node_id,
           properties(person) as person_properties,
           properties(r) as relationship_properties

15. COMMUNITY DETECTION VECTOR SEARCH (Find nodes in same communities as similar nodes):
    CALL db.index.vector.queryNodes('index_name', 8, $query_vector)
    YIELD node as seed, score
    MATCH (seed)-[r*2..4]-(community_member)
    WHERE elementId(seed) <> elementId(community_member)
    WITH community_member, score, count(*) as connection_strength, collect(distinct [rel in r | type(rel)]) as paths
    WHERE connection_strength >= 2
    RETURN elementId(community_member) as node_id,
           score * (connection_strength * 0.1) as score,
           'community_detection' as search_type,
           connection_strength,
           paths as relationship_paths,
           properties(community_member) as properties,
           labels(community_member) as labels

16. CROSS-DOMAIN VECTOR SEARCH (Bridge different node types through relationships):
    CALL db.index.vector.queryNodes('index_name', 10, $query_vector)
    YIELD node as source, score
    MATCH (source)-[r1]->(bridge)-[r2]->(target)
    WHERE labels(source) <> labels(target)
    RETURN elementId(target) as node_id,
           score * 0.8 as score,
           'cross_domain_search' as search_type,
           labels(source) as source_labels,
           labels(target) as target_labels,
           type(r1) as first_relationship_type,
           type(r2) as second_relationship_type,
           properties(target) as properties,
           labels(target) as labels,
           elementId(source) as source_node_id,
           properties(bridge) as bridge_properties

17. WEIGHTED RELATIONSHIP VECTOR SEARCH (Use relationship weights/properties for scoring):
    CALL db.index.vector.queryNodes('index_name', 12, $query_vector)
    YIELD node as start, score as base_score
    MATCH (start)-[r]->(end)
    WITH end, base_score, r,
         CASE WHEN 'weight' in keys(properties(r)) THEN properties(r).weight ELSE 1.0 END as weight
    RETURN elementId(end) as node_id,
           base_score * weight as score,
           'weighted_relationship_search' as search_type,
           type(r) as relationship_type,
           weight,
           properties(end) as properties,
           labels(end) as labels,
           properties(r) as relationship_properties,
           elementId(start) as start_node_id

18. SIMILARITY CLUSTERING VECTOR SEARCH (Group similar results and expand clusters):
    CALL db.index.vector.queryNodes('index_name', 15, $query_vector)
    YIELD node, score
    WITH collect({node: node, score: score}) as similar_nodes
    UNWIND similar_nodes as sn
    MATCH (sn.node)-[r]-(cluster_member)
    WHERE NOT cluster_member IN [n.node for n in similar_nodes]
    RETURN elementId(cluster_member) as node_id,
           avg(sn.score) * 0.7 as score,
           'similarity_clustering' as search_type,
           type(r) as relationship_type,
           properties(cluster_member) as properties,
           labels(cluster_member) as labels,
           elementId(sn.node) as cluster_center_id

19. DYNAMIC PROPERTY VECTOR SEARCH (Flexible property matching without assuming specific properties):
    CALL db.index.vector.queryNodes('index_name', 10, $query_vector)
    YIELD node as similar, score
    MATCH (similar)-[r]-(related)
    WITH related, score, properties(related) as props, r
    WHERE size(keys(props)) > 0
    RETURN elementId(related) as node_id,
           score * 0.9 as score,
           'dynamic_property_search' as search_type,
           type(r) as relationship_type,
           keys(props) as available_properties,
           props as properties,
           labels(related) as labels,
           elementId(similar) as similar_node_id

20. GRAPH CENTRALITY VECTOR SEARCH (Find central nodes in neighborhoods of similar nodes):
    CALL db.index.vector.queryNodes('index_name', 8, $query_vector)
    YIELD node as seed, score
    MATCH (seed)-[r*1..2]-(neighbor)
    WITH neighbor, score, count(*) as centrality_score, collect(distinct [rel in r | type(rel)]) as paths
    WHERE centrality_score >= 3
    ORDER BY centrality_score DESC
    RETURN elementId(neighbor) as node_id,
           score * (centrality_score * 0.1) as score,
           'centrality_based_search' as search_type,
           centrality_score,
           paths as relationship_paths,
           properties(neighbor) as properties,
           labels(neighbor) as labels

21. ADAPTIVE MULTI-STRATEGY UNION SEARCH (Combine multiple techniques dynamically):
    CALL db.index.vector.queryNodes('index_name', 5, $query_vector)
    YIELD node as base, score as base_score
    MATCH (base)-[r1]-(level1)-[r2]-(level2)
    RETURN elementId(level2) as node_id,
           base_score * 0.6 as score,
           'multi_hop_union' as search_type,
           [type(r1), type(r2)] as relationship_path,
           properties(level2) as properties,
           labels(level2) as labels,
           elementId(base) as base_node_id
    UNION
    CALL db.index.vector.queryNodes('index_name', 5, $query_vector)
    YIELD node as center, score as center_score
    MATCH (left)-[r1]-(center)-[r2]-(right)
    WHERE elementId(left) <> elementId(right)
    RETURN elementId(left) as node_id,
           center_score * 0.8 as score,
           'bidirectional_union' as search_type,
           type(r1) as relationship_type,
           properties(left) as properties,
           labels(left) as labels,
           elementId(center) as center_node_id
    UNION
    CALL db.index.vector.queryNodes('index_name', 5, $query_vector)
    YIELD node as hub, score as hub_score
    MATCH (hub)-[r]-(connected)
    WITH connected, hub_score, count(r) as connections, collect(distinct type(r)) as relationship_types
    WHERE connections >= 2
    RETURN elementId(connected) as node_id,
           hub_score * (connections * 0.1) as score,
           'hub_based_union' as search_type,
           relationship_types,
           properties(connected) as properties,
           labels(connected) as labels,
           elementId(hub) as hub_node_id

22. SEMANTIC RELATIONSHIP DISCOVERY (Find semantically related nodes through any relationship):
    CALL db.index.vector.queryNodes('index_name', 12, $query_vector)
    YIELD node as semantic_seed, score
    MATCH (semantic_seed)-[r*1..3]-(semantic_target)
    WHERE elementId(semantic_seed) <> elementId(semantic_target)
    WITH semantic_target, score, 
         [rel in relationships(r) | type(rel)] as relationship_path,
         length(r) as path_length
    RETURN elementId(semantic_target) as node_id,
           score / path_length as score,
           'semantic_relationship_discovery' as search_type,
           relationship_path,
           path_length,
           properties(semantic_target) as properties,
           labels(semantic_target) as labels,
           elementId(semantic_seed) as seed_node_id

23. CONTEXTUAL PROPERTY EXPANSION (Expand based on shared property patterns):
    CALL db.index.vector.queryNodes('index_name', 10, $query_vector)
    YIELD node as context_node, score
    WITH context_node, score, keys(properties(context_node)) as context_props
    MATCH (context_node)-[r]-(similar_node)
    WHERE elementId(similar_node) <> elementId(context_node)
      AND any(prop in context_props WHERE prop in keys(properties(similar_node)))
    WITH similar_node, score, r,
         [prop in context_props WHERE prop in keys(properties(similar_node))] as shared_props
    WHERE size(shared_props) >= 1
    RETURN elementId(similar_node) as node_id,
           score * (size(shared_props) * 0.2) as score,
           'contextual_property_expansion' as search_type,
           shared_props,
           type(r) as relationship_type,
           properties(similar_node) as properties,
           labels(similar_node) as labels,
           elementId(context_node) as context_node_id

24. RELATIONSHIP-TYPE CLUSTERING SEARCH (Group by relationship types and expand clusters):
    CALL db.index.vector.queryNodes('index_name', 8, $query_vector)
    YIELD node as anchor, score
    MATCH (anchor)-[r]->(target)
    WITH type(r) as rel_type, collect({target: target, score: score, anchor: anchor}) as type_cluster
    WHERE size(type_cluster) >= 2
    UNWIND type_cluster as cluster_item
    MATCH (cluster_item.target)-[r2]-(extended)
    WHERE type(r2) = rel_type AND elementId(extended) <> elementId(cluster_item.target)
    RETURN elementId(extended) as node_id,
           avg(cluster_item.score) * 0.7 as score,
           'relationship_type_clustering' as search_type,
           rel_type as cluster_relationship_type,
           type(r2) as extended_relationship_type,
           properties(extended) as properties,
           labels(extended) as labels,
           elementId(cluster_item.anchor) as anchor_node_id
            """
            
            return comprehensive_schema.strip()
                
        except Exception as e:
            print(f"⚠️  Warning: Could not retrieve comprehensive graph schema: {e}")
            return self._generate_basic_schema()
    
    def _get_basic_graph_structure(self) -> str:
        """Get basic graph structure information."""
        try:
            # Get node labels and their counts
            node_info_result = self.graph.query("""
                MATCH (n) 
                RETURN labels(n) as node_labels, count(n) as count
                ORDER BY count DESC
                LIMIT 100
            """)
            
            # Get relationship types and their counts
            rel_info_result = self.graph.query("""
                MATCH ()-[r]->() 
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
                LIMIT 100
            """)
            
            # Process node information
            node_summary = []
            all_labels = set()
            for result in node_info_result:
                labels = result.get('node_labels', [])
                count = result.get('count', 0)
                all_labels.update(labels)
                label_str = ':'.join(labels) if labels else 'Unknown'
                node_summary.append(f"  - {label_str}: {count} nodes")
            
            # Process relationship information
            rel_summary = []
            for result in rel_info_result:
                rel_type = result.get('rel_type', 'Unknown')
                count = result.get('count', 0)
                rel_summary.append(f"  - {rel_type}: {count} relationships")
            
            basic_structure = f"""
NODE LABELS AND COUNTS:
{chr(10).join(node_summary)}

RELATIONSHIP TYPES AND COUNTS:
{chr(10).join(rel_summary)}

UNIQUE NODE LABELS: {', '.join(sorted(all_labels))}
            """
            
            return basic_structure.strip()
            
        except Exception as e:
            print(f"⚠️  Warning: Could not get basic graph structure: {e}")
            return "Basic graph structure unavailable"
    
    def _get_detailed_vector_indexes(self) -> str:
        """Get detailed information about all vector indexes."""
        try:
            # Get all vector indexes with detailed information
            indexes_result = self.graph.query("""
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties, state, populationPercent
                WHERE type = 'VECTOR'
                RETURN name, labelsOrTypes, properties, state, populationPercent
                ORDER BY name
            """)
            
            if not indexes_result:
                return "No vector indexes found in the database."
            
            # Filter out indexes that contain "__entity__" in their name
            indexes_result = [
                index_info for index_info in indexes_result 
                if "__entity__" not in index_info.get('name', '')
            ]
            
            if not indexes_result:
                return "No vector indexes found after filtering (all contained '__entity__')."
            
            # Group indexes by type (node vs relationship)
            node_indexes = []
            relationship_indexes = []
            document_indexes = []
            
            for index_info in indexes_result:
                name = index_info.get('name', 'Unknown')
                labels_or_types = index_info.get('labelsOrTypes', [])
                properties = index_info.get('properties', [])
                state = index_info.get('state', 'Unknown')
                population = index_info.get('populationPercent', 0)
                
                # Determine index type and categorize
                index_description = self._describe_vector_index(name, labels_or_types, properties, state, population)
                
                if '_rel_index' in name:
                    relationship_indexes.append(index_description)
                elif 'document_index' in name:
                    document_indexes.append(index_description)
                else:
                    node_indexes.append(index_description)
            
            # Format the output
            vector_info = []
            
            if node_indexes:
                vector_info.append("NODE VECTOR INDEXES:")
                vector_info.extend(node_indexes)
                vector_info.append("")
            
            if relationship_indexes:
                vector_info.append("RELATIONSHIP VECTOR INDEXES:")
                vector_info.extend(relationship_indexes)
                vector_info.append("")
            
            if document_indexes:
                vector_info.append("DOCUMENT-SPECIFIC VECTOR INDEXES:")
                vector_info.extend(document_indexes)
            
            return "\n".join(vector_info)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not get vector indexes information: {e}")
            return "Vector indexes information unavailable"
    
    def _describe_vector_index(self, name: str, labels_or_types: List[str], properties: List[str], state: str, population: float) -> str:
        """Create a detailed description of a vector index."""

        # Determine what this index searches
        if '_rel_index' in name:
            # Relationship index
            rel_type = labels_or_types[0] if labels_or_types else 'Unknown'
            property_name = properties[0] if properties else 'Unknown'
            base_property = property_name.replace('embedding_', '') if property_name.startswith('embedding_') else property_name
            
            description = f"""  • {name}
    Purpose: Vector similarity search on {rel_type} relationships
    Property: {property_name} (embeddings of '{base_property}' text)
    Usage: Find relationships with similar {base_property} content
    State: {state} ({population:.1f}% populated)
    Cypher: CALL db.index.vector.queryRelationships('{name}', top_k, $vector)"""
        
        elif 'document_index' in name:
            # Document-specific index
            property_name = properties[0] if properties else 'Unknown'
            base_property = property_name.replace('embedding_', '') if property_name.startswith('embedding_') else property_name
            
            description = f"""  • {name}
    Purpose: Vector similarity search on Document nodes
    Property: {property_name} (embeddings of '{base_property}' content)
    Usage: Find documents with similar {base_property} content
    State: {state} ({population:.1f}% populated)
    Cypher: CALL db.index.vector.queryNodes('{name}', top_k, $vector)"""
        
        else:
            # Regular node index
            label = labels_or_types[0] if labels_or_types else 'Unknown'
            property_name = properties[0] if properties else 'Unknown'
            base_property = property_name.replace('embedding_', '') if property_name.startswith('embedding_') else property_name
            
            description = f"""  • {name}
    Purpose: Vector similarity search on {label} nodes
    Property: {property_name} (embeddings of '{base_property}' text)
    Usage: Find {label} nodes with similar {base_property} content
    State: {state} ({population:.1f}% populated)
    Cypher: CALL db.index.vector.queryNodes('{name}', top_k, $vector)"""
        
        return description
    
    def _generate_basic_schema(self) -> str:
        """Generate a basic schema description as fallback."""
        try:
            # Get node labels
            node_labels_result = self.graph.query("""
                MATCH (n) 
                RETURN DISTINCT labels(n) as node_labels 
                LIMIT 20
            """)
            
            # Get relationship types
            rel_types_result = self.graph.query("""
                MATCH ()-[r]->() 
                RETURN DISTINCT type(r) as rel_type 
                LIMIT 20
            """)
            
            # Extract unique labels
            all_labels = set()
            for result in node_labels_result:
                labels = result.get('node_labels', [])
                all_labels.update(labels)
            
            # Extract relationship types
            rel_types = [result.get('rel_type', '') for result in rel_types_result]
            
            schema_description = f"""
Node Labels: {', '.join(sorted(all_labels))}
Relationship Types: {', '.join(sorted(rel_types))}

Common Patterns:
- Document nodes contain text content and are connected to entities via relationships
- Entity nodes (Person, Organization, Location, etc.) are mentioned in documents
- Use MENTIONED_IN or similar relationships to connect entities to documents
- Vector indexes may be available for similarity search on embedding properties
            """
            
            return schema_description.strip()
            
        except Exception as e:
            print(f"⚠️  Warning: Could not generate basic schema: {e}")
            return "Schema unavailable - use general graph traversal patterns"
    
    def _format_initial_documents(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Format the initial documents for the prompt."""
        if not relevant_docs:
            return "No initial documents retrieved."
        
        formatted_docs = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.get('source', 'Unknown')
            text = doc.get('text', '')
            score = doc.get('score', 0)
            
            # Truncate very long text
            if len(text) > 500:
                text = text[:500] + "..."
            
            formatted_docs.append(f"""
Document {i}:
Source: {source}
Similarity Score: {score:.4f}
Content: {text}
""")
        
        return "\n".join(formatted_docs)
    
    def generate_cypher_query(self, relevant_docs: List[Dict[str, Any]], user_query: str) -> Optional[str]:
        """
        Generate a Cypher query based on relevant documents and user query.
        
        Args:
            relevant_docs: List of initially retrieved relevant documents
            user_query: The user's original query
            
        Returns:
            Cypher query string if additional traversal is needed, None if sufficient context exists
        """
        try:
            # Get graph schema
            schema = self._get_graph_schema()
            
            # Format initial documents
            formatted_docs = self._format_initial_documents(relevant_docs)
            
            # Generate prompt
            messages = self.prompt_template.format_messages(
                schema=self._sanitize_schema_for_template(schema),
                initial_documents=formatted_docs,
                user_query=user_query
            )

            # print(f"🔍 Generated prompt: {messages}")
            
            # Get LLM response
            response = self.llm.invoke(messages)
            cypher_query = response.content.strip()
            
            # Clean markdown formatting from the response
            cypher_query = self._clean_cypher_response(cypher_query)
            
            # Debug: Show raw and cleaned responses
            print(f"🐛 Raw LLM response: {repr(response.content[:200])}...")
            print(f"🐛 Cleaned query: {repr(cypher_query[:200])}...")
            
            # Check if LLM determined sufficient context exists
            if cypher_query.lower() == "null":
                print("🔍 LLM determined initial documents are sufficient")
                return None
            
            print(f"🔍 Generated sophisticated Cypher query for additional traversal")
            print(f"Query: {cypher_query}")
            
            # Store the user query for vector embedding in execute method
            self._current_user_query = user_query
            
            return cypher_query
            
        except Exception as e:
            print(f"❌ Error generating Cypher query: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def execute_cypher_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute the generated Cypher query with vector parameters and return results.
        
        Args:
            cypher_query: The Cypher query to execute
            
        Returns:
            List of additional nodes found through graph traversal
        """
        try:
            print(f"🚀 Executing sophisticated Cypher query...")
            
            # Check if query contains vector search operations
            query_params = {}
            if '$query_vector' in cypher_query:
                # Generate query embedding for vector operations
                if hasattr(self, '_current_user_query'):
                    print("🔍 Creating query embedding for vector similarity search...")
                    from langchain_openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings()
                    query_vector = embeddings.embed_query(self._current_user_query)
                    query_params['query_vector'] = query_vector
                    print(f"✅ Query vector created with {len(query_vector)} dimensions")
                else:
                    print("⚠️  No user query available for vector embedding")
                    return []
            
            # Execute the query with parameters
            results = self.graph.query(cypher_query, query_params)
            
            # Add connections information to each node
            if results:
                results = self._add_connections_to_nodes(results)
            
            if results:
                print(f"✅ Found {len(results)} additional nodes through hybrid vector-graph traversal")
                
                # Display summary of search types used
                search_types = set()
                for result in results:
                    search_type = result.get('search_type', 'unknown')
                    search_types.add(search_type)
                
                if search_types:
                    print(f"🔍 Search strategies used: {', '.join(sorted(search_types))}")
                
                return results
            else:
                print("ℹ️  No additional nodes found through hybrid vector-graph traversal")
                return []
                
        except Exception as e:
            print(f"❌ Error executing hybrid Cypher query: {e}")
            print(f"Query was: {cypher_query}")
            return []
    
    def _add_connections_to_nodes(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add connections property to each node in the results.
        Connections format: "relationship_type + connected_node_id, relationship_type + connected_node_id, ..."
        
        Args:
            results: List of query results containing nodes
            
        Returns:
            List of results with connections property added to each node
        """
        if not results:
            return results
        
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Check if this result contains a node (has node_id)
            if 'node_id' in result:
                node_id = result['node_id']
                
                try:
                    # Get all relationships for this node
                    relationships_query = """
                    MATCH (n)-[r]-(connected)
                    WHERE elementId(n) = $node_id AND type(r) <> 'MENTIONS'
                    RETURN type(r) as relationship_type, 
                           COALESCE(connected.id, elementId(connected)) as connected_node_id
                    """
                    
                    relationships = self.graph.query(relationships_query, {'node_id': node_id})
                    
                    # Build connections string
                    connections_list = []
                    for rel in relationships:
                        rel_type = rel.get('relationship_type', '')
                        connected_id = rel.get('connected_node_id', '')
                        if rel_type and connected_id:
                            connections_list.append(f"{rel_type} {connected_id}")
                    
                    # Add connections property
                    connections_string = ", ".join(connections_list)
                    enhanced_result['connections'] = connections_string
                    
                    print(f"🔗 Added {len(connections_list)} connections for node {node_id}")
                    
                except Exception as e:
                    print(f"⚠️  Error getting connections for node {node_id}: {e}")
                    enhanced_result['connections'] = "Error retrieving connections"
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def filter_properties(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove properties that contain 'embedding' in their key names from nodes and relationships.
        
        Args:
            items: List of nodes/relationships from Cypher query results
            
        Returns:
            List of items with embedding properties removed
        """
        filtered_items = []
        
        for item in items:
            filtered_item = item.copy()
            
            # Remove embedding properties from the properties dict if it exists
            if 'properties' in filtered_item and isinstance(filtered_item['properties'], dict):
                filtered_item['properties'] = {
                    key: value for key, value in filtered_item['properties'].items()
                    if 'embedding' not in key.lower()
                }
            
            # Remove embedding properties that might be direct keys in the item
            keys_to_remove = [key for key in filtered_item.keys() if 'embedding' in key.lower()]
            for key in keys_to_remove:
                filtered_item.pop(key, None)
            
            filtered_items.append(filtered_item)
        
        return filtered_items

    def traverse_graph(self, relevant_docs: List[Dict[str, Any]], user_query: str) -> List[Dict[str, Any]]:
        """
        Retrieve additional context through hybrid vector-graph traversal.
        
        This method:
        1. Analyzes initial documents and user query
        2. Generates a Cypher query for additional context if needed
        3. Executes the query and returns additional nodes/relationships
        4. Filters out embedding properties from results
        
        Args:
            relevant_docs: List of initially retrieved documents
            user_query: The user's question
            
        Returns:
            List of additional context nodes/relationships with embedding properties removed
        """
        try:
            # Generate Cypher query for additional context
            cypher_query = self.generate_cypher_query(relevant_docs, user_query)
            
            if not cypher_query:
                print("No additional context needed - initial documents are sufficient")
                return []
            
            # Execute the query
            additional_nodes = self.execute_cypher_query(cypher_query)
            
            if additional_nodes:
                print(f"Retrieved {len(additional_nodes)} additional context items via hybrid search")
                
                # Filter embedding properties from both nodes and relationships
                filtered_items = self.filter_properties(additional_nodes)
                
                return filtered_items
            else:
                print("No additional context found through graph traversal")
                return []
                
        except Exception as e:
            print(f"Error in traverse_graph: {str(e)}")
            return []
    
    def _clean_cypher_response(self, response: str) -> str:
        """Clean the LLM response by extracting only the Cypher query."""
        cleaned = response.strip()
        
        # Handle the case where there's a markdown code block with explanatory text after
        if '```' in cleaned:
            # Split by code block markers
            parts = cleaned.split('```')
            
            # Look for the cypher query part
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                
                # Skip the opening marker (cypher, sql, etc.)
                if part.lower().startswith('cypher'):
                    part = part[6:].strip()
                
                # Check if this part contains Cypher keywords
                cypher_keywords = ['CALL', 'MATCH', 'WHERE', 'RETURN', 'WITH', 'YIELD', 'ORDER', 'LIMIT', 'UNION', 'OPTIONAL']
                if any(keyword in part.upper() for keyword in cypher_keywords):
                    # This looks like the Cypher query
                    return part.strip()
        
        # Fallback: remove markdown code blocks the old way
        if cleaned.startswith('```cypher'):
            cleaned = cleaned[9:].strip()
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:].strip()
        
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].strip()
        
        # Remove any trailing explanatory text after the query
        lines = cleaned.split('\n')
        cypher_lines = []
        found_cypher = False
        explanation_started = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if found_cypher and not explanation_started:
                    cypher_lines.append('')  # Keep empty lines within query
                continue
            
            # Check for explanatory text patterns
            if (line.lower().startswith('this query') or 
                line.lower().startswith('the query') or
                line.lower().startswith('explanation') or
                line.lower().startswith('note:')):
                explanation_started = True
                continue
            
            if explanation_started:
                continue
                
            # Check if this line looks like Cypher (starts with common Cypher keywords)
            cypher_keywords = ['CALL', 'MATCH', 'WHERE', 'RETURN', 'WITH', 'YIELD', 'ORDER', 'LIMIT', 'UNION', 'OPTIONAL']
            if any(line.upper().startswith(keyword) for keyword in cypher_keywords):
                found_cypher = True
                cypher_lines.append(line)
            elif found_cypher and not explanation_started:
                # Continue collecting lines that are part of the query
                cypher_lines.append(line)
            elif line.lower() == 'null':
                return 'null'
        
        # If we found Cypher lines, join them and clean up
        if cypher_lines:
            result = '\n'.join(cypher_lines).strip()
            # Remove any trailing explanatory text that might have slipped through
            if '\nThis query' in result:
                result = result.split('\nThis query')[0].strip()
            if '\nThe query' in result:
                result = result.split('\nThe query')[0].strip()
            return result
        
        # If no Cypher found but response looks like it might be "null" decision
        if 'sufficient' in cleaned.lower() or 'adequate' in cleaned.lower():
            return 'null'
        
        # If we get here, it's probably not a valid Cypher query
        print(f"⚠️  Response doesn't look like valid Cypher, treating as null: {cleaned[:100]}...")
        return "null"

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the context-to-cypher process."""
        return {
            "last_query": self.last_user_query,
            "num_initial_documents": len(self.last_initial_documents) if self.last_initial_documents else 0,
            "generated_cypher": self.last_generated_cypher,
            "cypher_execution_time": self.last_execution_time
        }

    def _log_interaction(self, user_query: str, initial_docs: List[Dict[str, Any]], 
                        generated_cypher: Optional[str], result_count: int) -> None:
        """Log the interaction for debugging purposes."""
        logger.info(f"ContextToCypher interaction:")
        logger.info(f"  User query: {user_query}")
        logger.info(f"  Initial documents: {len(initial_docs)}")
        logger.info(f"  Generated Cypher: {generated_cypher is not None}")
        logger.info(f"  Result count: {result_count}")
        if generated_cypher:
            logger.debug(f"  Cypher query: {generated_cypher}")


