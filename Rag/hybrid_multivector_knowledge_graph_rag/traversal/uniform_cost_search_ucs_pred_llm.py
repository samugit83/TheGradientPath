import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from .uniform_cost_search_ucs import UniformCostSearchUCS
from params import model
from prompts import RELATIONSHIP_SELECTION_PROMPT

class RelationshipSelection(BaseModel):
    """Model for relationship selection output."""
    selected_relationships: List[str] = Field(description="List of selected relationship types")
    reasoning: str = Field(description="Brief reasoning for the selection")

class UniformCostSearchUCSWithLLM:
    """
    A graph traversal class that implements predicate-constrained UCS.
    This approach uses an LLM to determine which relationship types are relevant
    for a given query, then filters the UCS traversal to only follow those relationships.
    """
    
    def __init__(
        self, 
        max_total_nodes: int = 150,
        remove_mentions_nodes: bool = True,
        model: str = model
    ):
        """
        Initialize the predicate-constrained UCS traversal.
        
        Args:
            max_total_nodes: Maximum total nodes to return (default: 150)
            remove_mentions_nodes: If True, filter out nodes discovered through MENTIONS relationships from initial docs (default: True)
            model: LLM model to use for relationship selection (default: "gpt-4o-mini")
        """
        self.max_total_nodes = max_total_nodes
        self.remove_mentions_nodes = remove_mentions_nodes
        self.model = model
        self._setup_environment()
        self._initialize_neo4j_connection()
        self._initialize_llm()
    
    def _setup_environment(self) -> None:
        """Load environment variables and validate required credentials."""
        load_dotenv()
        
        # Neo4j Credentials validation
        required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "OPENAI_API_KEY"]
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
    
    def _initialize_llm(self) -> None:
        """Initialize the LLM for relationship selection."""
        self.llm = ChatOpenAI(model=self.model)
        
        # Setup the output parser
        self.output_parser = PydanticOutputParser(pydantic_object=RelationshipSelection)
        
        # Create the prompt template
        self.prompt_template = PromptTemplate(
            template=RELATIONSHIP_SELECTION_PROMPT,
            input_variables=["user_query", "relationship_types"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
    
    def _get_all_relationship_types(self) -> List[str]:
        """
        Get all relationship types from the Neo4j database.
        
        Returns:
            List of all relationship types in the database
        """
        try:
            cypher_query = """
            MATCH ()-[r]->()
            RETURN DISTINCT type(r) as relationship_type
            ORDER BY relationship_type
            """
            
            result = self.graph.query(cypher_query)
            relationship_types = [row['relationship_type'] for row in result]

            print(relationship_types)
            
            print(f"🔍 Found {len(relationship_types)} relationship types in the database")
            return relationship_types
            
        except Exception as e:
            print(f"⚠️ Error getting relationship types: {e}")
            return []
    
    def _select_relevant_relationships(self, user_query: str, relationship_types: List[str]) -> List[str]:
        """
        Use LLM to select relevant relationship types for the user query.
        
        Args:
            user_query: The user's search query
            relationship_types: List of all available relationship types
            
        Returns:
            List of selected relationship types
        """
        if not relationship_types:
            print("⚠️ No relationship types available for selection")
            return []
        
        try:
            # Format the relationship types for the prompt
            formatted_types = "\n".join([f"- {rel_type}" for rel_type in relationship_types])
            
            # Create the prompt
            prompt = self.prompt_template.format(
                user_query=user_query,
                relationship_types=formatted_types
            )
            
            # Get LLM response
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            # Parse the response
            parsed_response = self.output_parser.parse(response.content)
            
            selected_rels = parsed_response.selected_relationships
            if "MENTIONS" not in selected_rels:
                selected_rels.append("MENTIONS")
            
            reasoning = parsed_response.reasoning
            
            print(f"🤖 LLM selected {len(selected_rels)} relationship types:")
            for rel in selected_rels:
                print(f"   - {rel}")
            print(f"💭 Reasoning: {reasoning}")
            
            return selected_rels
            
        except Exception as e:
            print(f"⚠️ Error selecting relationships with LLM: {e}")
            print("Falling back to no filtering...")
            return []
    
    def traverse_graph(self, relevant_docs: List[Dict[str, Any]], user_query: str) -> List[Dict[str, Any]]:
        """
        Retrieve additional context through predicate-constrained UCS traversal.
        
        This method:
        1. Gets all relationship types from the database
        2. Uses LLM to select relevant relationship types for the query
        3. Performs UCS traversal filtering by selected relationship types
        4. Returns additional context nodes
        
        Args:
            relevant_docs: List of initially retrieved documents
            user_query: The user's question
            
        Returns:
            List of additional context nodes with embedding properties removed
        """
        try:
            print(f"🔍 Starting predicate-constrained UCS traversal")
            
            # Get all relationship types from the database
            all_relationship_types = self._get_all_relationship_types()
            
            if not all_relationship_types:
                print("⚠️ No relationship types found in database")
                return []
            
            # Use LLM to select relevant relationship types
            selected_relationships = self._select_relevant_relationships(user_query, all_relationship_types)
            
            # Initialize UniformCostSearchUCS with relationship filtering
            if selected_relationships:
                print(f"🎯 Using {len(selected_relationships)} selected relationship types for filtering")
                ucs_traversal = UniformCostSearchUCS(
                    max_total_nodes=self.max_total_nodes,
                    remove_mentions_nodes=self.remove_mentions_nodes,
                    rel_type_filter=selected_relationships
                )
            else:
                print("⚠️ No relationships selected by LLM, proceeding without filtering")
                ucs_traversal = UniformCostSearchUCS(
                    max_total_nodes=self.max_total_nodes,
                    remove_mentions_nodes=self.remove_mentions_nodes,
                    rel_type_filter=None
                )
            
            # Perform the traversal
            additional_nodes = ucs_traversal.traverse_graph(relevant_docs, user_query)
            
            if additional_nodes:
                print(f"✅ Retrieved {len(additional_nodes)} additional context nodes via predicate-constrained UCS")
            else:
                print("⚠️ No additional context found through predicate-constrained UCS traversal")
            
            return additional_nodes
            
        except Exception as e:
            print(f"⚠️ Error in predicate-constrained UCS traversal: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_traversal_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the traversal configuration.
        
        Returns:
            Dictionary containing traversal statistics
        """
        return {
            "method": "uniform_cost_search_ucs_pred_llm",
            "max_total_nodes": self.max_total_nodes,
            "remove_mentions_nodes": self.remove_mentions_nodes,
            "model": self.model
        }
