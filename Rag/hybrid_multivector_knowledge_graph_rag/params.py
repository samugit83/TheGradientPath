# LLM model to use
model = "o4-mini"

# Add the import for prompts at the top of the file
from prompts import KNOWLEDGE_GRAPH_EXTRACTION_PROMPT, KNOWLEDGE_GRAPH_ADDITIONAL_INSTRUCTIONS

# ==================== QUERY CONFIGURATION ====================
# Tuples of (embedding_property, target_label) for vector similarity search
# Each tuple defines which embedding property to search on which node label
VECTOR_SEARCH_CONFIGURATIONS = [
    ("text", "Document"),           # Search embedding_text on Document nodes
    ("hyp_queries", "Document")     # Search embedding_hyp_queries on Document nodes
]

TOP_K_INITIAL = 2 # Maximum number of results to retrieve per vector search
TOP_K_TRAVERSAL = 100 # Maximum number of results to retrieve per graph traversal
ACTIVATE_INITIAL_VECTOR_SEARCH = True # Whether to activate the initial vector search

# ==================== GRAPH TRAVERSAL CONFIGURATION ====================
# Graph traversal method selection
# Options: "context_to_cypher", "kop_limited_bfs", "kop_limited_bfs_pred_llm", "depth_limited_dfs", "depth_limited_dfs_pred_llm", "uniform_cost_search_ucs", "uniform_cost_search_ucs_pred_llm", "astar_search_heuristic", "astar_search_heuristic_pred_llm", "beam_search_over_the_graph", "beam_search_over_the_graph_pred_llm"
GRAPH_TRAVERSAL_METHOD = "beam_search_over_the_graph_pred_llm"


# ==================== INGESTION CONFIGURATION ====================
# Configuration parameters for knowledge graph extraction

# DOCUMENT PROCESSING CONFIGURATION 

# Directory path containing documents to be processed
# Supports glob patterns (e.g., "docs/*", "data/**/*.txt")
documents_source_path = "docs/*"

# SEMANTIC CHUNKING CONFIGURATION

# Method for determining breakpoints in semantic chunking
# Options: "percentile", "standard_deviation", "interquartile"
semantic_chunker_breakpoint_type = "percentile"

# Threshold value for semantic chunking breakpoints
# For percentile: value between 0-100 (e.g., 95.0 = top 5% discontinuities)
# Higher values = fewer, larger chunks; Lower values = more, smaller chunks
semantic_chunker_breakpoint_threshold = 95.0

# Minimum size (in characters) for any semantic chunk
# Prevents creation of very small chunks that lack context
semantic_chunker_min_chunk_size = 2000

# VECTOR EMBEDDING CONFIGURATION

# Dimensionality of vector embeddings (must match embedding model)
# OpenAI text-embedding-ada-002: 1536 dimensions
# OpenAI text-embedding-3-small: 1536 dimensions  
# OpenAI text-embedding-3-large: 3072 dimensions
vector_embedding_dimensions = 1536

# Similarity function used for vector comparisons in Neo4j indexes
# Options: 'cosine', 'euclidean'
# 'cosine' is recommended for most text embeddings
vector_similarity_function = 'cosine'

# Maximum character length for text values to be embedded
# Prevents embedding of very long texts that may cause API errors
max_embedding_text_length = 10000

# Prefix to exclude from embedding (e.g., base64 data URLs)
# Values starting with this prefix will not be embedded
embedding_exclusion_prefix = 'data:'

# Batch size for processing embeddings to avoid API rate limits
# Smaller values = more API calls but less likely to hit rate limits
embedding_batch_size = 10



# ==================== DATABASE QUERY CONFIGURATION ====================

# Maximum number of nodes/relationships to fetch in single queries
# Prevents memory issues with very large graphs
database_query_limit = 1000


# .................. Vector index configuration ..................
# Whether to create a vector index on Document nodes for similarity search
add_vector_index = True
filter_node_labels_to_index = ["Document", "Concept", "Person"]
filter_node_properties_to_index = ["text", "id"]
filter_rels_labels_to_index = ["ALL"]  # Process all relationship types
filter_rels_properties_to_index = ["description"]  # Only embed description property

document_multi_vector_properties = [
    {"property_name": "hyp_queries", 
     "prompt": "Create 3 hypothetical queries that users might ask about the document content. The queries should be presented one after another without bullet points or other symbols."},
    {"property_name": "italian_translation", 
     "prompt": "Translate the document content into Italian. The translation should be accurate and faithful to the original content, without losing any important details or context."}
]


# ------------------ Graph schema extraction configuration ------------------

# Whether to add a base entity label to all nodes in Neo4j
# When True, each node gets an additional '__Entity__' label alongside its specific type (Person, Country, etc.)
# This creates optimized indexes that significantly improve query performance and import speed across all entity types
# The base label enables faster cross-entity searches and graph traversals without any performance overhead
baseEntityLabel = True

# Whether to include source document information with extracted entities
include_source = True

# Node type constraints - only these entity types will be extracted
use_allowed_nodes = False
allowed_nodes = [
    # Documents and Query-related nodes
    "Document", "Article", "Book", "Chapter", "Section", "Page",
    
    # People - Individuals and their roles/professions
    "Person", "Scientist", "Physicist", "Engineer", "Entrepreneur",
    "Mathematician", "Chemist", "Historian", "Author", "Poet",
    "Musician", "Pianist", "Violinist", "Conductor",
    "Physician", "Psychiatrist", "Surgeon",
    "Teacher", "Professor", "Student", "Philosopher",
    "Politician", "Statesman", "Monarch", "Emperor", "PrimeMinister",
    "ReligiousLeader", "Rabbi", "Pope", "Clergy",
    "Family", "Dynasty", "SiblingGroup", "SpousePair",
    
    # Places - Geographic locations and landmarks
    "Place", "Continent", "Region", "Country", "Kingdom", "Empire",
    "State", "Province", "Canton", "Department", "County", "District",
    "Borough", "Municipality", "City", "Town", "Village", "Settlement",
    "Island", "Peninsula", "River", "Lake", "Mountain", "Cemetery",
    "Square", "Street", "Park", "Bridge", "Neighborhood", "Quarter",
    
    # Institutions & Organizations - Formal entities and groups
    "Organization", "Company", "Startup", "Factory", "Laboratory",
    "University", "Polytechnic", "College", "School", "Academy",
    "ResearchInstitute", "Hospital", "Clinic", "Theatre", "Museum",
    "Library", "Archive", "PublishingHouse", "Newspaper", "Journal",
    "PoliticalParty", "Movement", "Committee", "GovernmentAgency",
    "Army", "MilitaryUnit",
    
    # Events - Significant occurrences and happenings
    "Event", "War", "Revolution", "Coup", "Uprising", "Massacre",
    "Conference", "Symposium", "Congress", "Lecture", "Visit",
    "Tour", "Expedition", "Election", "Treaty", "Ceremony",
    
    # Works & Intangibles - Creative and intellectual outputs
    "Theory", "ScientificPaper", "Dissertation",
    "Patent", "Invention", "Equation", "LawOfPhysics", "Principle",
    "Statute", "Regulation", "Policy", "Letter", "Manuscript",
    "Artwork", "Sculpture", "Statue", "Composition", "Song",
    "Film", "Play", "Performance"
]

# Relationship type constraints - only these relationship types will be extracted
use_allowed_relationships = False
allowed_relationships = [
  # Biographical
  "BORN_IN","DIED_IN","BURIED_IN","LIVED_IN","RESIDENCE_OF",
  "MOVED_TO","EMIGRATED_TO","IMMIGRATED_FROM","VISITED",
  "TRAVELLED_WITH","ACCOMPANIED",
  "PARENT_OF","CHILD_OF","SIBLING_OF","SPOUSE_OF","PARTNER_OF",
  "ANCESTOR_OF","DESCENDANT_OF","GRANDPARENT_OF","GRANDCHILD_OF",
  # Education & Career
  "STUDIED_AT","ALUMNUS_OF","TAUGHT_AT","EMPLOYED_BY",
  "FOUNDED","COFOUNDED","FOUNDER_OF","DIRECTED","MANAGED",
  "MEMBER_OF","CHAIR_OF","PRESIDENT_OF","DEAN_OF",
  "WORKED_WITH","COLLABORATED_WITH","MENTORED","SUPERVISED",
  # Intellectual Contributions
  "AUTHORED","COAUTHORED","EDITED","TRANSLATED",
  "PUBLISHED_IN","CITED_BY","CRITICIZED","EXTENDED",
  "DEVELOPED","DISCOVERED","INVENTED","PATENTED",
  "FORMULATED","PROVED","REFUTED","ADVOCATED",
  "NAMED_AFTER","DEDICATED_TO",
  # Awards & Recognition
  "AWARDED","RECIPIENT_OF","NOMINATED_FOR","HONORED_BY",
  # Cultural & Religious
  "CONVERTED_TO","CONVERTED_FROM","BELONGS_TO_TRADITION",
  "PRACTICES","OBSERVES","CELEBRATES",
  # Geospatial containment
  "LOCATED_IN","PART_OF","CAPITAL_OF","BORDERS",
  "ADJACENT_TO","CROSSES","CONNECTS_TO","SERVES_AREA",
  # Organisational hierarchy
  "SUBSIDIARY_OF","DEPARTMENT_OF","UNIT_OF","PROJECT_OF",
  # Event relations
  "OCCURRED_IN","HELD_AT","ORGANIZED_BY","LED_BY",
  "CAUSE_OF","RESULT_OF","TRIGGERED","ENDED_WITH",
  # Transport
  "STOPS_AT","LINKS","OPERATED_BY",
  # Temporal / sequential
  "PRECEDED_BY","SUCCEEDED_BY","CONTEMPORARY_OF",
  "RELATED_TO","INSTANCE_OF","TYPE_OF","ALIAS_OF",
  "DOC_VARIANT"
]

# Whether to use specific relationship tuples that define valid source->relationship->target combinations
# This provides more granular control than just allowed_relationships
use_relationships_tuples = False # set to false if use_allowed_relationships is true
allowed_relationships_tuples = [
    ("Person", "SPOUSE", "Person"),          # Person married to Person
    ("Person", "NATIONALITY", "Country"),    # Person's nationality is Country
    ("Person", "WORKED_AT", "Organization"), # Person worked at Organization
]


# Whether to extract and store properties/attributes for nodes
# If True it extract all properties/attributes thatt LLM finds in the text
# If you specify a list it will only extract the ones in the list
# If False it will not extract any properties/attributes
node_properties = True # or for example["born_year", "died_year", "gender", "nationality", "location", "organization"]

# Whether to extract and store properties/attributes for relationships
# If True it extract all properties/attributes thatt LLM finds in the text
# If you specify a list it will only extract the ones in the list
# If False it will not extract any properties/attributes
relationship_properties = ["description"] # or for example["description", "start_date", "end_date"]

# use_custom_prompt = False  #if is True it exclude all previous parameters
use_custom_prompt = False  # if is True it exclude all previous parameters
if use_custom_prompt:
    custom_prompt = KNOWLEDGE_GRAPH_EXTRACTION_PROMPT
else:
    custom_prompt = None

# Additional instructions for relationship descriptions
add_descriptions_to_relationships = True
if add_descriptions_to_relationships:
    description_prompt = KNOWLEDGE_GRAPH_ADDITIONAL_INSTRUCTIONS
else:
    description_prompt = None

