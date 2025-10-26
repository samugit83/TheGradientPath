#!/usr/bin/env python3
"""
Text-to-SQL Agent with Vector Embeddings Support
Professional agent for converting natural language queries to SQL using OpenAI GPT models.
"""

import os
import json
import logging
import psycopg2
import psycopg2.extensions
import sqlglot
import traceback
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from utils import generate_db_schema
from prompt import (
    create_text_to_sql_prompt, 
    create_final_answer_prompt, 
    create_sql_retry_prompt,
    create_final_answer_user_message
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


class AgentTextToSql:
    """
    Professional Text-to-SQL Agent that converts natural language queries to SQL.
    
    This agent uses OpenAI's GPT models to understand user intent and generate
    accurate SQL queries based on the database schema. It supports vector embeddings
    for semantic search capabilities.
    """
    
    # Default database configuration
    DEFAULT_DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'books_db',
        'user': 'bookadmin',
        'password': 'bookpass123'
    }
    
    def __init__(self, db_config: Dict[str, Any] = None, model: str = "gpt-4.1", temperature: float = 0.1):
        """
        Initialize the Text-to-SQL Agent.
        
        Args:
            db_config: Database configuration dictionary (optional, uses default if not provided)
            model: OpenAI model to use (default: gpt-4.1)
            temperature: Model temperature for response generation (default: 0.1 for consistency)
        """
        self.model = model
        self.temperature = temperature
        self.client = None
        self.database_schema = None
        self.db_config = db_config or self.DEFAULT_DB_CONFIG
        
        # Initialize OpenAI client
        self._initialize_openai_client()
        
        # Load database schema
        self._load_database_schema()
    
    def _initialize_openai_client(self) -> None:
        """Initialize OpenAI client with API key from environment."""
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please add your OpenAI API key to the .env file."
            )
        
        self.client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    def _load_database_schema(self) -> None:
        """Generate database schema directly from database using utils."""
        try:
            # Connect to database
            connection = psycopg2.connect(**self.db_config)
            
            # Generate schema
            formatted_text, json_data = generate_db_schema(connection)
            self.database_schema = formatted_text
            
            # Close connection
            connection.close()
            
            logger.info("Database schema generated successfully")
            
        except Exception as e:
            logger.error(f"Error loading database schema: {str(e)}")
            raise
    
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for the Text-to-SQL agent.
        
        Returns:
            str: System prompt for OpenAI API
        """
        return create_text_to_sql_prompt(self.database_schema)

    def generate_sql(self, user_request: str) -> Dict[str, Any]:
        """
        Generate SQL query from natural language request.
        
        Args:
            user_request: Natural language description of the desired query
            
        Returns:
            Dict with 'sql_query' and 'need_embedding' fields
        """
        try:
            logger.info(f"Processing user request: {user_request}")
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_request}
                ],
                temperature=self.temperature,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Extract and parse JSON response
            response_text = response.choices[0].message.content.strip()
            result = json.loads(response_text)
            
            # Validate response structure
            required_fields = ["sql_query", "need_embedding", "embedding_params"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field '{field}' in LLM response")
            
            # Validate embedding_params structure
            if not isinstance(result["embedding_params"], list):
                raise ValueError("embedding_params must be a list")
            
            # If need_embedding is true, ensure embedding_params is not empty
            if result["need_embedding"] and not result["embedding_params"]:
                raise ValueError("need_embedding is true but embedding_params is empty")
            
            # If need_embedding is false, ensure embedding_params is empty
            if not result["need_embedding"] and result["embedding_params"]:
                raise ValueError("need_embedding is false but embedding_params is not empty")
            
            # Validate that embedding_params count matches placeholder count in SQL
            if result["need_embedding"]:
                placeholder_count = result["sql_query"].count('%s')
                params_count = len(result["embedding_params"])
                
                if placeholder_count != params_count:
                    logger.warning(
                        f"Placeholder mismatch: SQL has {placeholder_count} %s placeholders "
                        f"but embedding_params has {params_count} entries"
                    )
                    logger.warning("This may cause execution errors. The LLM should be retrained.")
            
            logger.info("SQL query generated successfully")
            logger.info(f"Generated SQL: {result['sql_query']}")
            logger.info(f"Needs Embedding: {result['need_embedding']}")
            if result['need_embedding']:
                logger.info(f"Embedding Parameters: {len(result['embedding_params'])} parameter(s)")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            raise
    
    def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process a user request and return structured response.
        
        Args:
            user_request: Natural language description of the desired query
            
        Returns:
            Dict containing the generated SQL, need_embedding flag, embedding_params, and metadata
        """
        try:
            result = self.generate_sql(user_request)
            
            return {
                "success": True,
                "user_request": user_request,
                "sql_query": result["sql_query"],
                "need_embedding": result["need_embedding"],
                "embedding_params": result["embedding_params"],
                "model_used": self.model
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {
                "success": False,
                "user_request": user_request,
                "error": str(e),
                "model_used": self.model,
                "need_embedding": None,
                "embedding_params": []
            }
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded database schema.
        
        Returns:
            Dict containing schema metadata
        """
        return {
            "schema_loaded": self.database_schema is not None,
            "schema_length": len(self.database_schema) if self.database_schema else 0,
            "model": self.model,
            "temperature": self.temperature
        }
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using OpenAI API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
                dimensions=EMBEDDING_DIMENSIONS
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def _generate_embeddings_for_params(self, embedding_params: List[Dict[str, str]]) -> List[str]:
        """
        Generate embeddings for all parameters in the query.
        
        Args:
            embedding_params: List of embedding parameter dictionaries
            
        Returns:
            List of embedding vectors as PostgreSQL-formatted strings
        """
        embeddings = []
        
        for param in embedding_params:
            text_to_embed = param['text_to_embed']
            embedding = self._generate_embedding(text_to_embed)
            
            # Convert to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            embeddings.append(embedding_str)
        
        return embeddings
    
    def _validate_sql_query(self, sql_query: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL query for security and safety.
        
        Checks:
        1. Query must be parseable
        2. Only SELECT statements allowed (no INSERT, UPDATE, DELETE, DROP, etc.)
        3. No dangerous operations (CREATE, ALTER, TRUNCATE, etc.)
        4. No multiple statements (semicolon separation)
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check for multiple statements (basic protection against SQL injection)
            statements = sql_query.strip().split(';')
            # Remove empty strings from split
            statements = [s.strip() for s in statements if s.strip()]
            
            if len(statements) > 1:
                return False, "Multiple SQL statements detected. Only single SELECT queries are allowed."
            
            # Define dangerous keywords that should not appear
            dangerous_keywords = [
                'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
                'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE',
                'EXEC', 'EXECUTE', 'CALL'
            ]
            
            # Check for dangerous keywords in uppercase query
            query_upper = sql_query.upper()
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    return False, f"Dangerous operation detected: {keyword}"
            
            # Additional check: look for INTO clause (SELECT INTO is a write operation)
            if 'INTO' in query_upper and 'INTO' not in query_upper[query_upper.find('FROM'):]:
                # INTO appears before FROM, which could be SELECT INTO
                return False, "SELECT INTO operations are not allowed"
            
            # Check if query starts with SELECT (case-insensitive)
            if not query_upper.strip().startswith('SELECT'):
                return False, "Only SELECT queries are allowed"
            
            # Check for vector columns in GROUP BY (this will cause errors in PostgreSQL)
            if 'GROUP BY' in query_upper:
                # Extract the GROUP BY clause
                group_by_start = query_upper.find('GROUP BY')
                group_by_clause = sql_query[group_by_start:].split('ORDER BY')[0].split('LIMIT')[0]
                
                # Check if any _embed fields are in GROUP BY
                if '_embed' in group_by_clause.lower():
                    return False, "Vector columns (fields ending with '_embed') cannot be used in GROUP BY clause. Use primary keys or scalar fields only."
            
            # Prepare query for parsing by handling pgvector operators
            # sqlglot doesn't understand PostgreSQL's <-> operator for vector distance
            # We'll temporarily replace it for parsing validation
            query_for_parsing = sql_query
            has_vector_ops = False
            
            # Replace pgvector operators with standard operators for parsing
            if '<->' in query_for_parsing:
                has_vector_ops = True
                # Replace vector distance operator with a dummy function call
                query_for_parsing = query_for_parsing.replace('<->', '+')
            
            # Parse the SQL query to check structure
            try:
                parsed = sqlglot.parse_one(query_for_parsing, read='postgres')
            except Exception as e:
                # If parsing fails, do a basic syntax check instead
                logger.warning(f"SQL parsing warning: {str(e)}")
                # Allow the query if it passed all other checks
                if query_upper.strip().startswith('SELECT'):
                    logger.info("SQL query validation passed (basic check)")
                    return True, None
                return False, f"SQL parsing error: {str(e)}"
            
            # Check if it's a SELECT statement
            if not isinstance(parsed, sqlglot.exp.Select):
                statement_type = type(parsed).__name__
                return False, f"Only SELECT queries are allowed. Detected: {statement_type}"
            
            logger.info("SQL query validation passed")
            return True, None
            
        except Exception as e:
            logger.error(f"SQL validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def execute_sql(self, sql_query: str, need_embedding: bool = False, 
                    embedding_params: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Execute SQL query against the database with validation.
        
        Args:
            sql_query: SQL query to execute
            need_embedding: Whether the query needs embedding parameters
            embedding_params: List of embedding parameter dictionaries (required if need_embedding is True)
            
        Returns:
            Dict containing query results and metadata
        """
        try:
            # SECURITY: Validate SQL query before execution
            logger.info("Validating SQL query for security...")
            is_valid, error_message = self._validate_sql_query(sql_query)
            
            if not is_valid:
                logger.error(f"SQL validation failed: {error_message}")
                
                # Check if this is a security issue or a fixable error
                is_security_issue = any(keyword in error_message for keyword in [
                    'Dangerous operation', 'Multiple SQL statements', 'Only SELECT queries',
                    'SELECT INTO', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE'
                ])
                
                return {
                    "success": False,
                    "error": f"Query validation failed: {error_message}",
                    "results": [],
                    "column_names": [],
                    "row_count": 0,
                    "validation_failed": True,
                    "is_security_issue": is_security_issue  # Distinguish security from syntax errors
                }
            
            logger.info("Connecting to database for query execution...")
            connection = psycopg2.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Generate embeddings if needed
            query_params = []
            if need_embedding:
                if not embedding_params:
                    raise ValueError("embedding_params required when need_embedding is True")
                
                logger.info(f"Generating {len(embedding_params)} embedding(s) for query...")
                embeddings_generated = self._generate_embeddings_for_params(embedding_params)
                
                # Count how many %s placeholders are in the query
                placeholder_count = sql_query.count('%s')
                
                # If there are more placeholders than embeddings, replicate embeddings in order
                # This handles cases where the LLM reuses the same embedding multiple times
                if placeholder_count > len(embeddings_generated):
                    logger.info(f"Replicating {len(embeddings_generated)} embeddings to match {placeholder_count} placeholders")
                    
                    # Replicate embeddings cyclically to match placeholder count
                    query_params = []
                    for i in range(placeholder_count):
                        embedding_index = i % len(embeddings_generated)
                        query_params.append(embeddings_generated[embedding_index])
                else:
                    query_params = embeddings_generated
            
            # Execute query
            placeholder_count = sql_query.count('%s')
            logger.info(f"Executing SQL query...")
            
            # Verify parameter count matches
            if query_params and len(query_params) != placeholder_count:
                error_msg = f"Parameter count mismatch: query expects {placeholder_count} but got {len(query_params)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if query_params:
                try:
                    # Manual parameter substitution to avoid psycopg2 issues with complex queries
                    query_with_params = sql_query
                    for param in query_params:
                        # Replace first occurrence of %s::vector with the embedding
                        query_with_params = query_with_params.replace('%s::vector', f"'{param}'::vector", 1)
                    
                    # Execute without parameters (they're already in the query)
                    cursor.execute(query_with_params)
                except psycopg2.Error as e:
                    # Catch ALL PostgreSQL-specific errors and provide better error message
                    logger.error(f"PostgreSQL execution error: {str(e)}")
                    cursor.close()
                    connection.close()
                    raise ValueError(f"PostgreSQL error: {str(e)}")
                except Exception as e:
                    # Catch any other errors
                    logger.error(f"Unexpected error during execution: {str(e)}")
                    cursor.close()
                    connection.close()
                    raise
            else:
                try:
                    cursor.execute(sql_query)
                except psycopg2.Error as e:
                    logger.error(f"PostgreSQL execution error: {str(e)}")
                    cursor.close()
                    connection.close()
                    raise ValueError(f"PostgreSQL error: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error during execution: {str(e)}")
                    cursor.close()
                    connection.close()
                    raise
            
            # Fetch results
            try:
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # Convert results to list of dictionaries
                results_list = []
                for row in results:
                    row_dict = {}
                    for i, col_name in enumerate(column_names):
                        row_dict[col_name] = row[i]
                    results_list.append(row_dict)
                
                logger.info(f"Query executed successfully. Retrieved {len(results_list)} row(s)")
                
                # Close connection
                cursor.close()
                connection.close()
                
                return {
                    "success": True,
                    "results": results_list,
                    "column_names": column_names,
                    "row_count": len(results_list)
                }
                
            except psycopg2.ProgrammingError:
                # No results to fetch (e.g., INSERT, UPDATE, DELETE)
                connection.commit()
                affected_rows = cursor.rowcount
                
                cursor.close()
                connection.close()
                
                logger.info(f"Query executed successfully. {affected_rows} row(s) affected")
                
                return {
                    "success": True,
                    "results": [],
                    "column_names": [],
                    "row_count": 0,
                    "affected_rows": affected_rows
                }
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            if 'connection' in locals() and connection:
                connection.rollback()
                connection.close()
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "column_names": [],
                "row_count": 0
            }
    
    def generate_final_answer(self, user_request: str, query_results: Dict[str, Any], 
                              sql_query: str = None) -> str:
        """
        Generate a natural language answer based on the user request and query results.
        
        Args:
            user_request: Original user request
            query_results: Results from the SQL query execution
            sql_query: The SQL query that was executed (optional, for context)
            
        Returns:
            Natural language answer as a string
        """
        try:
            logger.info("Generating final natural language answer...")
            
            # Prepare the results summary
            if not query_results.get('success', False):
                results_text = f"Error executing query: {query_results.get('error', 'Unknown error')}"
            elif query_results['row_count'] == 0:
                results_text = "No results found."
            else:
                # Format results as a readable text
                results_text = f"Found {query_results['row_count']} result(s):\n\n"
                for i, row in enumerate(query_results['results'][:20], 1):  # Limit to first 20 rows
                    results_text += f"Result {i}:\n"
                    for key, value in row.items():
                        # Skip embedding columns and similarity scores for readability
                        if not key.endswith('_embed') and key != 'similarity' and key != 'combined_similarity':
                            results_text += f"  - {key}: {value}\n"
                    results_text += "\n"
                
                if query_results['row_count'] > 20:
                    results_text += f"... and {query_results['row_count'] - 20} more results\n"
            
            # Create prompt for final answer generation
            system_prompt = create_final_answer_prompt()
            user_message = create_final_answer_user_message(user_request, results_text, sql_query)
            
            # Make API call to generate answer
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # Slightly higher for more natural responses
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Error generating final answer: {str(e)}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"
    
    def _regenerate_sql_with_error_feedback(self, user_request: str, 
                                              attempt_history: List[Dict[str, str]], 
                                              attempt: int) -> Dict[str, Any]:
        """
        Regenerate SQL query by providing comprehensive error feedback to the LLM.
        
        Args:
            user_request: Original user request
            attempt_history: List of previous attempts with their SQL and errors
            attempt: Current attempt number
            
        Returns:
            Dict with regenerated SQL query
        """
        logger.info(f"Regenerating SQL query with error feedback from {len(attempt_history)} previous attempt(s)...")
        
        # Create system prompt
        system_prompt = self._create_system_prompt()
        
        # Build comprehensive error history
        error_history_text = ""
        for i, prev_attempt in enumerate(attempt_history, 1):
            error_history_text += f"""
ATTEMPT {i}:
SQL Query: {prev_attempt['sql']}
Error: {prev_attempt['error']}
---"""
        
        # Create user message using prompt function
        user_message = create_sql_retry_prompt(user_request, error_history_text)
        
        try:
            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Extract and parse JSON response
            response_text = response.choices[0].message.content.strip()
            result = json.loads(response_text)
            
            # Validate response structure
            required_fields = ["sql_query", "need_embedding", "embedding_params"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field '{field}' in LLM response")
            
            return result
            
        except Exception as e:
            logger.error(f"Error regenerating SQL: {str(e)}")
            raise
    
    def process_request_with_execution(self, user_request: str, max_retries: int = 4) -> Dict[str, Any]:
        """
        Complete pipeline with retry mechanism: Generate SQL, execute it, and generate final answer.
        
        If execution fails, the system will retry up to max_retries times, providing comprehensive
        error feedback including all previous attempts to the LLM for regeneration.
        
        Args:
            user_request: Natural language description of the desired query
            max_retries: Maximum number of retry attempts (default: 4)
            
        Returns:
            Dict containing all information including the final answer
        """
        attempt = 0
        last_error = None
        sql_result = None
        query_results = None
        attempt_history = []  # Track all previous attempts
        
        while attempt < max_retries:
            try:
                attempt += 1
                logger.info("=" * 80)
                logger.info(f"ATTEMPT {attempt}/{max_retries}")
                logger.info("=" * 80)
                
                # Step 1: Generate SQL query (or regenerate with error feedback)
                if attempt == 1:
                    logger.info("STEP 1: GENERATING SQL QUERY")
                    logger.info("=" * 80)
                    sql_result = self.generate_sql(user_request)
                else:
                    logger.info(f"STEP 1: REGENERATING SQL QUERY (Attempt {attempt})")
                    logger.info("=" * 80)
                    sql_result = self._regenerate_sql_with_error_feedback(
                        user_request=user_request,
                        attempt_history=attempt_history,
                        attempt=attempt
                    )
                
                # Step 2: Execute SQL query
                logger.info("=" * 80)
                logger.info("STEP 2: EXECUTING SQL QUERY")
                logger.info("=" * 80)
                query_results = self.execute_sql(
                    sql_query=sql_result['sql_query'],
                    need_embedding=sql_result['need_embedding'],
                    embedding_params=sql_result['embedding_params']
                )
                
                # Check if execution was successful
                if not query_results.get('success', False):
                    last_error = query_results.get('error', 'Unknown execution error')
                    logger.warning(f"Attempt {attempt} failed: {last_error}")
                    
                    # Add this failed attempt to history
                    attempt_history.append({
                        'sql': sql_result['sql_query'],
                        'error': last_error
                    })
                    
                    # Check if this is a security issue (should abort) or fixable error (can retry)
                    if query_results.get('validation_failed', False):
                        is_security_issue = query_results.get('is_security_issue', True)
                        
                        if is_security_issue:
                            logger.error("SECURITY ISSUE detected - aborting retries")
                            break
                        else:
                            logger.warning("Validation failed but error is fixable - will retry with feedback")
                            # Continue to next attempt
                            continue
                    
                    # Continue to next attempt
                    continue
                
                # Step 3: Generate final answer (only if execution succeeded)
                logger.info("=" * 80)
                logger.info("STEP 3: GENERATING FINAL ANSWER")
                logger.info("=" * 80)
                final_answer = self.generate_final_answer(
                    user_request=user_request,
                    query_results=query_results,
                    sql_query=sql_result['sql_query']
                )
                
                logger.info("=" * 80)
                logger.info(f"PIPELINE COMPLETED SUCCESSFULLY (Attempt {attempt})")
                logger.info("=" * 80)
                
                return {
                    "success": True,
                    "user_request": user_request,
                    "sql_query": sql_result['sql_query'],
                    "need_embedding": sql_result['need_embedding'],
                    "embedding_params": sql_result['embedding_params'],
                    "query_results": query_results,
                    "final_answer": final_answer,
                    "model_used": self.model,
                    "attempts": attempt,
                    "failed_attempts": attempt_history  # Include history for transparency
                }
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt} error: {last_error}")
                
                # Add this failed attempt to history
                if sql_result:
                    attempt_history.append({
                        'sql': sql_result.get('sql_query', 'Query generation failed'),
                        'error': last_error
                    })
                
                # If this is not the last attempt, continue retrying
                if attempt < max_retries:
                    continue
                # If it's the last attempt, break and return error
                break
        
        # All attempts failed
        logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
        
        return {
            "success": False,
            "user_request": user_request,
            "error": f"Failed after {attempt} attempts. Last error: {last_error}",
            "model_used": self.model,
            "attempts": attempt,
            "last_sql_query": sql_result['sql_query'] if sql_result else None,
            "failed_attempts": attempt_history  # Include complete history for debugging
        }
