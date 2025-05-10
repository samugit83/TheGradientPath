from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any
from databases import Database
from server import MCPWizServer, Context, InMemoryKeyStore
from fastapi import FastAPI
from dotenv import load_dotenv # Import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()

class ServerContext(dict):
    def __init__(self, db: Database, key_store: InMemoryKeyStore):
        super().__init__()
        self.db = db
        self.key_store = key_store

@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncIterator[Dict[str, Any]]:
    db = Database("sqlite+aiosqlite:///./demo.db")
    key_store = InMemoryKeyStore()
    await db.connect()
    try:
        app.state.db = db
        app.state.key_store = key_store
        context = ServerContext(db=db, key_store=key_store)
        yield context
    finally:
        await db.disconnect()

mcp = MCPWizServer("My App", lifespan=app_lifespan)


@mcp.tool()
async def text_to_sql_query(ctx: Context, text: str) -> list[dict]:
    """Convert natural language to a SQL query, execute it, and return the results."""

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    db = ctx.request_context.lifespan_context.db
    table_info_query = """
    SELECT sql FROM sqlite_master 
    WHERE type='table' AND name='people'
    """
    table_info = await db.fetch_one(table_info_query)
    table_context = table_info[0] if table_info else "Table schema not found"

    print(f"Table Context: {table_context}")

    sql_conversion_prompt = f"""
    You are a SQL expert. Convert the following natural language query to a valid SQL query.
    
    Database Context:
    {table_context}
    
    Natural Language Query: {text}
    
    Requirements:
    1. Use standard SQL syntax
    2. Always include proper table aliases
    3. Use appropriate SQL functions for date/time operations
    4. Include proper error handling for NULL values
    5. Use parameterized queries where appropriate
    6. Follow SQL best practices for performance
    7. Return only the SQL query without any explanations
    
    Example format:
    SELECT p.first_name, p.last_name
    FROM people p
    WHERE p.age > 18
    ORDER BY p.last_name;
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a SQL expert that converts natural language to SQL queries. Always follow SQL best practices and security guidelines."},
                {"role": "user", "content": sql_conversion_prompt}
            ],
            temperature=0.1,  
            max_tokens=500,  
            top_p=0.1     
        )
        
        sql_query = response.choices[0].message.content.strip()

        print(f"SQL Query: {sql_query}")
        
        if sql_query.startswith("```sql"):
            sql_query = sql_query[len("```sql"):].strip()
        elif sql_query.startswith("```"): 
            sql_query = sql_query[len("```"):].strip()
        
        if sql_query.endswith("```"):
            sql_query = sql_query[:-len("```")].strip()
        rows = await db.fetch_all(sql_query)
        results = [dict(r) for r in rows]
        
        formatted_results = ""
        for row in results:
            formatted_results += str(row) + "\n"
            
        return formatted_results.strip()

            
    except Exception as e:
        return f"Error generating SQL: {str(e)}"

@mcp.tool()
async def query_db(ctx: Context, sql: str) -> list[dict]:
    """Run a SQL query"""
    if ctx.request_context and ctx.request_context.lifespan_context:
        rows = await ctx.request_context.lifespan_context.db.fetch_all(sql)
        return [dict(r) for r in rows]
    else:
        # Handle case where context might not be fully populated, e.g., during certain test scenarios or if setup is incorrect
        # Depending on requirements, could raise an error or return an empty list with a warning.
        # For now, returning an error message consistent with potential issues.
        # Consider logging this situation as well.
        return [{"error": "Lifespan context not available for database query."}]

@mcp.tool()
def add(a: float, b: float, c: float = 0.0) -> str:
    """It's a calculator that adds numbers""" 
    return str(a + b + c)

@mcp.prompt()
def summarize_text(text_to_summarize: str, max_sentences: int = 3, style: str = "neutral") -> str:
    """
    Summarize the provided text concisely, adhering to a sentence limit and a specific style.
    """
    return f"""Please summarize the following text in no more than {max_sentences} sentences.
The summary should be written in a {style} tone.
Focus on extracting the most critical information and key takeaways.

Text to summarize:
{text_to_summarize}
"""

@mcp.prompt()
def translate_text(text_to_translate: str, target_language: str = "Spanish", context_hint: str | None = None, formality: str = "neutral") -> str:
    """
    Translate the provided text to the target language, optionally using a context hint and specifying formality.
    """
    prompt_lines = [
        f"Please translate the following text to {target_language}.",
        f"The desired formality of the translation is {formality}."
    ]
    if context_hint:
        prompt_lines.append(f"For context, this text is related to: {context_hint}.")
    
    prompt_lines.append("\nText to translate:")
    prompt_lines.append(text_to_translate)
    
    return "\n".join(prompt_lines)



@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "App configuration here"

@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Dynamic user data"""
    return f"Profile data for user {user_id}"



if __name__ == "__main__":
    mcp.run(port=8000)
