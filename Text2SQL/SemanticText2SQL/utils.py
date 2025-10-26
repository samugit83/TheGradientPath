#!/usr/bin/env python3
"""
Database Schema Utilities for Text-to-SQL Applications
"""

import json
from typing import Dict, List, Any

def get_table_schema(cursor, table_name: str) -> Dict[str, Any]:
    """Get detailed schema information for a specific table."""
    # Get column information
    cursor.execute("""
        SELECT 
            column_name,
            data_type,
            character_maximum_length,
            is_nullable,
            column_default,
            ordinal_position
        FROM information_schema.columns 
        WHERE table_name = %s 
        ORDER BY ordinal_position;
    """, (table_name,))
    
    columns = []
    for row in cursor.fetchall():
        col_info = {
            'name': row[0],
            'type': row[1],
            'max_length': row[2],
            'nullable': row[3] == 'YES',
            'default': row[4],
            'position': row[5]
        }
        columns.append(col_info)
    
    # Get primary key information
    cursor.execute("""
        SELECT column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
        ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY kcu.ordinal_position;
    """, (table_name,))
    
    primary_keys = [row[0] for row in cursor.fetchall()]
    
    # Get foreign key information
    cursor.execute("""
        SELECT 
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name,
            tc.constraint_name
        FROM information_schema.table_constraints AS tc 
        JOIN information_schema.key_column_usage AS kcu
        ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
        ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
    """, (table_name,))
    
    foreign_keys = []
    for row in cursor.fetchall():
        fk_info = {
            'column': row[0],
            'references_table': row[1],
            'references_column': row[2],
            'constraint_name': row[3]
        }
        foreign_keys.append(fk_info)
    
    # Get unique constraints
    cursor.execute("""
        SELECT column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
        ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = %s AND tc.constraint_type = 'UNIQUE'
        ORDER BY kcu.ordinal_position;
    """, (table_name,))
    
    unique_columns = [row[0] for row in cursor.fetchall()]
    
    # Get check constraints
    cursor.execute("""
        SELECT cc.constraint_name, cc.check_clause
        FROM information_schema.check_constraints cc
        JOIN information_schema.table_constraints tc
        ON cc.constraint_name = tc.constraint_name
        WHERE tc.table_name = %s;
    """, (table_name,))
    
    check_constraints = []
    for row in cursor.fetchall():
        check_constraints.append({
            'name': row[0],
            'condition': row[1]
        })
    
    return {
        'table_name': table_name,
        'columns': columns,
        'primary_keys': primary_keys,
        'foreign_keys': foreign_keys,
        'unique_columns': unique_columns,
        'check_constraints': check_constraints
    }

def get_all_tables(cursor) -> List[str]:
    """Get list of all user tables in the database."""
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
    """)
    return [row[0] for row in cursor.fetchall()]

def format_schema_for_llm(schema_data: Dict[str, Any]) -> str:
    """Format schema data in a human-readable format optimized for LLMs."""
    
    output = []
    output.append("=" * 80)
    output.append(f"TABLE SCHEMA: {schema_data['table_name'].upper()}")
    output.append("=" * 80)
    
    # Table description
    output.append(f"\nTable: {schema_data['table_name']}")
    output.append("-" * 50)
    
    # Columns
    output.append("\nCOLUMNS:")
    for col in schema_data['columns']:
        col_desc = f"  • {col['name']} ({col['type']}"
        if col['max_length']:
            col_desc += f"({col['max_length']})"
        col_desc += ")"
        
        if col['name'] in schema_data['primary_keys']:
            col_desc += " [PRIMARY KEY]"
        if not col['nullable']:
            col_desc += " [NOT NULL]"
        if col['default']:
            col_desc += f" [DEFAULT: {col['default']}]"
        
        output.append(col_desc)
    
    # Primary Keys
    if schema_data['primary_keys']:
        output.append(f"\nPRIMARY KEY: {', '.join(schema_data['primary_keys'])}")
    
    # Foreign Keys
    if schema_data['foreign_keys']:
        output.append("\nFOREIGN KEYS:")
        for fk in schema_data['foreign_keys']:
            output.append(f"  • {fk['column']} → {fk['references_table']}.{fk['references_column']}")
    
    # Unique Constraints
    if schema_data['unique_columns']:
        output.append(f"\nUNIQUE COLUMNS: {', '.join(schema_data['unique_columns'])}")
    
    # Check Constraints
    if schema_data['check_constraints']:
        output.append("\nCHECK CONSTRAINTS:")
        for check in schema_data['check_constraints']:
            output.append(f"  • {check['name']}: {check['condition']}")
    
    output.append("\n")
    return "\n".join(output)

def generate_relationships_summary(schemas: List[Dict[str, Any]]) -> str:
    """Generate a summary of table relationships."""
    output = []
    output.append("=" * 80)
    output.append("TABLE RELATIONSHIPS SUMMARY")
    output.append("=" * 80)
    
    # Create relationship map
    relationships = {}
    for schema in schemas:
        table = schema['table_name']
        relationships[table] = {
            'references': [],
            'referenced_by': []
        }
        
        for fk in schema['foreign_keys']:
            relationships[table]['references'].append(fk['references_table'])
    
    # Find which tables reference each table
    for schema in schemas:
        table = schema['table_name']
        for other_schema in schemas:
            if other_schema['table_name'] != table:
                for fk in other_schema['foreign_keys']:
                    if fk['references_table'] == table:
                        relationships[table]['referenced_by'].append(other_schema['table_name'])
    
    # Print relationships
    for table, rels in relationships.items():
        if rels['references'] or rels['referenced_by']:
            output.append(f"\n{table.upper()}:")
            if rels['references']:
                output.append(f"  References: {', '.join(rels['references'])}")
            if rels['referenced_by']:
                output.append(f"  Referenced by: {', '.join(rels['referenced_by'])}")
    
    output.append("\n")
    return "\n".join(output)

def generate_db_schema(connection) -> tuple[str, str]:
    """
    Generate complete database schema documentation.
    
    Args:
        connection: psycopg2 database connection
        
    Returns:
        tuple: (formatted_schema_text, json_schema_data)
    """
    cursor = connection.cursor()
    
    try:
        # Get all tables
        tables = get_all_tables(cursor)
        
        # Get schema for each table
        schemas = []
        for table in tables:
            schema = get_table_schema(cursor, table)
            schemas.append(schema)
        
        # Generate formatted output
        output_lines = []
        output_lines.append("DATABASE SCHEMA EXTRACTION")
        output_lines.append("=" * 50)
        output_lines.append(f"Total Tables: {len(schemas)}")
        output_lines.append("")
        
        # Print schema for each table
        for schema in schemas:
            output_lines.append(format_schema_for_llm(schema))
        
        # Print relationships summary
        output_lines.append(generate_relationships_summary(schemas))
        
        formatted_text = "\n".join(output_lines)
        json_data = json.dumps(schemas, indent=2, default=str)
        
        return formatted_text, json_data
        
    finally:
        cursor.close()
