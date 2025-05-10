import asyncio
from databases import Database
from datetime import datetime, timedelta
import random
from faker import Faker

# Initialize Faker for generating realistic data
fake = Faker()

async def init_db():
    # Connect to the database
    db = Database("sqlite+aiosqlite:///./demo.db")
    await db.connect()

    # Create the people table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE,
        age INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    await db.execute(create_table_query)

    # Generate and insert 30 sample records
    for _ in range(30):
        first_name = fake.first_name()
        last_name = fake.last_name()
        email = fake.email()
        age = random.randint(18, 80)
        created_at = datetime.now() - timedelta(days=random.randint(0, 365))

        insert_query = """
        INSERT INTO people (first_name, last_name, email, age, created_at)
        VALUES (:first_name, :last_name, :email, :age, :created_at)
        """
        
        await db.execute(
            insert_query,
            {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "age": age,
                "created_at": created_at
            }
        )

    # Verify the data
    query = "SELECT COUNT(*) as count FROM people"
    result = await db.fetch_one(query)
    print(f"Successfully inserted {result['count']} records into the people table")

    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(init_db()) 