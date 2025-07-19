#!/bin/bash
set -e

echo "🚀 Starting Vision-RAG application..."

# Initialize database
echo "🔧 Initializing database..."
python init_db.py --setup

# Check database health
echo "🩺 Checking database health..."
python init_db.py --check

echo "✅ Database is ready!"

# Execute the main command
echo "🏃 Starting application..."
exec "$@" 