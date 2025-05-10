# My App

This project uses MCPWizServer.

## Setup

1.  **Clone the repository (optional):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>/server
    ```

2.  **Create and activate a virtual environment:**

    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application (Development)

To start the server with **hot reloading** enabled (automatically restarts when you save code changes), run the following command from the project root directory:

```bash
cd server
source venv/bin/activate
uvicorn main:mcp.app --reload --port 8000

cd client
source venv/bin/activate
python main.py
```

This uses Uvicorn directly and is the recommended way during development.

The application will be available at `http://127.0.0.1:8000` by default.

## API Examples (using curl)

Assuming the server is running on `http://127.0.0.1:8000` and you have `jq` installed for pretty-printing JSON.

### Health Check

Checks if the server is running.

```bash
curl http://127.0.0.1:8000/health | jq
```

Expected Response:
```json
{
  "ok": true
}
```

### Server Info

Gets basic information about the server and its capabilities.

```bash
curl -H "X-API-Key: alpha-123" http://127.0.0.1:8000/info | jq
```

Expected Response (example):
```json
{
  "name": "My App",
  "version": "0.1.0",
  "capabilities": {
    "tools": {}
  }
}
```

### List Tools

Gets the list of available tools.

```bash
curl -H "X-API-Key: alpha-123" http://127.0.0.1:8000/tools/list | jq
```

Expected Response (example):
```json
{
  "tools": [
    {
      "name": "query_db",
      "description": "Run a SQL query",
      "input_schema": {
        "type": "object",
        "properties": {
          "sql": {
            "type": "string",
            "title": "Sql"
          }
        },
        "required": [
          "sql"
        ]
      },
      "annotations": null
    },
    {
      "name": "add",
      "description": "Pure calculator, no context needed",
      "input_schema": {
        "type": "object",
        "properties": {
          "a": {
            "type": "number",
            "title": "A"
          },
          "b": {
            "type": "number",
            "title": "B"
          }
        },
        "required": [
          "a",
          "b"
        ]
      },
      "annotations": null
    }
  ]
}
```

### Call a Tool

Gets a result for a given tool.

```bash
curl -X POST -H "X-API-Key: alpha-123" http://127.0.0.1:8000/tools/call \
     -H "Content-Type: application/json" \
     -d '{
           "name": "add",
           "arguments": {
             "a": 5,
             "b": 7
           }
         }' | jq
```

Expected Response:
```json
{
  "content": [
    {
      "type": "text",
      "text": "12.0"
    }
  ]
}
```

---

