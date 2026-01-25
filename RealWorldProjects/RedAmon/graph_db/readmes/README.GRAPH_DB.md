# Neo4j Graph Database for RedAmon

## Quick Start

```bash
cd graph_db
docker compose up -d
```

## Endpoints

- **Browser UI**: http://localhost:7474
- **Bolt (Python driver)**: bolt://localhost:7687

## Credentials

Configured via root `.env` file:
- `NEO4J_URI` - Bolt connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USER` - Username (default: `neo4j`)
- `NEO4J_PASSWORD` - Your password

## Configuration

Set `UPDATE_GRAPH_DB = True` in `params.py` to automatically populate the graph after `domain_discovery` module completes.

## Docker Commands

```bash
# Start Neo4j
cd graph_db && docker compose up -d

# Stop Neo4j
docker compose down

# Stop and remove all data (fresh start)
docker compose down -v

# View logs
docker compose logs -f

# View last 50 lines of logs
docker compose logs --tail 50

# Check container status
docker compose ps

# Restart Neo4j
docker compose restart

# Enter container shell
docker exec -it redamon-neo4j bash
```

## Cypher Queries

Run these in the Neo4j Browser at http://localhost:7474

### View All Data

```cypher
-- Show all nodes and relationships
MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m

-- Show all nodes (browser auto-draws relationships)
MATCH (n) RETURN n

-- Count all nodes by type
MATCH (n) RETURN labels(n) AS type, count(n) AS count
```

### Query by Project

```cypher
-- Show all nodes and relationships for a project
MATCH (n {project_id: "first_test"})
OPTIONAL MATCH (n)-[r]->(m)
RETURN n, r, m

-- Filter by both user_id and project_id
MATCH (n {user_id: "samgiam", project_id: "first_test"})
OPTIONAL MATCH (n)-[r]->(m)
RETURN n, r, m
```

### Delete Data

```cypher
-- Delete all nodes and relationships (clear database)
MATCH (n) DETACH DELETE n

-- Delete all data for a specific project
MATCH (n {project_id: "first_test"})
DETACH DELETE n

-- Delete by user_id and project_id
MATCH (n {user_id: "samgiam", project_id: "first_test"})
DETACH DELETE n
```

## Automatic Integration

When `UPDATE_GRAPH_DB = True`, the graph is automatically populated after `domain_discovery` with:

- **Domain** node (root) with WHOIS data
- **Subdomain** nodes
- **IP** nodes (from DNS resolution)
- **DNSRecord** nodes (TXT, MX, NS, etc.)
- All relationships between them

## Manual Usage

```python
from graph_db import Neo4jClient

with Neo4jClient() as client:
    # Load existing recon data
    import json
    with open("recon/output/recon_example.com.json") as f:
        recon_data = json.load(f)

    # Initialize graph
    stats = client.update_graph_from_domain_discovery(recon_data, "user_id", "project_id")
    print(stats)
```

## Standalone Graph Update Script

Use `update_graph_from_json.py` to run graph updates independently from the main pipeline:

```bash
# Run from project root
cd "/home/samuele/Progetti didattici/RedAmon"
python -m graph_db.update_graph_from_json

# Or run directly
python graph_db/update_graph_from_json.py
```

### Configuration

Edit the script to select which modules to run:

```python
# Run all modules (default)
UPDATE_MODULES = []

# Run specific modules only
UPDATE_MODULES = ["vuln_scan"]
UPDATE_MODULES = ["http_probe", "vuln_scan"]
UPDATE_MODULES = ["domain_discovery", "port_scan", "http_probe", "vuln_scan"]
```

### Available Update Modules

| Module | Creates | Relationships |
|--------|---------|---------------|
| `domain_discovery` | Domain, Subdomain, IP, DNSRecord | HAS_SUBDOMAIN, RESOLVES_TO, HAS_DNS_RECORD |
| `port_scan` | Port, Service | HAS_PORT, RUNS_SERVICE |
| `http_probe` | BaseURL, Technology, Header | SERVES_URL, USES_TECHNOLOGY, HAS_HEADER |
| `vuln_scan` | Endpoint, Parameter, Vulnerability | HAS_ENDPOINT, HAS_PARAMETER, HAS_VULNERABILITY, FOUND_AT, AFFECTS_PARAMETER |
| `gvm_scan` | Vulnerability, CVE | HAS_VULNERABILITY (from IP/Subdomain), HAS_CVE |

### Use Cases

- Re-import data after schema changes
- Update graph from existing JSON without re-running scans
- Run specific updates (e.g., only vuln_scan after adding new findings)
- Debug/test graph update functions

## Graph Schema

See [GRAPH.SCHEMA.md](../readmes/GRAPH.SCHEMA.md) for the complete schema documentation.

```
(Domain) <-[:BELONGS_TO]- (Subdomain) -[:RESOLVES_TO]-> (IP)
                               |                          |
                        [:HAS_DNS_RECORD]          [:HAS_VULNERABILITY]
                               |                          |
                               v                          v
                         (DNSRecord)              (Vulnerability)
```

### Vulnerability Node

The `:Vulnerability` node stores all security findings from multiple sources:

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier |
| `source` | string | `"nuclei"`, `"security_check"`, `"sqlmap"`, etc. |
| `type` | string | Vulnerability type (e.g., `"xss"`, `"direct_ip_http"`, `"waf_bypass"`) |
| `severity` | string | `"info"`, `"low"`, `"medium"`, `"high"`, `"critical"` |
| `name` | string | Human-readable title |
| `description` | string | Detailed finding description |
| `url` | string | Affected URL |
| `matched_at` | string | Exact match location |
| `matched_ip` | string | IP address (for IP-based findings) |
| `template_id` | string | Nuclei template ID (null for security checks) |
| `evidence` | string | Proof/extracted data |
| `is_dast_finding` | boolean | True if from DAST fuzzing |

**Vulnerability Sources:**

| Source | Description | Connected To |
|--------|-------------|--------------|
| `nuclei` | Nuclei scanner findings (DAST) | `:BaseURL`, `:Endpoint`, `:Parameter` |
| `security_check` | Custom security checks | `:IP`, `:Subdomain` |
| `gvm` | GVM/OpenVAS infrastructure scan | `:IP`, `:Subdomain`, `:CVE` |

**Security Check Types:**

| Type | Severity | Description |
|------|----------|-------------|
| `direct_ip_http` | medium | HTTP accessible directly via IP without TLS |
| `direct_ip_https` | low | HTTPS accessible directly via IP |
| `ip_api_exposed` | high | API endpoint exposed on IP without TLS |
| `waf_bypass` | high | WAF bypass via direct IP access |
| `tls_mismatch` | medium | TLS certificate mismatch |

**Relationships:**

```cypher
-- Nuclei findings (from vuln_scan)
(:BaseURL)-[:HAS_VULNERABILITY]->(:Vulnerability)
(:Vulnerability)-[:FOUND_AT]->(:Endpoint)
(:Vulnerability)-[:AFFECTS_PARAMETER]->(:Parameter)

-- Security check findings (direct IP access)
(:IP)-[:HAS_VULNERABILITY]->(:Vulnerability)

-- WAF bypass findings
(:Subdomain)-[:HAS_VULNERABILITY]->(:Vulnerability)
(:Subdomain)-[:WAF_BYPASS_VIA]->(:IP)
```

### Query Examples

```cypher
-- Find all security check vulnerabilities
MATCH (v:Vulnerability {source: "security_check"})
RETURN v.type, v.severity, v.name, v.url

-- Find WAF bypass opportunities
MATCH (s:Subdomain)-[:WAF_BYPASS_VIA]->(i:IP)
RETURN s.name AS subdomain, i.address AS bypass_ip

-- Find all vulnerabilities for an IP
MATCH (i:IP {address: "44.228.249.3"})-[:HAS_VULNERABILITY]->(v:Vulnerability)
RETURN v.type, v.severity, v.description

-- Find high/critical vulnerabilities by source
MATCH (v:Vulnerability)
WHERE v.severity IN ["high", "critical"]
RETURN v.source, v.type, v.name, v.url
ORDER BY v.severity DESC
```

## GVM/OpenVAS Vulnerability Integration

The `gvm_scan` module imports vulnerability findings from GVM/OpenVAS infrastructure scans into the graph.

### GVM Vulnerability Node Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique ID: `gvm-{oid}-{target_ip}-{port}` |
| `oid` | string | GVM OID (e.g., `1.3.6.1.4.1.25623.1.0.117318`) |
| `name` | string | Vulnerability name |
| `severity` | string | `low`, `medium`, `high`, `critical` |
| `cvss_score` | float | CVSS score (e.g., 4.3) |
| `cvss_vector` | string | CVSS vector string |
| `threat` | string | GVM threat level |
| `description` | string | Detailed description |
| `solution` | string | Remediation advice |
| `solution_type` | string | Solution type (e.g., `Mitigation`) |
| `target_ip` | string | Affected IP address |
| `target_port` | integer | Affected port |
| `target_hostname` | string | Affected hostname |
| `family` | string | Vulnerability family (e.g., `SSL and TLS`) |
| `qod` | integer | Quality of Detection (0-100%) |
| `cve_ids` | array | List of related CVE IDs |
| `source` | string | Always `gvm` |
| `scanner` | string | Always `OpenVAS` |
| `scan_timestamp` | string | When the scan was performed |

### GVM Relationships

```cypher
-- GVM vulnerabilities linked to IP
(:IP)-[:HAS_VULNERABILITY]->(:Vulnerability {source: "gvm"})

-- GVM vulnerabilities linked to Subdomain (if hostname matches)
(:Subdomain)-[:HAS_VULNERABILITY]->(:Vulnerability {source: "gvm"})

-- GVM vulnerability to CVE enrichment
(:Vulnerability {source: "gvm"})-[:HAS_CVE]->(:CVE)

-- CVE to CWE/CAPEC chain (shared with technology_cves)
(:CVE)-[:HAS_CWE]->(:MitreData)-[:HAS_CAPEC]->(:Capec)
```

### GVM Graph Visualization

```
     IP / Subdomain
           |
     HAS_VULNERABILITY
           |
           v
    ┌─────────────────┐
    │  Vulnerability  │  (source: "gvm")
    │  OID, severity  │
    └────────┬────────┘
             │ HAS_CVE
    ┌────────▼────────┐
    │      CVE        │  (shared with technology_cves)
    │ CVE-2011-3389   │
    └────────┬────────┘
             │ HAS_CWE
    ┌────────▼────────┐
    │    MitreData    │
    │    CWE-327      │
    └────────┬────────┘
             │ HAS_CAPEC
    ┌────────▼────────┐
    │     Capec       │
    │   CAPEC-217     │
    └─────────────────┘
```

### GVM Query Examples

```cypher
-- Find all GVM vulnerabilities
MATCH (v:Vulnerability {source: "gvm"})
RETURN v.name, v.severity, v.target_ip, v.target_port
ORDER BY v.cvss_score DESC

-- Find GVM vulnerabilities with CVE enrichment
MATCH (v:Vulnerability {source: "gvm"})-[:HAS_CVE]->(c:CVE)
RETURN v.name, v.severity, collect(c.id) AS cves

-- Find all vulnerabilities affecting an IP (any source)
MATCH (i:IP {address: "44.228.249.3"})-[:HAS_VULNERABILITY]->(v:Vulnerability)
RETURN v.source, v.name, v.severity

-- Find GVM vulnerabilities with full MITRE chain
MATCH (v:Vulnerability {source: "gvm"})-[:HAS_CVE]->(c:CVE)-[:HAS_CWE]->(cwe:MitreData)-[:HAS_CAPEC]->(cap:Capec)
RETURN v.name, c.id AS cve, cwe.cwe_id AS cwe, cap.capec_id AS capec

-- Count vulnerabilities by source
MATCH (v:Vulnerability)
RETURN v.source, count(v) AS count,
       collect(DISTINCT v.severity) AS severities
ORDER BY count DESC
```

### GVM Data File

GVM scan results are stored in:
```
gvm_scan/output/gvm_{target_domain}.json
```

The script `update_graph_from_json.py` automatically loads this file when running the `gvm_scan` module.

## Troubleshooting

```bash
# Check if Neo4j is running
docker compose ps

# Check logs for errors
docker compose logs --tail 100

# Verify connection from Python
python -c "from graph_db import Neo4jClient; c = Neo4jClient(); print('OK' if c.verify_connection() else 'FAIL'); c.close()"

# Reset database (delete all data)
docker compose down -v && docker compose up -d
```

## Requirements

```bash
pip install neo4j
```
