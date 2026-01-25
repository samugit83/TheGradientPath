# RedAmon - Nuclei Vulnerability Scanner

## Complete Technical Documentation

> **Module:** `recon/nuclei_scan.py`  
> **Purpose:** Template-based web application vulnerability scanning using ProjectDiscovery's Nuclei  
> **Author:** RedAmon Security Suite

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [CVE Lookup (Technology-Based)](#cve-lookup-technology-based)
4. [Installation](#installation)
5. [Configuration Parameters](#configuration-parameters)
6. [Architecture & Flow](#architecture--flow)
7. [Function Reference](#function-reference)
8. [Nuclei Arguments Explained](#nuclei-arguments-explained)
9. [Template Categories](#template-categories)
10. [Output Data Structure](#output-data-structure)
11. [Nmap vs Nuclei Comparison](#nmap-vs-nuclei-comparison)
12. [Usage Examples](#usage-examples)
13. [Troubleshooting](#troubleshooting)

---

## Overview

The `nuclei_scan.py` module integrates ProjectDiscovery's Nuclei scanner into RedAmon's reconnaissance pipeline. Nuclei is a fast, template-based vulnerability scanner that excels at web application security testing.

**⚠️ Important:** Nuclei runs exclusively via Docker. No native installation is supported.

### Why Nuclei?

| Feature | Nmap NSE | Nuclei |
|---------|----------|--------|
| Templates | ~150 scripts | **8,000+ templates** |
| Update Frequency | With nmap releases | **Daily community updates** |
| Web App Depth | Surface-level | **Deep application testing** |
| CVE Coverage | Older CVEs | **Recent CVEs (within days)** |
| Focus | Network layer | **Application layer** |

### How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Recon Data     │────▶│  nuclei_scan.py  │────▶│  Enriched JSON  │
│  (hostnames,    │     │                  │     │  with web vulns,│
│   IPs, ports)   │     │  1. Build URLs   │     │  CVEs, exposed  │
└─────────────────┘     │  2. Run Katana   │     │  panels, etc.   │
                        │  3. Run Nuclei   │     └─────────────────┘
        ┌───────────────│  4. Parse JSONL  │
        │               │  5. Classify     │
        ▼               └──────────────────┘
┌─────────────────┐
│   Nmap Data     │  (Used to discover HTTP/HTTPS ports)
│   (optional)    │
└─────────────────┘
```

### DAST Mode Pipeline

When `NUCLEI_DAST_MODE = True`, Katana crawler is integrated:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Base URLs  │────▶│   KATANA    │────▶│   NUCLEI    │────▶│   Results   │
│             │     │  (Crawler)  │     │  (Scanner)  │     │             │
│ example.com │     │             │     │             │     │ • XSS: 5    │
│             │     │ Discovers:  │     │ -dast mode  │     │ • SQLi: 2   │
│             │     │ ?param URLs │     │ fuzzes all  │     │ • SSTI: 1   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**Why Katana is needed for DAST:**
- Nuclei DAST templates inject payloads into URL parameters
- Without parameters (`?key=value`), DAST finds nothing
- Katana discovers these parameters by crawling the site
- JavaScript parsing finds hidden API endpoints

---

## Features

| Feature | Description |
|---------|-------------|
| **9000+ Templates** | CVEs, misconfigs, exposures, takeovers, default logins |
| **CVE Detection** | Automatic CVE ID and CVSS extraction |
| **Smart URL Building** | Uses nmap data to discover HTTP/HTTPS ports |
| **Katana Integration** | Web crawler discovers URLs with parameters for DAST |
| **DAST Mode** | Active fuzzing for XSS, SQLi, SSTI, Command Injection |
| **Auto-Update Templates** | Automatic template updates before each scan |
| **Category Classification** | Auto-categorizes findings (XSS, SQLi, RCE, etc.) |
| **Tor Integration** | Anonymous scanning via SOCKS proxy |
| **Authenticated Crawling** | Custom headers support for login-protected pages |
| **Incremental Saving** | Results saved progressively |
| **CVE Lookup** | Technology-based CVE lookup (like Nmap's vulners) |

---

## CVE Lookup (Technology-Based)

The Nuclei module includes an integrated CVE lookup feature that queries the NVD (National Vulnerability Database) for known vulnerabilities based on technologies detected by httpx. This replicates what Nmap's `vulners` script does.

### How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  httpx detects  │────▶│  CVE Lookup      │────▶│  technology_cves│
│  technologies:  │     │                  │     │  in JSON output │
│  • Nginx:1.19.0 │     │  Query NVD API   │     │                 │
│  • PHP:5.6.40   │     │  for each tech   │     │  23 CVEs found  │
│  • jQuery:3.5.1 │     │  with version    │     │  2 CRITICAL     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Configuration

```python
# params.py - CVE Lookup Configuration
# =============================================================================

# Enable/disable technology-based CVE lookup
CVE_LOOKUP_ENABLED = True

# Data source: "nvd" (free, rate limited) or "vulners" (needs API key)
CVE_LOOKUP_SOURCE = "nvd"

# Maximum CVEs to return per technology
CVE_LOOKUP_MAX_CVES = 20

# Minimum CVSS score to include (0.0 = all, 4.0 = medium+, 7.0 = high+)
CVE_LOOKUP_MIN_CVSS = 0.0

# Vulners API key (optional - for better results with vulners source)
# Get free API key at: https://vulners.com/
VULNERS_API_KEY = ""
```

### Supported Technologies

The CVE lookup supports 40+ common technologies with proper CPE (Common Platform Enumeration) mappings:

| Category | Technologies |
|----------|--------------|
| **Web Servers** | nginx, Apache, IIS, Tomcat, Lighttpd, Caddy |
| **Languages** | PHP, Python, Node.js, Ruby, Java |
| **Databases** | MySQL, MariaDB, PostgreSQL, MongoDB, Redis, Elasticsearch |
| **CMS/Frameworks** | WordPress, Drupal, Joomla, Django, Laravel, Spring |
| **JavaScript** | jQuery, Angular, React, Vue, Bootstrap |
| **Security** | OpenSSH, OpenSSL |
| **Other** | Varnish, HAProxy, Grafana, Jenkins, GitLab |

### Example Output

When scanning `testphp.vulnweb.com`:

```
============================================================
CVE LOOKUP - Technology-Based Vulnerability Discovery
============================================================
    Source: NVD
    Min CVSS: 0.0

[*] Technologies with versions: 3
    [1/3] Nginx:1.19.0... ✓ 6 CVEs found
    [2/3] PHP:5.6.40... ✓ 17 CVEs found
    [3/3] nginx/1.19.0... ✓ 6 CVEs found

[+] CVE LOOKUP SUMMARY:
    Total unique CVEs: 23
    🔴 CRITICAL: 2
    🟠 HIGH: 10
    🟡 MEDIUM: 10
============================================================
```

### JSON Output Structure

```json
{
  "technology_cves": {
    "lookup_timestamp": "2025-12-31T14:30:00.000000",
    "source": "nvd",
    "technologies_checked": 3,
    "technologies_with_cves": 3,
    "by_technology": {
      "Nginx:1.19.0": {
        "technology": "Nginx:1.19.0",
        "product": "nginx",
        "version": "1.19.0",
        "cve_count": 6,
        "critical": 0,
        "high": 5,
        "cves": [
          {
            "id": "CVE-2021-23017",
            "cvss": 7.7,
            "severity": "HIGH",
            "description": "A security issue in nginx resolver...",
            "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-23017",
            "source": "nvd"
          }
        ]
      },
      "PHP:5.6.40": {
        "technology": "PHP:5.6.40",
        "product": "php",
        "version": "5.6.40",
        "cve_count": 17,
        "critical": 2,
        "high": 5,
        "cves": [...]
      }
    },
    "all_cves": [
      {"id": "CVE-2017-8923", "cvss": 9.8, "severity": "CRITICAL", ...},
      {"id": "CVE-2019-9641", "cvss": 9.8, "severity": "CRITICAL", ...},
      ...
    ],
    "summary": {
      "total_cves": 23,
      "critical": 2,
      "high": 10,
      "medium": 10,
      "low": 1
    }
  }
}
```

### Nuclei Template CVEs vs Technology CVE Lookup

| Source | What It Detects | Type |
|--------|-----------------|------|
| **Nuclei Templates** | Specific CVEs with exploits/checks | **Confirmed exploitable** |
| **CVE Lookup** | All CVEs for detected version | **Potential vulnerabilities** |

**Example:**
- Nuclei template for CVE-2021-23017 → Checks if nginx resolver is vulnerable → Confirmed if matched
- CVE Lookup for nginx 1.19.0 → Lists CVE-2021-23017 → May or may not be exploitable (depends on config)

**Both are valuable:**
- Nuclei = "This IS vulnerable"
- CVE Lookup = "This COULD be vulnerable based on version"

### Rate Limiting

The NVD API has rate limits:
- **Without API key:** 5 requests per 30 seconds
- **With API key:** 50 requests per 30 seconds

The script automatically adds delays between requests to avoid rate limiting.

### Vulners Integration (Optional)

For better results (like Nmap's vulners script), you can use the Vulners API:

1. Get a free API key at [vulners.com](https://vulners.com/)
2. Set in `params.py`:
   ```python
   CVE_LOOKUP_SOURCE = "vulners"
   VULNERS_API_KEY = "your-api-key-here"
   ```

Vulners provides:
- Faster responses
- More CVE data
- Same format as Nmap's vulners script

---

## Installation

### Requirements

- **Docker** installed and running
- That's it! Templates are included in the Docker image

### Setup

```bash
# Make sure Docker is running
sudo systemctl start docker

# Run the scan - image will be pulled automatically
python3 recon/main.py
```

The Python script handles everything:
- Pulls the Docker image automatically if needed
- Mounts targets and output directories
- Runs nuclei inside the container
- Parses results and updates your recon JSON

### Verify Docker is Ready

```bash
# Check Docker is running
docker info

# Optionally pre-pull the image
docker pull projectdiscovery/nuclei:latest
```

---

## Configuration Parameters

All parameters are defined in `params.py`:

### Severity Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUCLEI_SEVERITY` | `list` | `["critical", "high", "medium", "low"]` | Severity levels to include |

**Severity Levels:**
- `critical` - Remote code execution, authentication bypass
- `high` - SQL injection, XSS, SSRF
- `medium` - Information disclosure, misconfigurations
- `low` - Minor issues, informational findings
- `info` - Technology detection, version info (excluded by default)

### Template Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUCLEI_TEMPLATES` | `list` | `[]` | Specific template folders (empty = all) |
| `NUCLEI_EXCLUDE_TEMPLATES` | `list` | `["fuzzing"]` | Templates to exclude |
| `NUCLEI_CUSTOM_TEMPLATES` | `list` | `[]` | Paths to custom templates |

**Template Folders:**
```
cves/               - Known CVE exploits
vulnerabilities/    - General vulnerabilities
misconfiguration/   - Security misconfigurations
exposures/          - Exposed files/panels/data
technologies/       - Technology detection
default-logins/     - Default credentials
takeovers/          - Subdomain takeovers
file/               - Interesting files
fuzzing/            - Fuzzing templates (noisy)
```

### Tag Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUCLEI_TAGS` | `list` | `[]` | Template tags to include |
| `NUCLEI_EXCLUDE_TAGS` | `list` | `["dos"]` | Tags to exclude |

**Popular Tags:**
```
cve         - CVE templates
xss         - Cross-site scripting
sqli        - SQL injection
rce         - Remote code execution
lfi         - Local file inclusion
ssrf        - Server-side request forgery
xxe         - XML external entity
ssti        - Server-side template injection
exposure    - Data exposure
misconfig   - Misconfigurations
default-login - Default credentials
takeover    - Subdomain takeover
tech        - Technology detection
dos         - Denial of service (excluded by default)
fuzz        - Fuzzing templates
```

### Rate Limiting & Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUCLEI_RATE_LIMIT` | `int` | `100` | Requests per second |
| `NUCLEI_BULK_SIZE` | `int` | `25` | Hosts to process in parallel |
| `NUCLEI_CONCURRENCY` | `int` | `25` | Templates to run in parallel |
| `NUCLEI_TIMEOUT` | `int` | `10` | Request timeout (seconds) |
| `NUCLEI_RETRIES` | `int` | `1` | Retries for failed requests |

**Recommended Settings:**

| Scenario | Rate Limit | Bulk Size | Concurrency |
|----------|------------|-----------|-------------|
| **Safe/Slow** | 50 | 10 | 10 |
| **Normal** | 100 | 25 | 25 |
| **Aggressive** | 300 | 50 | 50 |
| **Internal Network** | 500+ | 100 | 100 |

### Scan Modes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUCLEI_DAST_MODE` | `bool` | `True` | Enable DAST mode for active fuzzing |
| `NUCLEI_AUTO_UPDATE_TEMPLATES` | `bool` | `True` | Auto-update templates before scan |
| `NUCLEI_NEW_TEMPLATES_ONLY` | `bool` | `False` | Only use newly added templates |
| `NUCLEI_HEADLESS` | `bool` | `False` | Enable headless browser for JS apps |

### Katana Web Crawler (DAST Mode)

When `NUCLEI_DAST_MODE` is enabled, **Katana** automatically crawls target websites to discover URLs with parameters for active fuzzing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `KATANA_DOCKER_IMAGE` | `str` | `"projectdiscovery/katana:latest"` | Docker image for Katana |
| `KATANA_DEPTH` | `int` | `3` | Maximum crawl depth (links to follow) |
| `KATANA_MAX_URLS` | `int` | `500` | Maximum URLs to discover |
| `KATANA_RATE_LIMIT` | `int` | `50` | Requests per second |
| `KATANA_TIMEOUT` | `int` | `300` | Crawl timeout (seconds) |
| `KATANA_JS_CRAWL` | `bool` | `True` | Parse JavaScript for URLs |
| `KATANA_PARAMS_ONLY` | `bool` | `True` | Only keep URLs with `?parameters` |
| `KATANA_SCOPE` | `str` | `"dn"` | Crawl scope: `dn`/`rdn`/`fqdn` |
| `KATANA_CUSTOM_HEADERS` | `list` | `[]` | Custom headers for auth crawling |

**Scope Options:**
- `dn` - Domain name (stays within same domain)
- `rdn` - Root domain (includes all subdomains)
- `fqdn` - Fully qualified domain name (exact hostname only)

**Authenticated Crawling:**
```python
# For sites requiring authentication
KATANA_CUSTOM_HEADERS = [
    "Cookie: session=abc123; token=xyz789",
    "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
]
```

### Network Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUCLEI_SYSTEM_RESOLVERS` | `bool` | `True` | Use system DNS resolvers |
| `NUCLEI_FOLLOW_REDIRECTS` | `bool` | `True` | Follow HTTP redirects |
| `NUCLEI_MAX_REDIRECTS` | `int` | `10` | Maximum redirects to follow |
| `NUCLEI_SCAN_ALL_IPS` | `bool` | `False` | Scan IPs in addition to hostnames |
| `NUCLEI_INTERACTSH` | `bool` | `True` | Enable OOB testing (blind vulns) |

### Docker Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUCLEI_DOCKER_IMAGE` | `str` | `"projectdiscovery/nuclei:latest"` | Docker image to use |

**Docker Mode Advantages:**
- No Go installation required
- No binary management
- Templates included in image
- Easy updates (image pulled automatically)
- Isolated execution

**How it works:**
The Python script automatically runs:
```bash
docker run --rm \
  -v /tmp/targets:/targets:ro \
  -v /tmp/output:/output \
  projectdiscovery/nuclei:latest \
  [nuclei arguments...]
```

---

## Architecture & Flow

### Execution Flow

```
1. INITIALIZATION
   └── Check Docker availability
   └── Pull nuclei image if needed
   └── Ensure templates volume exists
   └── Auto-update templates (if enabled)
   └── Check Tor availability

2. TARGET EXTRACTION
   └── Parse recon_data JSON
   └── Extract unique IPs and hostnames
   └── Build IP-to-hostname mapping

3. URL BUILDING
   └── Generate HTTP/HTTPS URLs for hostnames
   └── Use nmap data for non-standard ports
   └── Optionally include IP-based URLs

4. KATANA CRAWLING (if DAST mode enabled)
   └── Pull Katana Docker image
   └── Crawl each base URL
   └── Parse JavaScript for hidden URLs
   └── Filter URLs with parameters (?key=value)
   └── Return discovered URLs for fuzzing

5. SCAN EXECUTION
   └── Create temporary targets file (base URLs + Katana URLs)
   └── Build nuclei Docker command with all parameters
   └── Add -dast flag for active fuzzing
   └── Execute nuclei container with JSONL output
   └── Monitor progress

6. RESULT PROCESSING
   └── Parse JSONL output line by line
   └── Extract CVEs from findings
   └── Classify by category and severity
   └── Aggregate statistics

7. DATA ENRICHMENT
   └── Add "nuclei" section to recon_data
   └── Include discovered_urls section
   └── Save incrementally to JSON file
   └── Generate summary statistics
```

### DAST Mode with Katana Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DAST MODE WORKFLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  Base URLs   │───▶│   KATANA     │───▶│   NUCLEI     │───▶│  Results  │ │
│  │              │    │  (Crawler)   │    │  (Scanner)   │    │           │ │
│  │ example.com  │    │              │    │              │    │ SQLi: 2   │ │
│  │ api.example  │    │ Discovers:   │    │ Fuzzes with: │    │ XSS: 5    │ │
│  │              │    │ /search?q=   │    │ -dast flag   │    │ RCE: 1    │ │
│  │              │    │ /api?id=     │    │              │    │           │ │
│  │              │    │ /user?name=  │    │ Payloads:    │    │           │ │
│  └──────────────┘    │ /login?user= │    │ ' OR 1=1--   │    └───────────┘ │
│                      │              │    │ <script>     │                   │
│                      └──────────────┘    │ {{7*7}}      │                   │
│                             │            └──────────────┘                   │
│                             │                   │                           │
│                             ▼                   ▼                           │
│                   ┌─────────────────────────────────────┐                  │
│                   │        discovered_urls JSON         │                  │
│                   │  • base_urls: [example.com, ...]    │                  │
│                   │  • dast_urls_with_params: [...]     │                  │
│                   │  • all_scanned_urls: [...]          │                  │
│                   └─────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Katana for DAST?

| Problem | Solution with Katana |
|---------|---------------------|
| DAST needs URLs with parameters | Katana discovers `?param=value` URLs |
| JavaScript hides endpoints | Katana parses JS files for URLs |
| Forms have hidden inputs | Katana extracts form action URLs |
| APIs have undocumented endpoints | Katana follows links recursively |
| Auth required for some pages | Custom headers support authentication |

### Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              params.py                                    │
│  NUCLEI_SEVERITY, NUCLEI_RATE_LIMIT, NUCLEI_TAGS, NUCLEI_DOCKER_IMAGE... │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          nuclei_scan.py                                   │
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐  │
│  │ extract_targets │───▶│ build_target    │───▶│ build_nuclei        │  │
│  │ _from_recon()   │    │ _urls()         │    │ _command()          │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘  │
│           │                     │                        │               │
│           ▼                     ▼                        ▼               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    run_nuclei_scan()                             │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │    │
│  │  │ Docker        │─▶│ parse_nuclei  │─▶│ aggregate         │   │    │
│  │  │ (nuclei CLI)  │  │ _finding()    │  │ results           │   │    │
│  │  └───────────────┘  └───────────────┘  └───────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   recon_<domain>.json    │
                    │   with "nuclei" section  │
                    └──────────────────────────┘
```

---

## Function Reference

### `is_docker_installed() -> bool`

**Purpose:** Checks if Docker is available in PATH.

**Returns:** `True` if `docker` command is found.

---

### `is_docker_running() -> bool`

**Purpose:** Checks if Docker daemon is running.

**Mechanism:** Runs `docker info` and checks return code.

**Returns:** `True` if Docker daemon is accessible.

---

### `pull_nuclei_docker_image() -> bool`

**Purpose:** Pulls the nuclei Docker image if not present.

**Returns:** `True` if pull succeeded.

---

### `extract_targets_from_recon(recon_data: dict) -> Tuple[Set[str], Set[str], Dict[str, List[str]]]`

**Purpose:** Extracts IPs, hostnames, and mappings from recon data.

**Identical to nmap_scan.py implementation** - extracts from:
- Root domain DNS records
- Subdomain DNS records

**Returns:** `(ips, hostnames, ip_to_hostnames)`

---

### `build_target_urls(hostnames: Set[str], ips: Set[str], nmap_data: Optional[dict]) -> List[str]`

**Purpose:** Builds list of target URLs for nuclei.

**Logic:**
```python
# For each hostname:
urls.append(f"http://{hostname}")
urls.append(f"https://{hostname}")

# Add non-standard ports from nmap data:
# e.g., http://example.com:8080, https://example.com:8443

# If NUCLEI_SCAN_ALL_IPS:
#   Also add IP-based URLs
```

**Port Detection:**
- Default HTTP: 80, 8080, 8000, 8888
- Default HTTPS: 443, 8443, 4443, 9443
- Additional ports from nmap service detection

---

### `build_nuclei_command(targets_file: str, output_file: str, use_proxy: bool) -> List[str]`

**Purpose:** Constructs the nuclei Docker CLI command.

**Generated Command Structure:**
```bash
docker run --rm \
  -v /targets:/targets:ro \
  -v /output:/output \
  projectdiscovery/nuclei:latest \
  -l /targets/targets.txt \
  -jsonl \
  -o /output/output.jsonl \
  -silent \
  -nc \
  -severity critical,high,medium,low \
  -t cves/ -t vulnerabilities/ \
  -exclude-templates fuzzing/ \
  -tags cve,xss,sqli \
  -exclude-tags dos \
  -rate-limit 100 \
  -bulk-size 25 \
  -concurrency 25 \
  -timeout 10 \
  -retries 1 \
  -as \                      # Automatic scan
  -system-resolvers \
  -follow-redirects \
  -max-redirects 10 \
  -proxy socks5://127.0.0.1:9050  # If Tor enabled
```

---

### `parse_nuclei_finding(finding: dict) -> dict`

**Purpose:** Parses a single nuclei JSON finding into standardized format.

**CVE Extraction Sources:**
1. `info.classification.cve-id`
2. `info.classification.cve`

**Category Classification:**

| Tags Containing | Category |
|-----------------|----------|
| `xss` | xss |
| `sqli` | sqli |
| `rce` | rce |
| `lfi` | lfi |
| `ssrf` | ssrf |
| `xxe` | xxe |
| `ssti` | ssti |
| `cve` | cve |
| `exposure` | exposure |
| `misconfig` | misconfiguration |
| `default-login` | authentication |
| `panel` | exposed_panel |
| `takeover` | takeover |
| `ssl`, `tls` | ssl_tls |
| `cloud`, `aws`, `azure`, `gcp` | cloud |

**Returns:**
```python
{
    "template_id": "CVE-2021-44228",
    "name": "Apache Log4j RCE",
    "description": "...",
    "severity": "critical",
    "category": "cve",
    "tags": ["cve", "rce", "log4j"],
    "cves": [{"id": "CVE-2021-44228", "cvss": 10.0, "url": "..."}],
    "target": "https://example.com",
    "matched_at": "https://example.com/api/login",
    "curl_command": "curl ...",
    ...
}
```

---

### `run_nuclei_scan(recon_data: dict, output_file: Path = None) -> dict`

**Purpose:** Main orchestrator function for nuclei scanning.

**Phases:**
1. Check Docker availability
2. Pull nuclei image if needed
3. Ensure templates volume exists (with auto-update if enabled)
4. Check Tor availability
5. Extract targets from recon data
6. Build target URLs (using nmap data if available)
7. **Run Katana crawler** (if DAST mode enabled)
8. Create temporary files (base URLs + Katana discovered URLs)
9. Execute nuclei Docker container with -dast flag
10. Parse JSONL results
11. Aggregate and classify findings
12. Add to recon_data["nuclei"] including discovered_urls
13. Save incrementally

**Returns:** Enriched `recon_data` with `nuclei` section.

---

### `pull_katana_docker_image() -> bool`

**Purpose:** Pulls the Katana Docker image if not already present.

**Returns:** `True` if pull succeeded or image already exists.

---

### `run_katana_crawler(target_urls: List[str], use_proxy: bool = False) -> List[str]`

**Purpose:** Run Katana web crawler to discover URLs with parameters for DAST fuzzing.

**Args:**
- `target_urls`: Base URLs to start crawling from
- `use_proxy`: Whether to use Tor proxy

**Behavior:**
1. Pulls Katana Docker image
2. For each base URL, runs Katana with configured depth/rate
3. Parses JavaScript files for hidden endpoints
4. Filters URLs to only those with `?parameters`
5. Returns deduplicated list of discovered URLs

**Docker Command Generated:**
```bash
docker run --rm \
  projectdiscovery/katana:latest \
  -u https://example.com \
  -d 3 \                    # Crawl depth
  -silent \
  -nc \
  -rl 50 \                  # Rate limit
  -timeout 300 \
  -fs dn \                  # Scope
  -murl 500 \               # Max URLs
  -jc \                     # JavaScript crawl
  -H "Cookie: session=..." \ # Custom headers (optional)
  -proxy socks5://127.0.0.1:9050  # If Tor enabled
```

**Returns:** List of URLs with parameters (e.g., `["https://example.com/search?q=test"]`)

---

### `ensure_templates_volume() -> bool`

**Purpose:** Ensures the nuclei-templates Docker volume exists and has templates.

**Behavior:**
1. Checks if `nuclei-templates` Docker volume exists
2. Creates volume if missing
3. Checks if templates are downloaded (looks for .yaml files)
4. Downloads/updates templates if needed or if `NUCLEI_AUTO_UPDATE_TEMPLATES` is True
5. Templates persist across scans in the Docker volume

**Returns:** `True` if templates are ready.

---

## Updating Templates & Vulnerability Data

Nuclei's vulnerability detection is entirely based on YAML templates maintained by the community. Understanding how templates are updated is critical for effective scanning.

### How Nuclei Gets CVE/CVSS Data

Nuclei does **NOT calculate CVSS scores** - template authors embed pre-calculated scores from NVD:

```yaml
# Example template structure
info:
  name: "Apache Log4j RCE"
  severity: critical
  classification:
    cve-id: CVE-2021-44228
    cvss-score: 10.0                    # From NVD
    cvss-metrics: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H
```

| Data Field | Source | When Added |
|------------|--------|------------|
| `cve-id` | Template author | When template created |
| `cvss-score` | NIST NVD (copied by author) | When template created |
| `cvss-metrics` | NIST NVD (copied by author) | When template created |
| `severity` | Nuclei classification | Based on CVSS |

### Template Update Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     NUCLEI TEMPLATE UPDATE FLOW                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PROJECTDISCOVERY                      YOUR SYSTEM                     │
│   ┌─────────────────┐                   ┌─────────────────┐            │
│   │ nuclei-templates│                   │ Docker Volume   │            │
│   │    (GitHub)     │                   │ nuclei-templates│            │
│   │                 │   nuclei -ut      │                 │            │
│   │  9000+ YAML     │ ───────────────▶  │  Local copy of  │            │
│   │  templates      │                   │  all templates  │            │
│   │                 │                   │                 │            │
│   │ Updated daily   │                   │ Persists across │            │
│   │ by community    │                   │ scans           │            │
│   └─────────────────┘                   └─────────────────┘            │
│                                                  │                      │
│                                                  ▼                      │
│                                         ┌─────────────────┐            │
│                                         │  Nuclei Scan    │            │
│                                         │  Uses templates │            │
│                                         │  to detect CVEs │            │
│                                         └─────────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Automatic Template Updates

RedAmon automatically updates templates when `NUCLEI_AUTO_UPDATE_TEMPLATES = True` (default):

```python
# In params.py
NUCLEI_AUTO_UPDATE_TEMPLATES = True  # Runs "nuclei -ut" before each scan
```

**What happens on each scan:**
1. Check if `nuclei-templates` Docker volume exists
2. If `NUCLEI_AUTO_UPDATE_TEMPLATES = True`, run `nuclei -ut`
3. Templates are downloaded/updated in the persistent volume
4. Scan proceeds with latest templates

**Console output when updating:**
```
[*] Checking for template updates...
[✓] Templates updated successfully
```

### Manual Template Update

Force a template update without running a scan:

```bash
# Update templates via Docker
docker run --rm \
  -v nuclei-templates:/root/nuclei-templates \
  projectdiscovery/nuclei:latest \
  -ut

# Or delete the volume to force fresh download
docker volume rm nuclei-templates
# Next scan will download all templates fresh
```

### Template Update Frequency

| Source | Update Frequency | Coverage |
|--------|------------------|----------|
| **ProjectDiscovery GitHub** | Multiple times daily | Official templates |
| **Community contributions** | Continuous | New CVEs within hours/days |
| **Your local volume** | Per scan (if auto-update on) | Synced on demand |

**Recent template releases (example):**
- November 2025: 197 new templates, 83 CVEs
- October 2025: 243 new templates (Hacktoberfest)
- New 2025 CVEs like CVE-2025-53677, CVE-2025-3515

### What Gets Updated

| Template Category | Contents | Count |
|-------------------|----------|-------|
| `cves/` | CVE-specific exploit templates | ~3,500 |
| `vulnerabilities/` | Generic vulnerability checks | ~2,000 |
| `misconfiguration/` | Security misconfigs | ~1,500 |
| `exposures/` | Exposed files/panels | ~1,000 |
| `technologies/` | Tech detection | ~500 |
| `default-logins/` | Default credentials | ~200 |
| `dast/` | Active fuzzing (XSS, SQLi) | ~100 |
| **Total** | | **~9,000+** |

### Disabling Auto-Update

For faster scan startup (skip update check):

```python
# In params.py
NUCLEI_AUTO_UPDATE_TEMPLATES = False
```

**Console output when disabled:**
```
[✓] Templates volume ready (auto-update disabled)
```

### Verifying Template Status

```bash
# Check templates in volume
docker run --rm \
  -v nuclei-templates:/root/nuclei-templates \
  alpine \
  sh -c "find /root/nuclei-templates -name '*.yaml' | wc -l"
# Expected: ~9000+

# Check template version
docker run --rm \
  -v nuclei-templates:/root/nuclei-templates \
  projectdiscovery/nuclei:latest \
  -version
```

### Troubleshooting Template Updates

| Issue | Solution |
|-------|----------|
| "Templates not updating" | Delete volume: `docker volume rm nuclei-templates` |
| "Template parse error" | Update Nuclei image: `docker pull projectdiscovery/nuclei:latest` |
| "Breaking changes" | Nuclei engine version must match template requirements |
| Slow updates | Normal on first run (~2-5 min for 9000+ templates) |

---

## Nuclei Arguments Explained

### Output Formats

| Flag | Description |
|------|-------------|
| `-jsonl` | JSON Lines output (one JSON per line) |
| `-o <file>` | Output file path |
| `-silent` | Suppress banner and progress |
| `-nc` | No color output |

### Target Selection

| Flag | Description |
|------|-------------|
| `-l <file>` | List of targets from file |
| `-u <url>` | Single target URL |
| `-eh` | Exclude hosts |

### Template Selection

| Flag | Description |
|------|-------------|
| `-t <path>` | Template folder or file |
| `-exclude-templates <path>` | Exclude specific templates |
| `-tags <tags>` | Include templates with tags |
| `-exclude-tags <tags>` | Exclude templates with tags |
| `-severity <levels>` | Filter by severity |
| `-as` | Automatic scan (smart template selection) |
| `-nt` | New templates only |

### Rate Limiting

| Flag | Description |
|------|-------------|
| `-rate-limit <n>` | Maximum requests per second |
| `-bulk-size <n>` | Hosts to scan in parallel |
| `-concurrency <n>` | Templates to run in parallel |

### Network Options

| Flag | Description |
|------|-------------|
| `-timeout <s>` | Request timeout in seconds |
| `-retries <n>` | Number of retries |
| `-proxy <url>` | HTTP/SOCKS proxy |
| `-system-resolvers` | Use system DNS |
| `-follow-redirects` | Follow HTTP redirects |
| `-max-redirects <n>` | Max redirect depth |

### Advanced Features

| Flag | Description |
|------|-------------|
| `-headless` | Enable headless browser |
| `-no-interactsh` | Disable OOB testing |
| `-H <header>` | Custom HTTP headers |
| `-var <var=val>` | Template variables |

---

## Template Categories

### CVE Templates (`cves/`)

```
cves/
├── 2024/
│   ├── CVE-2024-XXXX.yaml
│   └── ...
├── 2023/
├── 2022/
└── ...
```

**Coverage:** 3000+ CVE templates updated within days of disclosure.

### Vulnerability Templates (`vulnerabilities/`)

```
vulnerabilities/
├── generic/
│   ├── basic-xss.yaml
│   ├── sqli-error-based.yaml
│   └── ...
├── wordpress/
├── joomla/
├── drupal/
└── ...
```

### Misconfiguration Templates (`misconfiguration/`)

```
misconfiguration/
├── apache/
├── nginx/
├── cors/
├── headers/
└── ...
```

### Exposure Templates (`exposures/`)

```
exposures/
├── configs/
│   ├── git-config.yaml
│   ├── env-file.yaml
│   └── ...
├── files/
├── logs/
└── ...
```

### Technology Templates (`technologies/`)

```
technologies/
├── tech-detect.yaml
├── waf-detect/
├── cms-detect/
└── ...
```

---

## Output Data Structure

### Complete JSON Schema

```json
{
  "nuclei": {
    "scan_metadata": {
      "scan_timestamp": "2024-01-15T12:00:00.000000",
      "scan_duration_seconds": 245.5,
      "nuclei_version": "Docker: projectdiscovery/nuclei:latest",
      "templates_available": 8000,
      "execution_mode": "docker",
      "docker_image": "projectdiscovery/nuclei:latest",
      "anonymous_mode": false,
      "severity_filter": ["critical", "high", "medium", "low"],
      "tags_filter": [],
      "exclude_tags": ["dos", "ddos"],
      "rate_limit": 100,
      "dast_mode": true,
      "dast_urls_discovered": 47,
      "katana_crawl_depth": 3,
      "katana_js_crawl": true,
      "katana_params_only": true,
      "katana_scope": "dn",
      "total_urls_scanned": 52,
      "total_hostnames": 5,
      "total_ips": 2
    },
    
    "discovered_urls": {
      "base_urls": [
        "http://example.com",
        "https://example.com",
        "http://api.example.com"
      ],
      "dast_urls_with_params": [
        "http://example.com/search?q=test",
        "http://example.com/api/users?id=1",
        "http://example.com/login?redirect=/dashboard",
        "http://example.com/product?category=books&sort=price"
      ],
      "all_scanned_urls": [
        "http://example.com",
        "https://example.com",
        "http://example.com/search?q=test",
        "..."
      ]
    },
    
    "by_target": {
      "https://example.com": {
        "findings": [
          {
            "template_id": "CVE-2021-44228",
            "name": "Apache Log4j RCE",
            "description": "Remote code execution via JNDI injection",
            "severity": "critical",
            "category": "cve",
            "tags": ["cve", "rce", "log4j", "jndi"],
            "cves": [
              {
                "id": "CVE-2021-44228",
                "cvss": 10.0,
                "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228"
              }
            ],
            "cvss_score": 10.0,
            "target": "https://example.com",
            "matched_at": "https://example.com/api/v1/login",
            "curl_command": "curl -X POST ...",
            "request": "POST /api/v1/login HTTP/1.1...",
            "response": "HTTP/1.1 200 OK...",
            "timestamp": "2024-01-15T12:05:00.000000"
          }
        ],
        "severity_counts": {
          "critical": 1,
          "high": 2,
          "medium": 5,
          "low": 3,
          "info": 10
        }
      }
    },
    
    "summary": {
      "total_findings": 50,
      "critical": 1,
      "high": 5,
      "medium": 15,
      "low": 10,
      "info": 19,
      "unknown": 0
    },
    
    "vulnerabilities": {
      "total": 31,
      "critical": [
        {
          "template_id": "CVE-2021-44228",
          "name": "Apache Log4j RCE",
          "target": "https://example.com",
          "matched_at": "https://example.com/api/v1/login",
          "category": "cve",
          "cves": ["CVE-2021-44228"],
          "cvss": 10.0
        }
      ],
      "high": [...],
      "medium": [...],
      "low": [...],
      "info": [...],
      "unknown": []
    },
    
    "all_cves": [
      {
        "id": "CVE-2021-44228",
        "cvss": 10.0,
        "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228"
      },
      {
        "id": "CVE-2021-45046",
        "cvss": 9.0,
        "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-45046"
      }
    ],
    
    "by_category": {
      "cve": [...],
      "xss": [...],
      "sqli": [...],
      "misconfiguration": [...],
      "exposure": [...]
    },
    
    "by_template": {
      "CVE-2021-44228": {
        "name": "Apache Log4j RCE",
        "severity": "critical",
        "findings_count": 3,
        "targets": ["https://example.com", "https://api.example.com"]
      }
    }
  }
}
```

---

## Nmap vs Nuclei Comparison

### When to Use Each Tool

| Use Case | Nmap | Nuclei | Both |
|----------|------|--------|------|
| Port discovery | ✅ | ❌ | |
| Service detection | ✅ | ❌ | |
| OS fingerprinting | ✅ | ❌ | |
| Network protocol vulns | ✅ | ❌ | |
| SMB/FTP/SSH vulns | ✅ | ⚠️ | |
| Web application vulns | ⚠️ | ✅ | |
| CVE detection | ⚠️ | ✅ | |
| Exposed panels | ❌ | ✅ | |
| Default credentials | ⚠️ | ✅ | |
| Cloud misconfigs | ❌ | ✅ | |
| **Full assessment** | | | ✅ |

### Complementary Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE VULNERABILITY SCAN                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Nmap Scan                       2. Nuclei Scan              │
│  ─────────────                      ────────────                │
│  • Port discovery                   • Uses nmap ports data      │
│  • Service detection                • Web application CVEs      │
│  • OS fingerprinting                • Exposed panels/files      │
│  • Network vulns                    • Default credentials       │
│  • Banner grabbing                  • Cloud misconfigs          │
│                                     • Technology detection      │
│           │                                   │                 │
│           └──────────────┬───────────────────┘                 │
│                          ▼                                      │
│              ┌─────────────────────────┐                       │
│              │    Combined Results     │                       │
│              │   (recon_domain.json)   │                       │
│              └─────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Basic Usage (via main.py)

```python
# Simply include "nuclei" in SCAN_MODULES in params.py
SCAN_MODULES = ["initial_recon", "nmap", "nuclei"]

# Run the full pipeline
python3 recon/main.py
```

### Standalone Enrichment

```python
from nuclei_scan import enrich_recon_file
from pathlib import Path

enriched = enrich_recon_file(Path("output/recon_example.com.json"))
```

### Command Line

```bash
# Enrich an existing recon file
python3 recon/nuclei_scan.py output/recon_example.com.json
```

### Custom Configuration Profiles

**Quick Web Scan (Fast):**
```python
NUCLEI_SEVERITY = ["critical", "high"]
NUCLEI_RATE_LIMIT = 200
NUCLEI_TEMPLATES = []
NUCLEI_EXCLUDE_TEMPLATES = []
```

**Comprehensive Security Audit:**
```python
NUCLEI_SEVERITY = ["critical", "high", "medium", "low", "info"]
NUCLEI_RATE_LIMIT = 50
NUCLEI_HEADLESS = True
NUCLEI_INTERACTSH = True
```

**CVE-Only Scan:**
```python
NUCLEI_TEMPLATES = ["cves"]
NUCLEI_SEVERITY = ["critical", "high", "medium"]
NUCLEI_TAGS = []
```

**Web Application Pentest:**
```python
NUCLEI_TAGS = ["xss", "sqli", "rce", "lfi", "ssrf", "ssti"]
NUCLEI_SEVERITY = ["critical", "high", "medium"]
NUCLEI_HEADLESS = True
```

**DAST Mode with Katana (Active Fuzzing):**
```python
# Enable DAST mode - Katana will crawl for URLs with parameters
NUCLEI_DAST_MODE = True
NUCLEI_SEVERITY = ["critical", "high", "medium"]

# Katana crawler settings
KATANA_DEPTH = 3              # How deep to crawl
KATANA_MAX_URLS = 500         # Max URLs to discover
KATANA_JS_CRAWL = True        # Parse JavaScript files
KATANA_PARAMS_ONLY = True     # Only URLs with ?params

# For authenticated scanning (optional)
KATANA_CUSTOM_HEADERS = [
    "Cookie: session=your_session_cookie",
    "Authorization: Bearer your_jwt_token"
]
```

**DAST Expected Findings:**
- SQL Injection in search forms
- XSS in URL parameters
- Command injection in API endpoints
- SSTI in template parameters
- Path traversal in file parameters

### Anonymous Scanning

```bash
# Start Tor
sudo systemctl start tor

# In params.py
USE_TOR_FOR_RECON = True
```

---

## Troubleshooting

### Common Issues

#### "Docker not found"

```bash
# Install Docker
sudo apt install docker.io  # Debian/Ubuntu
# or
sudo pacman -S docker       # Arch

# Start Docker daemon
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (optional, avoids sudo)
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect
```

#### "Docker daemon is not running"

```bash
# Start Docker
sudo systemctl start docker

# Check status
sudo systemctl status docker
```

#### Scan too slow

```python
# Increase performance in params.py
NUCLEI_RATE_LIMIT = 300
NUCLEI_BULK_SIZE = 50
NUCLEI_CONCURRENCY = 50
NUCLEI_SEVERITY = ["critical", "high"]  # Skip lower severities
```

#### Too many false positives

```python
# Be more selective
NUCLEI_SEVERITY = ["critical", "high"]
NUCLEI_EXCLUDE_TAGS = ["dos", "fuzz", "intrusive"]
NUCLEI_EXCLUDE_TEMPLATES = ["fuzzing"]
```

#### Memory issues

```python
# Reduce parallel operations
NUCLEI_BULK_SIZE = 10
NUCLEI_CONCURRENCY = 10
```

#### Tor connection issues

```bash
# Check Tor status
sudo systemctl status tor

# Test SOCKS proxy
curl --socks5-hostname localhost:9050 https://check.torproject.org/api/ip
```

#### DAST mode finds nothing

**Problem:** `NUCLEI_DAST_MODE = True` but no XSS/SQLi findings.

**Cause:** DAST templates need URLs with parameters (`?key=value`) to fuzz.

**Solution:**
```python
# Make sure Katana is configured properly
NUCLEI_DAST_MODE = True
KATANA_DEPTH = 3              # Increase if site is deep
KATANA_MAX_URLS = 1000        # Increase for larger sites
KATANA_PARAMS_ONLY = True     # Must be True for DAST
KATANA_JS_CRAWL = True        # Parse JavaScript

# Check the discovered_urls in output JSON
# If dast_urls_with_params is empty, Katana didn't find parameterized URLs
```

**Manual test:**
```bash
# Test Katana directly
docker run --rm projectdiscovery/katana:latest \
  -u https://example.com -d 3 -jc -silent | grep "?"
```

#### Katana not finding URLs behind login

**Solution:** Use custom headers for authentication:
```python
KATANA_CUSTOM_HEADERS = [
    "Cookie: session=abc123; PHPSESSID=xyz789",
    "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
]
```

**How to get the cookie:**
1. Log into the site in your browser
2. Open DevTools (F12) → Network tab
3. Refresh the page
4. Click any request → Headers → Copy the Cookie header value

#### Templates not updating

```bash
# Force template update
docker volume rm nuclei-templates
# Then run scan again - templates will re-download
```

### Debug Mode

Run nuclei manually via Docker to debug:

```bash
docker run --rm \
  -v $(pwd):/data \
  projectdiscovery/nuclei:latest \
  -u https://example.com \
  -t cves/ \
  -severity critical,high \
  -v -debug
```

---

## Security Considerations

⚠️ **Legal Warning:** Only scan systems you have explicit permission to test.

| Risk | Mitigation |
|------|------------|
| Rate limiting/bans | Reduce `NUCLEI_RATE_LIMIT` |
| WAF blocking | Reduce concurrency, use delays |
| Service disruption | Exclude `dos` and `fuzz` tags |
| Detection | Use Tor, reduce rate limit |
| Account lockouts | Exclude `brute` tag |

### Safe Defaults

```python
NUCLEI_RATE_LIMIT = 50
NUCLEI_SEVERITY = ["critical", "high", "medium"]
NUCLEI_EXCLUDE_TAGS = ["dos", "fuzz", "intrusive"]
NUCLEI_EXCLUDE_TEMPLATES = ["fuzzing"]
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| Docker | Container runtime for nuclei and katana |
| `projectdiscovery/nuclei:latest` | Nuclei Docker image (auto-pulled) |
| `projectdiscovery/katana:latest` | Katana crawler Docker image (auto-pulled for DAST) |
| `nuclei-templates` volume | Persistent template storage (auto-created) |
| Python 3.8+ | Script runtime |

---

## References

### Nuclei
- [Nuclei Documentation](https://docs.nuclei.sh/)
- [Nuclei Templates Repository](https://github.com/projectdiscovery/nuclei-templates)
- [DAST Templates](https://github.com/projectdiscovery/nuclei-templates/tree/main/dast)
- [Template Guide](https://docs.nuclei.sh/template-guide/introduction)
- [Docker Hub - Nuclei](https://hub.docker.com/r/projectdiscovery/nuclei)

### Katana
- [Katana Documentation](https://docs.projectdiscovery.io/tools/katana/overview)
- [Katana GitHub](https://github.com/projectdiscovery/katana)
- [Docker Hub - Katana](https://hub.docker.com/r/projectdiscovery/katana)
- [JavaScript Crawling Guide](https://docs.projectdiscovery.io/tools/katana/running#crawling-mode)

### ProjectDiscovery
- [ProjectDiscovery Blog](https://blog.projectdiscovery.io/)
- [ProjectDiscovery GitHub](https://github.com/projectdiscovery)

---

## Custom Security Checks

In addition to Nuclei templates, RedAmon includes custom Python-based security checks that detect vulnerabilities **not covered by Nuclei**.

### Why Custom Checks?

| Check Type | Nuclei Coverage | Custom Check Adds |
|------------|-----------------|-------------------|
| **Direct IP Access** | No | Detects WAF bypass via IP |
| **TLS Expiry** | No | Warns before cert expires |
| **DNS Security** | No | SPF, DMARC, DNSSEC, Zone Transfer |
| **Rate Limiting** | No | Brute-force protection detection |
| **Service Exposure** | Partial | Redis no-auth, K8s API, SMTP relay |

### Check Categories

#### Authentication Security
| Check | Severity | Description |
|-------|----------|-------------|
| `login_no_https` | High | Login form served over HTTP |
| `session_no_secure` | Medium | Session cookie missing Secure flag |
| `session_no_httponly` | Medium | Session cookie missing HttpOnly flag |
| `basic_auth_no_tls` | High | Basic Auth over unencrypted HTTP |

#### DNS Security
| Check | Severity | Description |
|-------|----------|-------------|
| `spf_missing` | Medium | No SPF record (email spoofing risk) |
| `dmarc_missing` | Medium | No DMARC record |
| `dnssec_missing` | Low | DNSSEC not enabled |
| `zone_transfer` | High | DNS zone transfer allowed (AXFR) |

#### Port/Service Security
| Check | Severity | Description |
|-------|----------|-------------|
| `admin_port_exposed` | Medium | SSH/RDP/VNC/Telnet publicly accessible |
| `database_exposed` | High | MySQL/PostgreSQL/MongoDB/Redis exposed |
| `redis_no_auth` | Critical | Redis responds without authentication |
| `kubernetes_api_exposed` | Critical/High | K8s API publicly accessible |
| `smtp_open_relay` | High | SMTP server accepts external relay |

#### Application Security
| Check | Severity | Description |
|-------|----------|-------------|
| `csp_unsafe_inline` | Medium | CSP allows 'unsafe-inline' |
| `insecure_form_action` | High | HTTPS form posts to HTTP |

#### Rate Limiting
| Check | Severity | Description |
|-------|----------|-------------|
| `no_rate_limiting` | Medium | No rate limit on login endpoints |

### Configuration

All checks are enabled by default. Disable in `params.py`:

```python
# Global switch
SECURITY_CHECK_ENABLED = True  # Set False to skip all custom checks

# Individual checks
SECURITY_CHECK_LOGIN_NO_HTTPS = True
SECURITY_CHECK_SPF_MISSING = True
SECURITY_CHECK_REDIS_NO_AUTH = True
SECURITY_CHECK_NO_RATE_LIMITING = True
# ... see params.py for full list

# Performance
SECURITY_CHECK_TIMEOUT = 10   # Request timeout (seconds)
SECURITY_CHECK_MAX_WORKERS = 10  # Parallel workers
```

### Output Structure

Custom security check findings are stored in `security_checks`:

```json
{
  "security_checks": {
    "scan_timestamp": "2025-01-04T12:00:00",
    "findings": [
      {
        "type": "redis_no_auth",
        "severity": "critical",
        "name": "Redis Without Authentication",
        "ip": "192.168.1.100",
        "port": 6379,
        "evidence": "PING command returned PONG without authentication",
        "recommendation": "Enable Redis AUTH and use strong passwords."
      }
    ],
    "summary": {
      "total_findings": 5,
      "critical": 1,
      "high": 2,
      "medium": 2
    }
  }
}
```

---

*Documentation generated for RedAmon v1.0 - Nuclei Scanner Module with Katana DAST Integration*
