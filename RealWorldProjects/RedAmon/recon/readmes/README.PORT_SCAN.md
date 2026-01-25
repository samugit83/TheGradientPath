# RedAmon - Naabu Port Scanner

## Complete Technical Documentation

> **Module:** `recon/naabu_scan.py`  
> **Purpose:** Fast, lightweight port scanning using ProjectDiscovery's Naabu  
> **Author:** RedAmon Security Suite

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Configuration Parameters](#configuration-parameters)
5. [Architecture & Flow](#architecture--flow)
6. [Output Data Structure](#output-data-structure)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The `naabu_scan.py` module integrates ProjectDiscovery's Naabu scanner into RedAmon's reconnaissance pipeline. Naabu is optimized for fast, reliable port scanning at scale.

**⚠️ Important:** Naabu runs exclusively via Docker. No native installation is supported.

### Why Naabu?

| Feature | Traditional Scanners | Naabu |
|---------|---------------------|-------|
| Speed | Minutes-hours | **Seconds-minutes** |
| Resource Usage | High | **Low** |
| Docker Support | Variable | **Native** |
| CDN Detection | Limited | **Built-in** |
| Rate Control | Basic | **Fine-grained** |

### How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Recon Data     │────▶│  naabu_scan.py   │────▶│  Enriched JSON  │
│  (hostnames,    │     │                  │     │  with ports,    │
│   IPs from DNS) │     │  1. Extract IPs  │     │  services, CDN  │
└─────────────────┘     │  2. Build targets│     └─────────────────┘
                        │  3. Run Naabu    │
                        │  4. Parse JSON   │
                        └──────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Fast Scanning** | SYN scan with configurable rate limiting |
| **CDN Detection** | Identifies CDN-protected hosts (Cloudflare, Akamai, etc.) |
| **Service Detection** | Maps common ports to service names |
| **Docker Execution** | No local installation required |
| **Tor Support** | Anonymous scanning via SOCKS proxy |
| **Passive Mode** | Query Shodan InternetDB instead of active scanning |
| **Incremental Saving** | Results saved progressively |

---

## Installation

### Requirements

- **Docker** installed and running
- **Root/sudo** for SYN scans (or use CONNECT mode)

### Setup

```bash
# Make sure Docker is running
sudo systemctl start docker

# Run the scan - image will be pulled automatically
python3 recon/main.py
```

### Verify Docker is Ready

```bash
# Check Docker is running
docker info

# Optionally pre-pull the image
docker pull projectdiscovery/naabu:latest
```

---

## Configuration Parameters

All parameters are defined in `params.py`:

### Docker Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NAABU_DOCKER_IMAGE` | `str` | `"projectdiscovery/naabu:latest"` | Docker image to use |

### Port Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NAABU_TOP_PORTS` | `str` | `"1000"` | Number of top ports ("100", "1000", "full") |
| `NAABU_CUSTOM_PORTS` | `str` | `""` | Custom ports (overrides TOP_PORTS) |

**Port Examples:**
```python
NAABU_TOP_PORTS = "100"           # Top 100 ports
NAABU_TOP_PORTS = "1000"          # Top 1000 ports (default)
NAABU_CUSTOM_PORTS = "22,80,443"  # Specific ports
NAABU_CUSTOM_PORTS = "1-65535"    # Full port range
```

### Scan Type

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NAABU_SCAN_TYPE` | `str` | `"s"` | Scan type: "s" (SYN) or "c" (CONNECT) |

**Scan Types:**
- `"s"` (SYN) - Faster, more reliable, **requires root/sudo**
- `"c"` (CONNECT) - No root needed, full TCP handshake

**SYN Scan (Half-Open) - Default:**
```
   You                                    Target
    │                                        │
    │──────────── 1. SYN ───────────────────►│
    │                                        │
    │◄─────────── 2. SYN/ACK ───────────────│  ← Port OPEN
    │                                        │
    │──────────── 3. RST (abort) ───────────►│  ← Connection killed
    │                                        │
    │         ❌ NO CONNECTION ESTABLISHED    │
    │         ❌ MINIMAL LOGGING ON TARGET    │
```

**CONNECT Scan (Full TCP):**
```
   You                                    Target
    │                                        │
    │──────────── 1. SYN ───────────────────►│
    │                                        │
    │◄─────────── 2. SYN/ACK ───────────────│
    │                                        │
    │──────────── 3. ACK ───────────────────►│  ← FULL HANDSHAKE
    │                                        │
    │         ✅ CONNECTION ESTABLISHED       │
    │         ✅ LOGGED BY APPLICATION        │
    │                                        │
    │──────────── 4. FIN ───────────────────►│
    │◄─────────── 5. FIN/ACK ───────────────│  ← GRACEFUL CLOSE
    │──────────── 6. ACK ───────────────────►│
```

**Comparison:**

| Aspect | SYN (`-s s`) | CONNECT (`-s c`) |
|--------|--------------|------------------|
| Packets sent | 2 (SYN, RST) | 6+ (full handshake) |
| Speed | ⚡ Faster | 🐢 Slower |
| Stealth | 🥷 Stealthier | 👀 Easily detected |
| Application logging | ❌ Usually not | ✅ Logged |
| Requires root | ✅ Yes | ❌ No |
| Works through proxy | ❌ No | ✅ Yes (SOCKS/Tor) |

### Performance Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NAABU_RATE_LIMIT` | `int` | `1000` | Packets per second |
| `NAABU_THREADS` | `int` | `25` | Concurrent threads |
| `NAABU_TIMEOUT` | `int` | `10000` | Timeout per port (ms) |
| `NAABU_RETRIES` | `int` | `3` | Retries for failed probes |

**Recommended Settings:**

| Scenario | Rate Limit | Threads | Timeout |
|----------|------------|---------|---------|
| **Safe/Slow** | 500 | 10 | 15000 |
| **Normal** | 1000 | 25 | 10000 |
| **Aggressive** | 3000 | 50 | 5000 |
| **Internal Network** | 5000+ | 100 | 3000 |

### Feature Flags

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NAABU_EXCLUDE_CDN` | `bool` | `True` | Skip non-standard ports on CDN hosts |
| `NAABU_DISPLAY_CDN` | `bool` | `True` | Show CDN information in output |
| `NAABU_SKIP_HOST_DISCOVERY` | `bool` | `True` | Assume all hosts are up |
| `NAABU_VERIFY_PORTS` | `bool` | `True` | Extra TCP check to verify ports |
| `NAABU_PASSIVE_MODE` | `bool` | `False` | Use Shodan InternetDB (no active scan) |

---

#### `NAABU_EXCLUDE_CDN` - CDN Port Filtering

**Problem:** CDN providers (Cloudflare, Akamai, AWS CloudFront) expose thousands of ports on their edge servers that belong to the CDN infrastructure, not the actual target.

```
Without CDN Exclusion:
┌─────────────────────────────────────────────────────────────────┐
│  Target: example.com (behind Cloudflare)                        │
│                                                                 │
│  Scan Result: Ports 22, 80, 443, 2052, 2053, 2082, 2083,       │
│               2086, 2087, 2095, 2096, 8080, 8443...            │
│                                                                 │
│  Reality: Only 80/443 are YOUR server                          │
│           All others are Cloudflare's infrastructure ❌         │
└─────────────────────────────────────────────────────────────────┘

With CDN Exclusion (NAABU_EXCLUDE_CDN = True):
┌─────────────────────────────────────────────────────────────────┐
│  Target: example.com (behind Cloudflare)                        │
│                                                                 │
│  Scan Result: Ports 80, 443                                    │
│  CDN Detected: cloudflare ✅                                    │
│                                                                 │
│  Clean results - only ports that reach YOUR origin server      │
└─────────────────────────────────────────────────────────────────┘
```

| Setting | Behavior |
|---------|----------|
| `True` (default) | Only scan 80/443 on CDN hosts, skip other ports |
| `False` | Scan all ports (includes CDN infrastructure noise) |

---

#### `NAABU_DISPLAY_CDN` - CDN Detection Display

Shows which CDN provider protects each host in the output.

```json
{
  "host": "cdn.example.com",
  "ip": "104.16.123.96",
  "ports": [80, 443],
  "cdn": "cloudflare",    // ← This field
  "is_cdn": true          // ← And this field
}
```

**Detected CDN Providers:**
- Cloudflare, Akamai, Fastly, AWS CloudFront
- Google Cloud CDN, Azure CDN, Incapsula
- MaxCDN, KeyCDN, StackPath, and more

---

#### `NAABU_SKIP_HOST_DISCOVERY` - Skip Ping Check (`-Pn`)

**Host Discovery** = Check if host is alive before scanning ports.

```
WITH Host Discovery (NAABU_SKIP_HOST_DISCOVERY = False):
┌─────────────────────────────────────────────────────────────────┐
│   You                                        Target             │
│    │                                            │               │
│    │   PHASE 1: Is host alive?                  │               │
│    │                                            │               │
│    │──────────── ICMP Ping ────────────────────►│               │
│    │◄─────────── Pong (or timeout) ────────────│               │
│    │                                            │               │
│    │   If NO response → Host "DOWN" → SKIP ❌   │               │
│    │   If response → Continue to port scan      │               │
│    │                                            │               │
│    │   PHASE 2: Port scan (only if UP)          │               │
│    │──────────── SYN port 80 ──────────────────►│               │
└─────────────────────────────────────────────────────────────────┘

WITHOUT Host Discovery (NAABU_SKIP_HOST_DISCOVERY = True):
┌─────────────────────────────────────────────────────────────────┐
│   You                                        Target             │
│    │                                            │               │
│    │   SKIP Phase 1 - Assume host is UP         │               │
│    │                                            │               │
│    │   Go directly to port scanning             │               │
│    │──────────── SYN port 80 ──────────────────►│               │
│    │──────────── SYN port 443 ─────────────────►│               │
└─────────────────────────────────────────────────────────────────┘
```

| Setting | Behavior | Use When |
|---------|----------|----------|
| `True` (default) | Skip ping, assume all hosts UP | Firewalls block ICMP, hosts from DNS |
| `False` | Ping first, skip "dead" hosts | Large IP ranges, internal networks |

**Why default is `True`:** RedAmon already confirmed hosts exist via DNS resolution. Many firewalls block ICMP ping, causing false negatives.

---

#### `NAABU_VERIFY_PORTS` - Double-Check Open Ports

After finding an open port via SYN scan, perform an additional TCP connection to verify.

```
Without Verification:
┌─────────────────────────────────────────────────────────────────┐
│   SYN scan says port 8080 is OPEN                              │
│   → Report as open (might be false positive)                   │
└─────────────────────────────────────────────────────────────────┘

With Verification (NAABU_VERIFY_PORTS = True):
┌─────────────────────────────────────────────────────────────────┐
│   SYN scan says port 8080 is OPEN                              │
│   → Try full TCP connection to verify                          │
│   → Connection successful? Report as open ✅                    │
│   → Connection failed? Discard (was false positive) ❌          │
└─────────────────────────────────────────────────────────────────┘
```

| Setting | Behavior | Trade-off |
|---------|----------|-----------|
| `True` (default) | Verify each open port with TCP connect | More accurate, slightly slower |
| `False` | Trust SYN scan results directly | Faster, may have false positives |

**Reduces false positives from:**
- Stateful firewalls that RST after SYN/ACK
- Load balancers with connection limits
- Rate-limiting that causes inconsistent responses

---

#### `NAABU_PASSIVE_MODE` - Shodan InternetDB (No Active Scan)

Instead of actively scanning the target, query **Shodan's InternetDB** for known open ports.

```
Active Scanning (NAABU_PASSIVE_MODE = False):
┌─────────────────────────────────────────────────────────────────┐
│   You ──────── SYN packets ──────────────────► Target           │
│                                                                 │
│   • Sends packets to target                                    │
│   • Target sees your IP                                        │
│   • May trigger IDS/IPS alerts                                 │
│   • Real-time results                                          │
└─────────────────────────────────────────────────────────────────┘

Passive Mode (NAABU_PASSIVE_MODE = True):
┌─────────────────────────────────────────────────────────────────┐
│   You ──────── API query ──────────────────► Shodan InternetDB  │
│                                                                 │
│   • NO packets to target                                       │
│   • Target never sees you                                      │
│   • 100% stealth                                               │
│   • Data may be days/weeks old                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Setting | Behavior | Trade-off |
|---------|----------|-----------|
| `False` (default) | Active SYN scan | Real-time, target aware |
| `True` | Query Shodan database | Stealth, but potentially stale data |

**Use Passive Mode when:**
- Initial reconnaissance (don't want to touch target yet)
- Target has strict IDS/IPS
- Legal constraints on active scanning
- Quick overview before active scan

**Limitations:**
- Data freshness depends on Shodan's last scan
- May miss recently opened ports
- May show ports that are now closed

### CDN Handling

When `NAABU_EXCLUDE_CDN = True`:
- CDN-protected hosts only scanned on ports 80/443
- Reduces false positives from CDN edge servers
- Still reports CDN detection in output

---

## Architecture & Flow

### Execution Flow

```
1. INITIALIZATION
   └── Check Docker availability
   └── Pull Naabu image if needed
   └── Check Tor availability (if enabled)

2. TARGET EXTRACTION
   └── Parse recon_data JSON
   └── Extract unique IPs from DNS records
   └── Extract hostnames/subdomains
   └── Build IP-to-hostname mapping

3. SCAN EXECUTION
   └── Create targets file (hostnames preferred over IPs)
   └── Build Naabu Docker command
   └── Execute with JSON Lines output
   └── Monitor progress

4. RESULT PROCESSING
   └── Parse JSONL output line by line
   └── Group results by host and IP
   └── Map ports to service names
   └── Calculate statistics

5. DATA ENRICHMENT
   └── Add "naabu" section to recon_data
   └── Save incrementally to JSON file
   └── Generate summary statistics
```

### Docker Command Structure

```bash
docker run --rm \
  --net=host \                              # Required for SYN scans
  -v /targets:/targets:ro \
  -v /output:/output \
  projectdiscovery/naabu:latest \
  -list /targets/targets.txt \
  -o /output/naabu_output.json \
  -json \
  -silent \
  -top-ports 1000 \
  -scan-type s \
  -rate 1000 \
  -c 25 \
  -timeout 10000 \
  -retries 3 \
  -cdn \                                    # Display CDN
  # -sD not used (not yet implemented in naabu)
  -Pn \                                     # Skip host discovery
  -verify                                   # Verify ports
```

---

## Output Data Structure

### Complete JSON Schema

```json
{
  "naabu": {
    "scan_metadata": {
      "scan_timestamp": "2024-01-15T12:00:00.000000",
      "scan_duration_seconds": 45.2,
      "docker_image": "projectdiscovery/naabu:latest",
      "scan_type": "syn",
      "ports_config": "top-1000",
      "rate_limit": 1000,
      "passive_mode": false,
      "proxy_used": false,
      "total_targets": 15,
      "cdn_exclusion": true
    },
    
    "by_host": {
      "example.com": {
        "host": "example.com",
        "ip": "93.184.216.34",
        "ports": [80, 443, 8080],
        "port_details": [
          {
            "port": 80,
            "protocol": "tcp",
            "service": "http"
          },
          {
            "port": 443,
            "protocol": "tcp",
            "service": "https"
          },
          {
            "port": 8080,
            "protocol": "tcp",
            "service": "http-proxy"
          }
        ],
        "cdn": null,
        "is_cdn": false
      },
      "cdn.example.com": {
        "host": "cdn.example.com",
        "ip": "104.16.123.96",
        "ports": [80, 443],
        "port_details": [...],
        "cdn": "cloudflare",
        "is_cdn": true
      }
    },
    
    "by_ip": {
      "93.184.216.34": {
        "ip": "93.184.216.34",
        "hostnames": ["example.com", "www.example.com"],
        "ports": [80, 443, 8080],
        "cdn": null,
        "is_cdn": false
      }
    },
    
    "all_ports": [22, 80, 443, 3306, 8080],
    
    "ip_to_hostnames": {
      "93.184.216.34": ["example.com", "www.example.com"]
    },
    
    "summary": {
      "hosts_scanned": 15,
      "ips_scanned": 12,
      "hosts_with_open_ports": 10,
      "total_open_ports": 45,
      "unique_ports": [22, 80, 443, 3306, 8080],
      "unique_port_count": 5,
      "cdn_hosts": 3
    }
  }
}
```

### Service Name Mapping

| Port | Service |
|------|---------|
| 21 | ftp |
| 22 | ssh |
| 23 | telnet |
| 25 | smtp |
| 53 | dns |
| 80 | http |
| 110 | pop3 |
| 143 | imap |
| 443 | https |
| 445 | microsoft-ds |
| 3306 | mysql |
| 3389 | ms-wbt-server |
| 5432 | postgresql |
| 6379 | redis |
| 8080 | http-proxy |
| 8443 | https-alt |
| 27017 | mongodb |

---

## Usage Examples

### Basic Usage (via main.py)

```python
# Include "naabu" in SCAN_MODULES in params.py
SCAN_MODULES = ["initial_recon", "naabu", "httpx", "nuclei"]

# Run the full pipeline
python3 recon/main.py
```

### Standalone Enrichment

```python
from naabu_scan import enrich_recon_file
from pathlib import Path

enriched = enrich_recon_file(Path("output/recon_example.com.json"))
```

### Command Line

```bash
# Enrich an existing recon file
python3 recon/naabu_scan.py output/recon_example.com.json
```

### Configuration Profiles

**Quick Scan (Fast):**
```python
NAABU_TOP_PORTS = "100"
NAABU_RATE_LIMIT = 3000
NAABU_THREADS = 50
NAABU_VERIFY_PORTS = False
```

**Comprehensive Scan:**
```python
NAABU_TOP_PORTS = "1000"
NAABU_RATE_LIMIT = 1000
NAABU_THREADS = 25
NAABU_VERIFY_PORTS = True
```

**Stealth Scan:**
```python
NAABU_TOP_PORTS = "100"
NAABU_RATE_LIMIT = 100
NAABU_THREADS = 5
NAABU_SCAN_TYPE = "c"  # CONNECT instead of SYN
USE_TOR_FOR_RECON = True
```

**Full Port Scan:**
```python
NAABU_CUSTOM_PORTS = "1-65535"
NAABU_RATE_LIMIT = 5000
NAABU_THREADS = 100
NAABU_TIMEOUT = 5000
```

---

## Troubleshooting

### Common Issues

#### "Docker not found"

```bash
# Install Docker
sudo apt install docker.io  # Debian/Ubuntu

# Start Docker daemon
sudo systemctl start docker
sudo systemctl enable docker
```

#### "Permission denied" for SYN scan

```bash
# Option 1: Run with sudo
sudo python3 recon/main.py

# Option 2: Use CONNECT scan (no root needed)
NAABU_SCAN_TYPE = "c"
```

#### "No open ports found"

Possible causes:
1. Target firewall blocking scans
2. Rate limiting triggered
3. CDN blocking non-80/443 ports

Solutions:
```python
# Reduce rate limit
NAABU_RATE_LIMIT = 100

# Disable CDN exclusion to see all ports
NAABU_EXCLUDE_CDN = False

# Try passive mode
NAABU_PASSIVE_MODE = True
```

#### Scan too slow

```python
# Increase performance
NAABU_RATE_LIMIT = 3000
NAABU_THREADS = 50
NAABU_TIMEOUT = 5000
NAABU_RETRIES = 1
```

### Debug Mode

Run Naabu manually via Docker:

```bash
docker run --rm --net=host \
  projectdiscovery/naabu:latest \
  -host example.com \
  -top-ports 100 \
  -v -debug
```

---

## Security Considerations

⚠️ **Legal Warning:** Only scan systems you have explicit permission to test.

| Risk | Mitigation |
|------|------------|
| Rate limiting/bans | Reduce `NAABU_RATE_LIMIT` |
| IDS/IPS detection | Use CONNECT mode, lower rate |
| CDN blocking | Use `NAABU_EXCLUDE_CDN = True` |
| Detection | Use Tor, reduce rate limit |

### Safe Defaults

```python
NAABU_RATE_LIMIT = 500
NAABU_THREADS = 10
NAABU_TOP_PORTS = "100"
NAABU_EXCLUDE_CDN = True
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| Docker | Container runtime for Naabu |
| `projectdiscovery/naabu:latest` | Naabu Docker image (auto-pulled) |
| Python 3.8+ | Script runtime |

---

## References

- [Naabu Documentation](https://github.com/projectdiscovery/naabu)
- [Naabu Docker Hub](https://hub.docker.com/r/projectdiscovery/naabu)
- [ProjectDiscovery Blog](https://blog.projectdiscovery.io/)

---

*Documentation generated for RedAmon v1.0 - Naabu Port Scanner Module*

