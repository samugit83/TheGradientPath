"""
RedAmon - Port Scanner Module

Fast, lightweight port scanning.
Runs via Docker for consistent environment and no installation required.

Features:
- SYN and CONNECT scan modes
- Service detection
- CDN/WAF detection
- Passive mode via Shodan InternetDB
- JSON output with structured results
"""

import json
import subprocess
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from recon.params import (
    NAABU_DOCKER_IMAGE,
    NAABU_TOP_PORTS,
    NAABU_CUSTOM_PORTS,
    NAABU_RATE_LIMIT,
    NAABU_THREADS,
    NAABU_TIMEOUT,
    NAABU_RETRIES,
    NAABU_SCAN_TYPE,
    NAABU_EXCLUDE_CDN,
    NAABU_DISPLAY_CDN,
    NAABU_SKIP_HOST_DISCOVERY,
    NAABU_VERIFY_PORTS,
    NAABU_PASSIVE_MODE,
    USE_TOR_FOR_RECON,
)


# =============================================================================
# Docker Helper Functions
# =============================================================================

def is_docker_installed() -> bool:
    """Check if Docker is installed."""
    return shutil.which("docker") is not None


def is_docker_running() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def pull_naabu_docker_image() -> bool:
    """Pull the Naabu Docker image if not present."""
    print(f"    [*] Checking Naabu Docker image: {NAABU_DOCKER_IMAGE}")

    # Check if image exists
    result = subprocess.run(
        ["docker", "images", "-q", NAABU_DOCKER_IMAGE],
        capture_output=True,
        text=True
    )

    if result.stdout.strip():
        print(f"    [✓] Image already available")
        return True

    print(f"    [*] Pulling image (this may take a moment)...")
    result = subprocess.run(
        ["docker", "pull", NAABU_DOCKER_IMAGE],
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode == 0:
        print(f"    [✓] Image pulled successfully")
        return True
    else:
        print(f"    [!] Failed to pull image: {result.stderr[:200]}")
        return False


def is_tor_running() -> bool:
    """Check if Tor SOCKS proxy is available."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', 9050))
        sock.close()
        return result == 0
    except Exception:
        return False


# =============================================================================
# Target Extraction
# =============================================================================

def extract_targets_from_recon(recon_data: dict) -> Tuple[Set[str], Set[str], Dict[str, List[str]]]:
    """
    Extract IPs and hostnames from recon data.

    Returns:
        Tuple of (unique_ips, unique_hostnames, ip_to_hostnames_mapping)
    """
    unique_ips = set()
    unique_hostnames = set()
    ip_to_hostnames = {}

    dns_data = recon_data.get("dns", {})

    # Extract from root domain
    domain_dns = dns_data.get("domain", {})
    domain_name = recon_data.get("domain", "")

    if domain_name:
        domain_ips = domain_dns.get("ips", {})
        ipv4_list = domain_ips.get("ipv4", [])
        ipv6_list = domain_ips.get("ipv6", [])

        if ipv4_list or ipv6_list:
            unique_hostnames.add(domain_name)
            for ip in ipv4_list + ipv6_list:
                unique_ips.add(ip)
                if ip not in ip_to_hostnames:
                    ip_to_hostnames[ip] = []
                if domain_name not in ip_to_hostnames[ip]:
                    ip_to_hostnames[ip].append(domain_name)

    # Extract from subdomains
    subdomains_dns = dns_data.get("subdomains", {})
    for subdomain, sub_data in subdomains_dns.items():
        if not sub_data.get("has_records", False):
            continue

        sub_ips = sub_data.get("ips", {})
        ipv4_list = sub_ips.get("ipv4", [])
        ipv6_list = sub_ips.get("ipv6", [])

        if ipv4_list or ipv6_list:
            unique_hostnames.add(subdomain)
            for ip in ipv4_list + ipv6_list:
                unique_ips.add(ip)
                if ip not in ip_to_hostnames:
                    ip_to_hostnames[ip] = []
                if subdomain not in ip_to_hostnames[ip]:
                    ip_to_hostnames[ip].append(subdomain)

    return unique_ips, unique_hostnames, ip_to_hostnames


# =============================================================================
# Naabu Command Builder
# =============================================================================

def get_host_path(container_path: str) -> str:
    """
    Convert container path to host path for Docker-in-Docker volume mounts.

    When running inside a container with mounted volumes, sibling containers
    need host paths, not container paths. This function translates paths
    using the HOST_RECON_OUTPUT_PATH environment variable.

    The recon container mounts: ./output:/app/recon/output
    So /app/recon/output/* inside the container maps to <host>/recon/output/* on host.
    """
    host_output_path = os.environ.get("HOST_RECON_OUTPUT_PATH", "")
    container_output_path = "/app/recon/output"

    if host_output_path and container_path.startswith(container_output_path):
        # Replace container path with host path
        return container_path.replace(container_output_path, host_output_path, 1)

    # If not in container or path doesn't match, return as-is
    return container_path


def build_naabu_command(targets_file: str, output_file: str, use_proxy: bool = False) -> List[str]:
    """
    Build the Docker command for running Naabu.

    Args:
        targets_file: Path to file containing targets (one per line)
        output_file: Path for JSON output
        use_proxy: Whether to use Tor proxy

    Returns:
        List of command arguments
    """
    # Convert container paths to host paths for sibling container volume mounts
    targets_host_path = get_host_path(str(Path(targets_file).parent))
    output_host_path = get_host_path(str(Path(output_file).parent))

    targets_filename = Path(targets_file).name
    output_filename = Path(output_file).name

    # Build Docker command
    # Note: Naabu requires --net=host for proper packet handling
    cmd = [
        "docker", "run", "--rm",
        "--net=host",  # Required for SYN scans
        "-v", f"{targets_host_path}:/targets:ro",
        "-v", f"{output_host_path}:/output",
    ]

    # Add image
    cmd.append(NAABU_DOCKER_IMAGE)

    # Input/Output
    cmd.extend(["-list", f"/targets/{targets_filename}"])
    cmd.extend(["-o", f"/output/{output_filename}"])
    cmd.append("-json")
    cmd.append("-silent")

    # Port configuration
    if NAABU_CUSTOM_PORTS:
        cmd.extend(["-p", NAABU_CUSTOM_PORTS])
    elif NAABU_TOP_PORTS:
        cmd.extend(["-top-ports", str(NAABU_TOP_PORTS)])

    # Scan type
    cmd.extend(["-scan-type", NAABU_SCAN_TYPE])

    # Performance settings
    cmd.extend(["-rate", str(NAABU_RATE_LIMIT)])
    cmd.extend(["-c", str(NAABU_THREADS)])
    cmd.extend(["-timeout", str(NAABU_TIMEOUT)])
    cmd.extend(["-retries", str(NAABU_RETRIES)])

    # Feature flags
    if NAABU_EXCLUDE_CDN:
        cmd.append("-exclude-cdn")

    if NAABU_DISPLAY_CDN:
        cmd.append("-cdn")

    if NAABU_SKIP_HOST_DISCOVERY:
        cmd.append("-Pn")

    if NAABU_VERIFY_PORTS:
        cmd.append("-verify")

    if NAABU_PASSIVE_MODE:
        cmd.append("-passive")

    # Proxy support (naabu expects just ip:port for socks5 proxy)
    if use_proxy:
        cmd.extend(["-proxy", "127.0.0.1:9050"])

    return cmd


# =============================================================================
# Result Parsing
# =============================================================================

def parse_naabu_output(output_file: str) -> Dict:
    """
    Parse Naabu JSON Lines output into structured format.

    Naabu outputs one JSON object per line:
    {"host":"example.com","ip":"93.184.216.34","port":80}
    {"host":"example.com","ip":"93.184.216.34","port":443}

    Returns:
        Structured dictionary with by_host, by_ip, and summary sections
    """
    by_host = {}
    by_ip = {}
    all_ports = set()

    if not Path(output_file).exists():
        return {"by_host": {}, "by_ip": {}, "all_ports": [], "summary": {}}

    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            host = entry.get("host", "")
            ip = entry.get("ip", "")
            port = entry.get("port")
            cdn = entry.get("cdn", "")
            cdn_name = entry.get("cdn-name", "")

            if port:
                all_ports.add(port)

            # Organize by host
            if host:
                if host not in by_host:
                    by_host[host] = {
                        "host": host,
                        "ip": ip,
                        "ports": [],
                        "port_details": [],
                        "cdn": cdn_name if cdn_name else None,
                        "is_cdn": bool(cdn or cdn_name)
                    }

                if port and port not in by_host[host]["ports"]:
                    by_host[host]["ports"].append(port)

                    # Determine service based on common port mappings
                    service = get_service_name(port)
                    by_host[host]["port_details"].append({
                        "port": port,
                        "protocol": "tcp",
                        "service": service
                    })

            # Organize by IP
            if ip:
                if ip not in by_ip:
                    by_ip[ip] = {
                        "ip": ip,
                        "hostnames": [],
                        "ports": [],
                        "cdn": cdn_name if cdn_name else None,
                        "is_cdn": bool(cdn or cdn_name)
                    }

                if host and host not in by_ip[ip]["hostnames"]:
                    by_ip[ip]["hostnames"].append(host)

                if port and port not in by_ip[ip]["ports"]:
                    by_ip[ip]["ports"].append(port)

    # Sort ports
    for host in by_host:
        by_host[host]["ports"].sort()
        by_host[host]["port_details"].sort(key=lambda x: x["port"])

    for ip in by_ip:
        by_ip[ip]["ports"].sort()

    all_ports_sorted = sorted(list(all_ports))

    # Build summary
    summary = {
        "hosts_scanned": len(by_host),
        "ips_scanned": len(by_ip),
        "hosts_with_open_ports": len([h for h in by_host.values() if h["ports"]]),
        "total_open_ports": sum(len(h["ports"]) for h in by_host.values()),
        "unique_ports": all_ports_sorted,
        "unique_port_count": len(all_ports_sorted),
        "cdn_hosts": len([h for h in by_host.values() if h.get("is_cdn")])
    }

    return {
        "by_host": by_host,
        "by_ip": by_ip,
        "all_ports": all_ports_sorted,
        "summary": summary
    }


def get_service_name(port: int) -> str:
    """Map common ports to service names."""
    port_services = {
        21: "ftp",
        22: "ssh",
        23: "telnet",
        25: "smtp",
        53: "dns",
        80: "http",
        110: "pop3",
        111: "rpcbind",
        135: "msrpc",
        139: "netbios-ssn",
        143: "imap",
        443: "https",
        445: "microsoft-ds",
        993: "imaps",
        995: "pop3s",
        1433: "ms-sql",
        1521: "oracle",
        3306: "mysql",
        3389: "ms-wbt-server",
        5432: "postgresql",
        5900: "vnc",
        6379: "redis",
        8080: "http-proxy",
        8443: "https-alt",
        8888: "http-alt",
        9200: "elasticsearch",
        27017: "mongodb",
    }
    return port_services.get(port, "unknown")


# =============================================================================
# File Ownership Handling
# =============================================================================

def get_real_user_ids() -> tuple:
    """Get the real user/group IDs (handles sudo)."""
    sudo_uid = os.environ.get('SUDO_UID')
    sudo_gid = os.environ.get('SUDO_GID')

    if sudo_uid and sudo_gid:
        return (int(sudo_uid), int(sudo_gid))
    return (os.getuid(), os.getgid())


def fix_file_ownership(file_path: Path) -> None:
    """Fix file ownership for files created by Docker (as root)."""
    try:
        uid, gid = get_real_user_ids()
        os.chown(str(file_path), uid, gid)
    except Exception:
        pass  # Silently ignore if we can't change ownership


# =============================================================================
# Main Scan Function
# =============================================================================

def run_port_scan(recon_data: dict, output_file: Path = None) -> dict:
    """
    Run Naabu port scan on targets from recon data.

    Args:
        recon_data: Dictionary containing DNS/subdomain data
        output_file: Path to save enriched results (optional)

    Returns:
        Enriched recon_data with "port_scan" section added
    """
    print("\n" + "="*60)
    print("NAABU PORT SCANNER")
    print("="*60)

    # Check Docker
    if not is_docker_installed():
        print("[!] Docker is not installed. Please install Docker first.")
        return recon_data

    if not is_docker_running():
        print("[!] Docker daemon is not running. Please start Docker.")
        return recon_data

    # Pull image if needed
    if not pull_naabu_docker_image():
        print("[!] Failed to get Naabu Docker image")
        return recon_data

    # Check Tor if enabled
    use_proxy = False
    if USE_TOR_FOR_RECON:
        if is_tor_running():
            print("    [✓] Tor proxy detected - enabling anonymous scanning")
            use_proxy = True
        else:
            print("    [!] Tor not running - scanning without proxy")

    # Extract targets
    print("\n[*] Extracting targets from recon data...")
    unique_ips, unique_hostnames, ip_to_hostnames = extract_targets_from_recon(recon_data)

    # Combine targets - prefer hostnames for better accuracy
    all_targets = list(unique_hostnames) + [ip for ip in unique_ips if ip not in [
        h_ip for h in unique_hostnames for h_ip in ip_to_hostnames.get(h, [])
    ]]

    if not all_targets:
        print("[!] No targets found in recon data")
        return recon_data

    print(f"    [*] Found {len(unique_hostnames)} hostnames and {len(unique_ips)} IPs")
    print(f"    [*] Total targets to scan: {len(all_targets)}")

    # Create temp directory for scan files
    scan_temp_dir = Path(__file__).parent / "output" / ".naabu_temp"
    scan_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write targets file
        targets_file = scan_temp_dir / "targets.txt"
        with open(targets_file, 'w') as f:
            for target in all_targets:
                f.write(f"{target}\n")

        # Set output file
        naabu_output = scan_temp_dir / "naabu_output.json"

        # Build and run command
        cmd = build_naabu_command(str(targets_file), str(naabu_output), use_proxy)

        print(f"\n[*] Starting Naabu scan...")
        print(f"    [*] Scan type: {'SYN' if NAABU_SCAN_TYPE == 's' else 'CONNECT'}")
        print(f"    [*] Ports: {NAABU_CUSTOM_PORTS if NAABU_CUSTOM_PORTS else f'top {NAABU_TOP_PORTS}'}")
        print(f"    [*] Rate limit: {NAABU_RATE_LIMIT} pps")

        if NAABU_PASSIVE_MODE:
            print(f"    [*] Mode: PASSIVE (Shodan InternetDB)")

        start_time = datetime.now()

        # Execute scan
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        _, stderr = process.communicate(timeout=1800)  # 30 min timeout

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if process.returncode != 0 and not naabu_output.exists():
            print(f"    [!] Scan failed: {stderr[:200] if stderr else 'Unknown error'}")
            return recon_data

        # Parse results
        print(f"\n[*] Parsing results...")
        results = parse_naabu_output(str(naabu_output))

        # Build final structure
        naabu_results = {
            "scan_metadata": {
                "scan_timestamp": start_time.isoformat(),
                "scan_duration_seconds": round(duration, 2),
                "docker_image": NAABU_DOCKER_IMAGE,
                "scan_type": "syn" if NAABU_SCAN_TYPE == "s" else "connect",
                "ports_config": NAABU_CUSTOM_PORTS if NAABU_CUSTOM_PORTS else f"top-{NAABU_TOP_PORTS}",
                "rate_limit": NAABU_RATE_LIMIT,
                "passive_mode": NAABU_PASSIVE_MODE,
                "proxy_used": use_proxy,
                "total_targets": len(all_targets),
                "cdn_exclusion": NAABU_EXCLUDE_CDN
            },
            "by_host": results["by_host"],
            "by_ip": results["by_ip"],
            "all_ports": results["all_ports"],
            "ip_to_hostnames": ip_to_hostnames,
            "summary": results["summary"]
        }

        # Print summary
        summary = results["summary"]
        print(f"\n[✓] Scan completed in {duration:.1f} seconds")
        print(f"    [*] Hosts with open ports: {summary['hosts_with_open_ports']}")
        print(f"    [*] Total open ports found: {summary['total_open_ports']}")
        print(f"    [*] Unique ports: {summary['unique_port_count']}")

        if summary.get('cdn_hosts', 0) > 0:
            print(f"    [*] CDN-protected hosts: {summary['cdn_hosts']}")

        if results["all_ports"]:
            print(f"    [*] Ports discovered: {', '.join(map(str, results['all_ports'][:20]))}" +
                  (f"... (+{len(results['all_ports'])-20} more)" if len(results['all_ports']) > 20 else ""))

        # Add to recon_data
        recon_data["port_scan"] = naabu_results

        # Save incrementally
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(recon_data, f, indent=2, default=str)
            fix_file_ownership(output_file)
            print(f"\n[✓] Results saved to {output_file}")

        return recon_data

    except subprocess.TimeoutExpired:
        print("[!] Scan timed out after 30 minutes")
        return recon_data
    except Exception as e:
        print(f"[!] Error during scan: {e}")
        return recon_data
    finally:
        # Cleanup temp files
        try:
            if scan_temp_dir.exists():
                for f in scan_temp_dir.iterdir():
                    f.unlink()
                scan_temp_dir.rmdir()
        except Exception:
            pass


# =============================================================================
# Standalone Entry Point
# =============================================================================

def enrich_recon_file(recon_file: Path) -> dict:
    """
    Enrich an existing recon JSON file with Naabu scan results.

    Args:
        recon_file: Path to existing recon JSON file

    Returns:
        Enriched recon data
    """
    print(f"\n[*] Loading recon file: {recon_file}")

    with open(recon_file, 'r') as f:
        recon_data = json.load(f)

    enriched = run_port_scan(recon_data, output_file=recon_file)

    return enriched

