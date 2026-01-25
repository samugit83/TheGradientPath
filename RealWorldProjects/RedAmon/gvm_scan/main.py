#!/usr/bin/env python3
"""
RedAmon - Vulnerability Scanner Main Entry Point
=================================================
Orchestrates GVM/OpenVAS vulnerability scanning using recon data.

Reads targets from recon JSON files and runs vulnerability scans
against discovered IPs and hostnames using GVM.

Usage:
    # From project root (with GVM running in Docker):
    python gvm_scan/main.py

    # Or run via Docker Compose:
    docker compose --profile scanner up python-scanner
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from params import (
    TARGET_DOMAIN,
    GVM_SCAN_TARGETS,
    GVM_CLEANUP_AFTER_SCAN,
    USE_RECON_FOR_TARGET,
    GVM_IP_LIST,
    GVM_HOSTNAME_LIST,
)

from gvm_scan.gvm_scanner import (
    GVMScanner,
    extract_targets_from_recon,
    load_recon_file,
    save_vuln_results,
    update_graph_from_gvm_results,
    GVM_AVAILABLE,
)


# Output directory for vulnerability results
OUTPUT_DIR = Path(__file__).parent / "output"


def check_recon_has_live_targets(recon_data: dict) -> tuple:
    """
    Check if recon data indicates any reachable/live targets.
    
    GVM can scan network-level vulnerabilities (SSH, FTP, etc.), not just HTTP.
    However, if both port_scan AND http_probe found nothing, hosts are likely 
    completely unreachable and GVM will also fail.
    
    Args:
        recon_data: Reconnaissance data from recon/main.py
        
    Returns:
        Tuple of (has_live_targets: bool, warning_message: str or None)
    """
    # Check port_scan results
    port_scan_data = recon_data.get('port_scan', {})
    port_summary = port_scan_data.get('summary', {})
    open_ports = port_summary.get('total_open_ports', 0)
    
    # Check http_probe results
    http_probe_data = recon_data.get('http_probe', {})
    http_summary = http_probe_data.get('summary', {})
    live_urls = http_summary.get('live_urls', 0)
    
    # Check if active scans were already skipped in recon pipeline
    active_scans_skipped = recon_data.get('metadata', {}).get('active_scans_skipped', False)
    
    # Case 1: Both port_scan and http_probe ran but found nothing
    port_scan_ran = 'port_scan' in recon_data
    http_probe_ran = 'http_probe' in recon_data
    
    if port_scan_ran and http_probe_ran:
        if open_ports == 0 and live_urls == 0:
            return False, (
                "No open ports and no live HTTP services found in recon data. "
                "Targets appear to be unreachable or heavily firewalled."
            )
    
    # Case 2: Only http_probe ran and found nothing (port_scan might have been skipped)
    if http_probe_ran and not port_scan_ran:
        if live_urls == 0:
            return False, (
                "No live HTTP services found in recon data. "
                "Port scan was not performed - GVM may still find vulnerabilities."
            )
    
    # Case 3: Active scans were skipped in recon pipeline
    if active_scans_skipped:
        return False, (
            "Active scans (resource_enum, vuln_scan) were skipped in recon pipeline. "
            "No live targets were found."
        )
    
    # Targets seem reachable
    return True, None


def run_vulnerability_scan(
    domain: str = TARGET_DOMAIN,
    scan_targets: str = GVM_SCAN_TARGETS,
    cleanup: bool = GVM_CLEANUP_AFTER_SCAN,
) -> dict:
    """
    Run vulnerability scan against targets from recon data.
    
    Args:
        domain: Target domain (reads from recon_<domain>.json)
        scan_targets: "both", "ips_only", or "hostnames_only"
        cleanup: Delete targets/tasks after scan
        
    Returns:
        Complete vulnerability scan results
    """
    print("\n" + "=" * 70)
    print("           RedAmon - GVM Vulnerability Scanner")
    print("=" * 70)
    print(f"  Target Domain: {domain}")
    print(f"  Scan Strategy: {scan_targets}")
    print(f"  Use Recon Data: {USE_RECON_FOR_TARGET}")
    print(f"  Cleanup After: {cleanup}")
    print("=" * 70 + "\n")

    # Check if GVM library is available
    if not GVM_AVAILABLE:
        print("[!] ERROR: python-gvm library not installed")
        print("[!] Install with: pip install python-gvm")
        return {"error": "python-gvm not installed"}

    # Get targets based on USE_RECON_FOR_TARGET setting
    root_domain = domain  # Default to input domain

    if USE_RECON_FOR_TARGET:
        # Load recon data
        print("[*] Loading recon data...")
        try:
            recon_data = load_recon_file(domain)
            # Get root_domain from recon metadata (consistent with recon/main.py)
            root_domain = recon_data.get("metadata", {}).get("root_domain", domain)
            print(f"    [+] Loaded: recon_{domain}.json")
            print(f"    [+] Root domain: {root_domain}")
        except FileNotFoundError as e:
            print(f"[!] ERROR: {e}")
            print(f"[!] Run domain recon first: python recon/main.py")
            return {"error": str(e)}

        # Check if recon data indicates reachable targets
        has_live_targets, warning_message = check_recon_has_live_targets(recon_data)
        
        if not has_live_targets:
            print(f"\n{'=' * 70}")
            print(f"[!] SKIPPING GVM SCAN: {warning_message}")
            print(f"{'=' * 70}")
            return {
                "error": "No live targets",
                "reason": warning_message,
                "skipped": True,
                "metadata": {
                    "scan_type": "vulnerability_scan",
                    "scan_timestamp": datetime.now().isoformat(),
                    "target_domain": root_domain,
                    "skipped_reason": warning_message
                }
            }

        # Extract targets from recon
        ips, hostnames = extract_targets_from_recon(recon_data)
    else:
        # Use manual target lists from params
        print("[*] Using manual target lists from params.py...")
        ips = set(GVM_IP_LIST)
        hostnames = set(GVM_HOSTNAME_LIST)
        print(f"    [+] GVM_IP_LIST: {len(ips)} IPs")
        print(f"    [+] GVM_HOSTNAME_LIST: {len(hostnames)} hostnames")

    print(f"    [+] Found {len(ips)} unique IPs")
    print(f"    [+] Found {len(hostnames)} unique hostnames")

    if not ips and not hostnames:
        if USE_RECON_FOR_TARGET:
            print("[!] No targets found in recon data")
        else:
            print("[!] No targets specified in GVM_IP_LIST or GVM_HOSTNAME_LIST")
        return {"error": "No targets found"}

    # Initialize results structure (use root_domain from recon metadata)
    results = {
        "metadata": {
            "scan_type": "vulnerability_scan",
            "scan_timestamp": datetime.now().isoformat(),
            "target_domain": root_domain,
            "scan_strategy": scan_targets,
            "use_recon_for_target": USE_RECON_FOR_TARGET,
            "target_source": "recon_data" if USE_RECON_FOR_TARGET else "manual_lists",
            "recon_file": f"recon_{root_domain}.json" if USE_RECON_FOR_TARGET else None,
            "targets": {
                "ips": list(ips),
                "hostnames": list(hostnames)
            }
        },
        "scans": [],
        "summary": {
            "total_vulnerabilities": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "log": 0,
            "hosts_scanned": 0,
        }
    }
    
    # Connect to GVM
    print("\n[*] Connecting to GVM...")
    scanner = GVMScanner()
    
    if not scanner.connect():
        print("[!] ERROR: Failed to connect to GVM")
        print("[!] Make sure GVM is running:")
        print("[!]   docker compose up -d")
        print("[!]   docker compose logs -f gvmd  # Wait for 'Starting GVMd'")
        return {"error": "Failed to connect to GVM"}
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"gvm_{root_domain}.json"
    
    def save_incremental():
        """Save current results incrementally."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    try:
        # =====================================================================
        # PHASE 1: Scan IPs (one at a time for incremental saving)
        # =====================================================================
        if scan_targets in ("both", "ips_only") and ips:
            ip_list = list(ips)
            print(f"\n[*] PHASE 1: Scanning {len(ip_list)} IP addresses (individually)...")
            print("-" * 50)
            
            for i, ip in enumerate(ip_list, 1):
                print(f"\n[*] IP {i}/{len(ip_list)}: {ip}")
                
                ip_results = scanner.scan_targets(
                    targets=[ip],
                    target_name=f"IP_{ip.replace('.', '_')}",
                    cleanup=cleanup
                )
                ip_results["scan_type"] = "ip_scan"
                ip_results["target_ip"] = ip
                results["scans"].append(ip_results)
                
                # Update summary
                if "severity_summary" in ip_results:
                    for sev, count in ip_results["severity_summary"].items():
                        results["summary"][sev] += count
                results["summary"]["total_vulnerabilities"] += ip_results.get("vulnerability_count", 0)
                results["summary"]["hosts_scanned"] += ip_results.get("hosts_scanned", 0)
                
                # Save after each IP
                save_incremental()
                print(f"    [+] Progress saved to {output_file}")
        
        # =====================================================================
        # PHASE 2: Scan Hostnames (one at a time for incremental saving)
        # =====================================================================
        if scan_targets in ("both", "hostnames_only") and hostnames:
            hostname_list = list(hostnames)
            print(f"\n[*] PHASE 2: Scanning {len(hostname_list)} hostnames (individually)...")
            print("-" * 50)
            
            for i, hostname in enumerate(hostname_list, 1):
                print(f"\n[*] Hostname {i}/{len(hostname_list)}: {hostname}")
                
                hostname_results = scanner.scan_targets(
                    targets=[hostname],
                    target_name=f"Host_{hostname.replace('.', '_')}",
                    cleanup=cleanup
                )
                hostname_results["scan_type"] = "hostname_scan"
                hostname_results["target_hostname"] = hostname
                results["scans"].append(hostname_results)
                
                # Update summary
                if "severity_summary" in hostname_results:
                    for sev, count in hostname_results["severity_summary"].items():
                        results["summary"][sev] += count
                results["summary"]["total_vulnerabilities"] += hostname_results.get("vulnerability_count", 0)
                results["summary"]["hosts_scanned"] += hostname_results.get("hosts_scanned", 0)

                # Save after each hostname
                save_incremental()
                print(f"    [+] Progress saved to {output_file}")
        
        # Final save
        save_vuln_results(results, domain)
        
    finally:
        scanner.disconnect()
    
    # Print summary
    summary = results["summary"]
    print(f"\n{'=' * 70}")
    print(f"[+] VULNERABILITY SCAN COMPLETE")
    print(f"[+] Domain: {root_domain}")
    print(f"[+] Total vulnerabilities: {summary['total_vulnerabilities']}")
    print(f"    • Critical: {summary['critical']}")
    print(f"    • High: {summary['high']}")
    print(f"    • Medium: {summary['medium']}")
    print(f"    • Low: {summary['low']}")
    print(f"    • Log: {summary['log']}")
    print(f"[+] Hosts scanned: {summary['hosts_scanned']}")
    print(f"[+] Output: {output_file}")
    print(f"{'=' * 70}")

    # Update Neo4j graph with GVM results
    graph_stats = update_graph_from_gvm_results(results)
    if "error" not in graph_stats:
        results["graph_update"] = graph_stats

    return results


def main():
    """Main entry point."""
    
    # Run the scan
    start_time = datetime.now()
    
    try:
        results = run_vulnerability_scan(
            domain=TARGET_DOMAIN,
            scan_targets=GVM_SCAN_TARGETS,
            cleanup=GVM_CLEANUP_AFTER_SCAN,
        )
        
        if "error" in results:
            print(f"\n[!] Scan failed: {results['error']}")
            return 1
        
    except KeyboardInterrupt:
        print("\n[!] Scan interrupted by user")
        return 130
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
        raise
    
    # Print duration
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n[*] Total scan time: {duration:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

