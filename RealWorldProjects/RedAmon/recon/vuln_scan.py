"""
RedAmon - Vulnerability Scanner Module
======================================
Template-based vulnerability scanning.
Enriches reconnaissance data with comprehensive web application vulnerability detection:
- CVE detection (8000+ templates)
- Web application vulnerabilities (SQLi, XSS, RCE, etc.)
- Exposed panels and sensitive files
- Misconfigurations
- Default credentials
- Cloud security issues
- Technology fingerprinting

Scans both IPs and hostnames (subdomains) for complete coverage.
Organizes results by target in the JSON output.
Supports proxy/Tor for anonymous scanning.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from recon.params import (
    NUCLEI_SEVERITY,
    NUCLEI_TEMPLATES,
    NUCLEI_EXCLUDE_TEMPLATES,
    NUCLEI_RATE_LIMIT,
    NUCLEI_BULK_SIZE,
    NUCLEI_CONCURRENCY,
    NUCLEI_TIMEOUT,
    NUCLEI_RETRIES,
    NUCLEI_TAGS,
    NUCLEI_EXCLUDE_TAGS,
    NUCLEI_DAST_MODE,
    NUCLEI_NEW_TEMPLATES_ONLY,
    NUCLEI_CUSTOM_TEMPLATES,
    NUCLEI_HEADLESS,
    NUCLEI_SYSTEM_RESOLVERS,
    NUCLEI_FOLLOW_REDIRECTS,
    NUCLEI_MAX_REDIRECTS,
    NUCLEI_SCAN_ALL_IPS,
    NUCLEI_INTERACTSH,
    NUCLEI_DOCKER_IMAGE,
    USE_TOR_FOR_RECON,
    # Katana crawler settings (for DAST mode)
    KATANA_DEPTH,
    # Template auto-update
    NUCLEI_AUTO_UPDATE_TEMPLATES,
    # CVE Lookup settings
    CVE_LOOKUP_ENABLED,
    CVE_LOOKUP_SOURCE,
    CVE_LOOKUP_MAX_CVES,
    CVE_LOOKUP_MIN_CVSS,
    VULNERS_API_KEY,
    # Security check settings (only non-redundant checks - others covered by Nuclei)
    SECURITY_CHECK_DIRECT_IP_HTTP,
    SECURITY_CHECK_DIRECT_IP_HTTPS,
    SECURITY_CHECK_IP_API_EXPOSED,
    SECURITY_CHECK_WAF_BYPASS,
    SECURITY_CHECK_ENABLED,  # Global switch to skip all security checks
    SECURITY_CHECK_TLS_EXPIRING_SOON,
    SECURITY_CHECK_TLS_EXPIRY_DAYS,
    # Security headers checks (only headers not covered by Nuclei)
    SECURITY_CHECK_MISSING_REFERRER_POLICY,
    SECURITY_CHECK_MISSING_PERMISSIONS_POLICY,
    SECURITY_CHECK_MISSING_COOP,
    SECURITY_CHECK_MISSING_CORP,
    SECURITY_CHECK_MISSING_COEP,
    SECURITY_CHECK_CACHE_CONTROL_MISSING,
    # Authentication security checks
    SECURITY_CHECK_LOGIN_NO_HTTPS,
    SECURITY_CHECK_SESSION_NO_SECURE,
    SECURITY_CHECK_SESSION_NO_HTTPONLY,
    SECURITY_CHECK_BASIC_AUTH_NO_TLS,
    # DNS security checks
    SECURITY_CHECK_SPF_MISSING,
    SECURITY_CHECK_DMARC_MISSING,
    SECURITY_CHECK_DNSSEC_MISSING,
    SECURITY_CHECK_ZONE_TRANSFER,
    # Port/Service security checks
    SECURITY_CHECK_ADMIN_PORT_EXPOSED,
    SECURITY_CHECK_DATABASE_EXPOSED,
    SECURITY_CHECK_REDIS_NO_AUTH,
    SECURITY_CHECK_KUBERNETES_API_EXPOSED,
    SECURITY_CHECK_SMTP_OPEN_RELAY,
    # Application security checks
    SECURITY_CHECK_CSP_UNSAFE_INLINE,
    SECURITY_CHECK_INSECURE_FORM_ACTION,
    # Rate limiting checks
    SECURITY_CHECK_NO_RATE_LIMITING,
    # Performance settings
    SECURITY_CHECK_TIMEOUT,
    SECURITY_CHECK_MAX_WORKERS,
)

# Import helpers from organized modules
from recon.helpers import (
    # Docker utilities
    is_docker_installed,
    is_docker_running,
    fix_file_ownership,
    pull_nuclei_docker_image,
    ensure_templates_volume,
    is_tor_running,
    # Target extraction
    extract_targets_from_recon,
    build_target_urls,
    # Nuclei helpers
    build_nuclei_command,
    parse_nuclei_finding,
    is_false_positive,
    # CVE lookup
    run_cve_lookup,
    # Security checks
    run_security_checks,
)


# =============================================================================
# Severity Definitions
# =============================================================================

SEVERITY_ORDER = ["critical", "high", "medium", "low", "info", "unknown"]

SEVERITY_COLORS = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
    "info": "⚪",
    "unknown": "⚫"
}


def run_vuln_scan(recon_data: dict, output_file: Path = None) -> dict:
    """
    Run nuclei scan on all URLs derived from recon data.
    
    Args:
        recon_data: Domain reconnaissance data dictionary
        output_file: Optional path to save incremental results
        
    Returns:
        Updated recon_data with nuclei results added
    """
    print("\n" + "=" * 70)
    print("         RedAmon - Nuclei Vulnerability Scanner")
    print("=" * 70)
    
    # Docker mode is required
    if not is_docker_installed():
        print("[!] Docker not found. Please install Docker to use Nuclei scanner.")
        print("[!] Skipping nuclei scan.")
        return recon_data
    
    if not is_docker_running():
        print("[!] Docker daemon is not running. Start it with: sudo systemctl start docker")
        print("[!] Skipping nuclei scan.")
        return recon_data
    
    # Pull image if needed (will skip if already present)
    pull_nuclei_docker_image(NUCLEI_DOCKER_IMAGE)
    
    # Ensure templates volume exists and has templates
    if not ensure_templates_volume(NUCLEI_DOCKER_IMAGE, NUCLEI_AUTO_UPDATE_TEMPLATES):
        print("[!] Could not setup nuclei templates. Skipping scan.")
        return recon_data
    
    print(f"  Execution Mode: DOCKER ({NUCLEI_DOCKER_IMAGE})")
    nuclei_version = f"Docker: {NUCLEI_DOCKER_IMAGE}"
    template_count = 8000  # Approximate, Docker image includes templates
    
    print(f"  Nuclei Version: {nuclei_version}")
    print(f"  Templates Available: ~{template_count}")
    
    # Check Tor status
    use_proxy = False
    if USE_TOR_FOR_RECON:
        if is_tor_running():
            use_proxy = True
            print(f"  [🧅] ANONYMOUS MODE: Using Tor SOCKS proxy")
        else:
            print("  [!] USE_TOR_FOR_RECON enabled but Tor not running")
            print("  [!] Falling back to direct scanning")
    
    # Extract targets
    ips, hostnames, ip_to_hostnames = extract_targets_from_recon(recon_data)
    
    if not hostnames and not ips:
        print("[!] No targets found in recon data")
        return recon_data
    
    # Build target URLs using httpx/naabu data if available
    target_urls = build_target_urls(hostnames, ips, recon_data, scan_all_ips=NUCLEI_SCAN_ALL_IPS)
    
    # For DAST mode, we need URLs with parameters from resource_enum
    dast_urls = []
    if NUCLEI_DAST_MODE:
        print(f"  DAST Mode: ENABLED (active fuzzing for XSS, SQLi, etc.)")

        # Get URLs with parameters from resource_enum (must be run before vuln_scan)
        resource_enum_data = recon_data.get("resource_enum")
        if resource_enum_data:
            discovered_urls = resource_enum_data.get("discovered_urls", [])
            # Filter for URLs with parameters
            dast_urls = [url for url in discovered_urls if '?' in url and '=' in url]
            if dast_urls:
                print(f"  [*] Using {len(dast_urls)} URLs with parameters from resource_enum")
            else:
                print(f"  [!] No URLs with parameters found in resource_enum - DAST scan may not find vulnerabilities")
        else:
            print(f"  [!] resource_enum not found - run resource_enum before vuln_scan for DAST mode")
    
    print(f"  Unique Hostnames: {len(hostnames)}")
    print(f"  Unique IPs: {len(ips)}")
    print(f"  Base URLs: {len(target_urls)}")
    if NUCLEI_DAST_MODE and dast_urls:
        print(f"  DAST URLs (with params): {len(dast_urls)}")
    print(f"  Scan IPs: {'YES' if NUCLEI_SCAN_ALL_IPS else 'NO (hostnames only)'}")
    print(f"  Severity Filter: {', '.join(NUCLEI_SEVERITY) if NUCLEI_SEVERITY else 'ALL'}")
    print(f"  Rate Limit: {NUCLEI_RATE_LIMIT} req/s")
    if NUCLEI_TAGS:
        print(f"  Tags: {', '.join(NUCLEI_TAGS)}")
    print("=" * 70 + "\n")
    
    # Create a temporary directory for nuclei files in a location Docker can access
    # Using the project's output directory to avoid permission/access issues
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nuclei_temp_dir = output_dir / ".nuclei_temp"
    nuclei_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create targets file
    # For DAST mode with discovered URLs, use those; otherwise use base URLs
    scan_urls = target_urls
    if NUCLEI_DAST_MODE and dast_urls:
        # Combine DAST URLs with base URLs for comprehensive coverage
        scan_urls = list(set(target_urls + dast_urls))
        print(f"[*] DAST scan will test {len(dast_urls)} URLs with parameters + {len(target_urls)} base URLs")
    
    targets_file = str(nuclei_temp_dir / "targets.txt")
    with open(targets_file, 'w') as f:
        for url in scan_urls:
            f.write(url + "\n")
    
    # Output file path
    nuclei_output_file = str(nuclei_temp_dir / "nuclei_output.jsonl")
    
    try:
        # Build and run nuclei command
        cmd = build_nuclei_command(
            targets_file=targets_file,
            output_file=nuclei_output_file,
            docker_image=NUCLEI_DOCKER_IMAGE,
            use_proxy=use_proxy,
            severity=NUCLEI_SEVERITY,
            templates=NUCLEI_TEMPLATES,
            exclude_templates=NUCLEI_EXCLUDE_TEMPLATES,
            custom_templates=NUCLEI_CUSTOM_TEMPLATES,
            tags=NUCLEI_TAGS,
            exclude_tags=NUCLEI_EXCLUDE_TAGS,
            rate_limit=NUCLEI_RATE_LIMIT,
            bulk_size=NUCLEI_BULK_SIZE,
            concurrency=NUCLEI_CONCURRENCY,
            timeout=NUCLEI_TIMEOUT,
            retries=NUCLEI_RETRIES,
            dast_mode=NUCLEI_DAST_MODE,
            new_templates_only=NUCLEI_NEW_TEMPLATES_ONLY,
            headless=NUCLEI_HEADLESS,
            system_resolvers=NUCLEI_SYSTEM_RESOLVERS,
            follow_redirects=NUCLEI_FOLLOW_REDIRECTS,
            max_redirects=NUCLEI_MAX_REDIRECTS,
            interactsh=NUCLEI_INTERACTSH,
        )
        
        print(f"[*] Running nuclei scan [DOCKER]...")
        print(f"[*] Command: {' '.join(cmd[:12])}...")
        
        # Run nuclei
        start_time = datetime.now()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor progress
        stdout, stderr = process.communicate()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if process.returncode != 0 and stderr:
            # Filter out common non-error messages
            error_lines = [l for l in stderr.split('\n') if l and 'WRN' not in l and 'INF' not in l]
            if error_lines:
                print(f"[!] Nuclei warnings: {error_lines[0][:100]}")
        
        # Parse results and filter false positives
        findings = []
        false_positives_filtered = []
        
        if Path(nuclei_output_file).exists():
            with open(nuclei_output_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            raw_finding = json.loads(line)
                            
                            # Check for false positives before parsing
                            is_fp, fp_reason = is_false_positive(raw_finding)
                            if is_fp:
                                # Log the filtered false positive
                                template_id = raw_finding.get("template-id", "unknown")
                                matched_at = raw_finding.get("matched-at", "unknown")
                                false_positives_filtered.append({
                                    "template_id": template_id,
                                    "matched_at": matched_at,
                                    "reason": fp_reason
                                })
                                continue  # Skip this finding
                            
                            parsed = parse_nuclei_finding(raw_finding)
                            findings.append(parsed)
                        except json.JSONDecodeError:
                            continue
        
        # Log filtered false positives
        if false_positives_filtered:
            print(f"    [*] Filtered {len(false_positives_filtered)} false positive(s):")
            for fp in false_positives_filtered[:5]:  # Show first 5
                print(f"        - {fp['template_id']}: {fp['reason'][:60]}...")
            if len(false_positives_filtered) > 5:
                print(f"        ... and {len(false_positives_filtered) - 5} more")
        
        # Organize results
        nuclei_results = {
            "scan_metadata": {
                "scan_timestamp": start_time.isoformat(),
                "scan_duration_seconds": duration,
                "nuclei_version": nuclei_version,
                "templates_available": template_count,
                "execution_mode": "docker",
                "docker_image": NUCLEI_DOCKER_IMAGE,
                "anonymous_mode": use_proxy,
                "severity_filter": NUCLEI_SEVERITY,
                "tags_filter": NUCLEI_TAGS,
                "exclude_tags": NUCLEI_EXCLUDE_TAGS,
                "rate_limit": NUCLEI_RATE_LIMIT,
                "dast_mode": NUCLEI_DAST_MODE,
                "dast_urls_discovered": len(dast_urls) if NUCLEI_DAST_MODE else 0,
                "katana_crawl_depth": KATANA_DEPTH if NUCLEI_DAST_MODE else None,
                "total_urls_scanned": len(scan_urls),
                "total_hostnames": len(hostnames),
                "total_ips": len(ips),
                "false_positives_filtered": len(false_positives_filtered),
            },
            "discovered_urls": {
                "base_urls": sorted(target_urls),
                "dast_urls_with_params": sorted(dast_urls) if dast_urls else [],
                "all_scanned_urls": sorted(scan_urls),
            },
            "by_target": {},
            "summary": {
                "total_findings": len(findings),
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "info": 0,
                "unknown": 0,
            },
            "vulnerabilities": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "info": 0,
                "unknown": 0,
            },
            "all_cves": [],
            "by_category": {},
            "by_template": {},
            "false_positives": false_positives_filtered if false_positives_filtered else [],
        }
        
        # Process findings
        all_cves = []
        
        for finding in findings:
            severity = finding["severity"]
            target = finding["target"]
            template_id = finding["template_id"]
            category = finding["category"]
            
            # Count by severity
            if severity in nuclei_results["summary"]:
                nuclei_results["summary"][severity] += 1
            else:
                nuclei_results["summary"]["unknown"] += 1
            
            # Group by target
            if target not in nuclei_results["by_target"]:
                nuclei_results["by_target"][target] = {
                    "findings": [],
                    "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
                }
            nuclei_results["by_target"][target]["findings"].append(finding)
            if severity in nuclei_results["by_target"][target]["severity_counts"]:
                nuclei_results["by_target"][target]["severity_counts"][severity] += 1
            
            # Add to severity-based vulnerability list
            finding_summary = {
                "template_id": template_id,
                "name": finding["name"],
                "target": target,
                "matched_at": finding["matched_at"],
                "category": category,
                "cves": [c["id"] for c in finding["cves"]],
                "cvss": finding["cvss_score"],
            }
            
            if severity in nuclei_results["vulnerabilities"]:
                nuclei_results["vulnerabilities"][severity] += 1
            else:
                nuclei_results["vulnerabilities"]["unknown"] += 1
            
            # Collect CVEs
            all_cves.extend(finding["cves"])
            
            # Group by category
            if category not in nuclei_results["by_category"]:
                nuclei_results["by_category"][category] = []
            nuclei_results["by_category"][category].append(finding_summary)
            
            # Group by template
            if template_id not in nuclei_results["by_template"]:
                nuclei_results["by_template"][template_id] = {
                    "name": finding["name"],
                    "severity": severity,
                    "findings_count": 0,
                    "targets": []
                }
            nuclei_results["by_template"][template_id]["findings_count"] += 1
            if target not in nuclei_results["by_template"][template_id]["targets"]:
                nuclei_results["by_template"][template_id]["targets"].append(target)
        
        # Deduplicate and sort CVEs
        seen_cves = set()
        unique_cves = []
        for cve in all_cves:
            if cve["id"] not in seen_cves:
                seen_cves.add(cve["id"])
                unique_cves.append(cve)
        unique_cves.sort(key=lambda x: x.get("cvss") or 0, reverse=True)
        nuclei_results["all_cves"] = unique_cves
        
        # Add to recon data
        recon_data["vuln_scan"] = nuclei_results
        
        # Save incrementally if output file provided
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(recon_data, f, indent=2)
            fix_file_ownership(output_file)  # Ensure correct ownership when running under sudo
        
        # Print summary
        print(f"\n{'=' * 70}")
        print(f"[+] NUCLEI SCAN COMPLETE")
        print(f"[+] Duration: {duration:.2f} seconds")
        print(f"[+] Execution mode: DOCKER")
        if use_proxy:
            print(f"[+] Anonymous mode: YES (via Tor)")
        print(f"[+] URLs scanned: {len(target_urls)}")
        print(f"[+] Total findings: {len(findings)}")
        
        # Vulnerability summary
        summary = nuclei_results["summary"]
        vuln_total = summary["total_findings"]

        if vuln_total > 0:
            print(f"\n[+] VULNERABILITY SUMMARY:")
            if summary['critical'] > 0:
                print(f"    🔴 CRITICAL: {summary['critical']}")
            if summary['high'] > 0:
                print(f"    🟠 HIGH: {summary['high']}")
            if summary['medium'] > 0:
                print(f"    🟡 MEDIUM: {summary['medium']}")
            if summary['low'] > 0:
                print(f"    🔵 LOW: {summary['low']}")
        
        if summary['info'] > 0:
            print(f"    ⚪ INFO: {summary['info']}")
        
        # CVE summary
        cve_count = len(unique_cves)
        if cve_count > 0:
            print(f"\n[+] CVEs FOUND: {cve_count}")
            for cve in unique_cves[:5]:
                cvss_str = f"CVSS {cve['cvss']}" if cve.get('cvss') else "CVSS N/A"
                print(f"    - {cve['id']} ({cvss_str})")
            if cve_count > 5:
                print(f"    ... and {cve_count - 5} more")
        
        # Top affected targets
        if nuclei_results["by_target"]:
            print(f"\n[+] FINDINGS BY TARGET:")
            sorted_targets = sorted(
                nuclei_results["by_target"].items(),
                key=lambda x: len(x[1]["findings"]),
                reverse=True
            )[:5]
            for target, data in sorted_targets:
                counts = data["severity_counts"]
                count_str = ", ".join([
                    f"{SEVERITY_COLORS.get(s, '')}{counts[s]}" 
                    for s in ["critical", "high", "medium", "low", "info"] 
                    if counts.get(s, 0) > 0
                ])
                print(f"    - {target[:50]}: {len(data['findings'])} findings ({count_str})")
        
        # Top categories
        if nuclei_results["by_category"]:
            print(f"\n[+] TOP VULNERABILITY CATEGORIES:")
            sorted_cats = sorted(
                nuclei_results["by_category"].items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:5]
            for cat, findings_list in sorted_cats:
                print(f"    - {cat}: {len(findings_list)} findings")
        
        print(f"{'=' * 70}")
        
        # Run CVE lookup for detected technologies (like Nmap's vulners)
        if CVE_LOOKUP_ENABLED and recon_data.get("http_probe"):
            cve_results = run_cve_lookup(
                recon_data=recon_data,
                enabled=CVE_LOOKUP_ENABLED,
                source=CVE_LOOKUP_SOURCE,
                max_cves=CVE_LOOKUP_MAX_CVES,
                min_cvss=CVE_LOOKUP_MIN_CVSS,
                vulners_api_key=VULNERS_API_KEY,
            )
            recon_data.update(cve_results)

            # Save with CVE data
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(recon_data, f, indent=2)
                fix_file_ownership(output_file)

        # Run custom security checks (Direct IP Access, TLS/SSL, Security Headers)
        # Skip entirely if global switch is disabled
        if not SECURITY_CHECK_ENABLED:
            print(f"\n[*] Custom security checks disabled (SECURITY_CHECK_ENABLED=False)")
        else:
            security_checks_enabled = {
                # Direct IP Access checks (unique - not covered by Nuclei)
                "direct_ip_http": SECURITY_CHECK_DIRECT_IP_HTTP,
                "direct_ip_https": SECURITY_CHECK_DIRECT_IP_HTTPS,
                "ip_api_exposed": SECURITY_CHECK_IP_API_EXPOSED,
                "waf_bypass": SECURITY_CHECK_WAF_BYPASS,
                # TLS/SSL checks (only expiring soon - others covered by Nuclei)
                "tls_expiring_soon": SECURITY_CHECK_TLS_EXPIRING_SOON,
                # Security Headers checks (only headers not covered by Nuclei)
                "missing_referrer_policy": SECURITY_CHECK_MISSING_REFERRER_POLICY,
                "missing_permissions_policy": SECURITY_CHECK_MISSING_PERMISSIONS_POLICY,
                "missing_coop": SECURITY_CHECK_MISSING_COOP,
                "missing_corp": SECURITY_CHECK_MISSING_CORP,
                "missing_coep": SECURITY_CHECK_MISSING_COEP,
                "cache_control_missing": SECURITY_CHECK_CACHE_CONTROL_MISSING,
                # Authentication security checks
                "login_no_https": SECURITY_CHECK_LOGIN_NO_HTTPS,
                "session_no_secure": SECURITY_CHECK_SESSION_NO_SECURE,
                "session_no_httponly": SECURITY_CHECK_SESSION_NO_HTTPONLY,
                "basic_auth_no_tls": SECURITY_CHECK_BASIC_AUTH_NO_TLS,
                # DNS security checks
                "spf_missing": SECURITY_CHECK_SPF_MISSING,
                "dmarc_missing": SECURITY_CHECK_DMARC_MISSING,
                "dnssec_missing": SECURITY_CHECK_DNSSEC_MISSING,
                "zone_transfer": SECURITY_CHECK_ZONE_TRANSFER,
                # Port/Service security checks
                "admin_port_exposed": SECURITY_CHECK_ADMIN_PORT_EXPOSED,
                "database_exposed": SECURITY_CHECK_DATABASE_EXPOSED,
                "redis_no_auth": SECURITY_CHECK_REDIS_NO_AUTH,
                "kubernetes_api_exposed": SECURITY_CHECK_KUBERNETES_API_EXPOSED,
                "smtp_open_relay": SECURITY_CHECK_SMTP_OPEN_RELAY,
                # Application security checks
                "csp_unsafe_inline": SECURITY_CHECK_CSP_UNSAFE_INLINE,
                "insecure_form_action": SECURITY_CHECK_INSECURE_FORM_ACTION,
                # Rate limiting checks
                "no_rate_limiting": SECURITY_CHECK_NO_RATE_LIMITING,
            }

            # Only run if at least one check is enabled
            if any(security_checks_enabled.values()):
                security_results = run_security_checks(
                    recon_data=recon_data,
                    enabled_checks=security_checks_enabled,
                    timeout=SECURITY_CHECK_TIMEOUT,
                    tls_expiry_days=SECURITY_CHECK_TLS_EXPIRY_DAYS,
                    max_workers=SECURITY_CHECK_MAX_WORKERS
                )

                # Merge security checks into vuln_scan results
                if "vuln_scan" in recon_data:
                    recon_data["vuln_scan"]["security_checks"] = security_results.get("security_checks", {})
                else:
                    recon_data["vuln_scan"] = {"security_checks": security_results.get("security_checks", {})}

                # Save with security check data
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(recon_data, f, indent=2)
                    fix_file_ownership(output_file)

    finally:
        # Cleanup temporary files and directory
        # Docker may create files as root, so we use Docker to clean up if needed
        try:
            Path(targets_file).unlink(missing_ok=True)
        except PermissionError:
            # File owned by root (from Docker), use docker to remove it
            subprocess.run(["docker", "run", "--rm", "-v", f"{nuclei_temp_dir}:/cleanup", 
                          "alpine", "rm", "-f", f"/cleanup/{Path(targets_file).name}"],
                         capture_output=True)
        
        try:
            Path(nuclei_output_file).unlink(missing_ok=True)
        except PermissionError:
            subprocess.run(["docker", "run", "--rm", "-v", f"{nuclei_temp_dir}:/cleanup",
                          "alpine", "rm", "-f", f"/cleanup/{Path(nuclei_output_file).name}"],
                         capture_output=True)
        
        try:
            nuclei_temp_dir.rmdir()  # Only removes if empty
        except Exception:
            pass
    
    return recon_data


def enrich_recon_file(recon_file: Path) -> dict:
    """
    Load a recon JSON file, enrich it with nuclei data, and save it back.
    
    Args:
        recon_file: Path to the recon JSON file
        
    Returns:
        Enriched recon data
    """
    # Load existing data
    with open(recon_file, 'r') as f:
        recon_data = json.load(f)
    
    # Run nuclei scan
    enriched_data = run_vuln_scan(recon_data, output_file=recon_file)
    
    # Save enriched data
    with open(recon_file, 'w') as f:
        json.dump(enriched_data, f, indent=2)
    
    print(f"[+] Enriched data saved to: {recon_file}")
    
    return enriched_data

