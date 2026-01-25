"""
RedAmon - CVE Lookup Helpers
============================
Functions for looking up CVEs from NVD and Vulners APIs based on detected technologies.
"""

import re
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# =============================================================================
# API URLs
# =============================================================================

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
VULNERS_API_URL = "https://vulners.com/api/v3/burp/software/"


# =============================================================================
# CPE Mappings for Common Technologies
# =============================================================================

CPE_MAPPINGS = {
    # Web Servers
    "nginx": ("f5", "nginx"),
    "apache": ("apache", "http_server"),
    "iis": ("microsoft", "internet_information_services"),
    "tomcat": ("apache", "tomcat"),
    "lighttpd": ("lighttpd", "lighttpd"),
    "caddy": ("caddyserver", "caddy"),
    # Languages/Runtimes
    "php": ("php", "php"),
    "python": ("python", "python"),
    "node.js": ("nodejs", "node.js"),
    "ruby": ("ruby-lang", "ruby"),
    # Databases
    "mysql": ("oracle", "mysql"),
    "mariadb": ("mariadb", "mariadb"),
    "postgresql": ("postgresql", "postgresql"),
    "mongodb": ("mongodb", "mongodb"),
    "redis": ("redis", "redis"),
    "elasticsearch": ("elastic", "elasticsearch"),
    # CMS/Frameworks
    "wordpress": ("wordpress", "wordpress"),
    "drupal": ("drupal", "drupal"),
    "joomla": ("joomla", "joomla"),
    "django": ("djangoproject", "django"),
    "laravel": ("laravel", "laravel"),
    "spring": ("vmware", "spring_framework"),
    # JavaScript
    "jquery": ("jquery", "jquery"),
    "angular": ("angular", "angular"),
    "react": ("facebook", "react"),
    "vue": ("vuejs", "vue.js"),
    "bootstrap": ("getbootstrap", "bootstrap"),
    # Security
    "openssh": ("openbsd", "openssh"),
    "openssl": ("openssl", "openssl"),
    # Other
    "varnish": ("varnish-software", "varnish_cache"),
    "grafana": ("grafana", "grafana"),
    "jenkins": ("jenkins", "jenkins"),
    "gitlab": ("gitlab", "gitlab"),
    "haproxy": ("haproxy", "haproxy"),
}


# =============================================================================
# Technology Parsing Utilities
# =============================================================================

def parse_technology_string(tech: str) -> Tuple[str, Optional[str]]:
    """Parse technology string like 'Nginx:1.19.0' into (name, version)."""
    tech = tech.strip()
    for delimiter in [':', '/', ' ']:
        if delimiter in tech:
            parts = tech.split(delimiter, 1)
            name = parts[0].strip().lower()
            version = parts[1].strip() if len(parts) > 1 else None
            if version:
                version = re.sub(r'^v', '', version)
            return name, version
    return tech.lower(), None


def normalize_product_name(name: str) -> str:
    """Normalize product name for lookup."""
    name = name.lower().strip()
    aliases = {
        "nginx": "nginx", "apache httpd": "apache", "microsoft-iis": "iis",
        "node": "node.js", "nodejs": "node.js", "postgres": "postgresql",
        "mongo": "mongodb", "wp": "wordpress", "ssh": "openssh",
    }
    return aliases.get(name, name)


def classify_cvss_score(score: float) -> str:
    """Classify CVSS score into severity level."""
    if score is None:
        return "unknown"
    if score >= 9.0:
        return "CRITICAL"
    if score >= 7.0:
        return "HIGH"
    if score >= 4.0:
        return "MEDIUM"
    if score >= 0.1:
        return "LOW"
    return "NONE"


# =============================================================================
# NVD API Lookup
# =============================================================================

def lookup_cves_nvd(
    product: str, 
    version: str = None, 
    max_results: int = 20,
    api_key: str = None
) -> List[Dict]:
    """
    Query NVD API for CVEs affecting a product/version.
    
    Args:
        product: Product name (e.g., 'nginx')
        version: Version string (e.g., '1.19.0')
        max_results: Maximum results to return
        api_key: Optional NVD API key for higher rate limits
        
    Returns:
        List of CVE dictionaries
    """
    cves = []
    product_normalized = normalize_product_name(product)
    cpe_info = CPE_MAPPINGS.get(product_normalized)

    params = {"resultsPerPage": max_results}
    headers = {}

    # Add API key if available
    if api_key:
        headers["apiKey"] = api_key

    if cpe_info and version:
        vendor, prod = cpe_info
        params["cpeName"] = f"cpe:2.3:a:{vendor}:{prod}:{version}:*:*:*:*:*:*:*"
    elif cpe_info:
        vendor, prod = cpe_info
        params["cpeName"] = f"cpe:2.3:a:{vendor}:{prod}:*:*:*:*:*:*:*:*"
    else:
        # Fallback to keyword search for unknown products
        keyword = product
        if version:
            keyword += f" {version}"
        params["keywordSearch"] = keyword

    try:
        response = requests.get(NVD_API_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        for vuln in data.get("vulnerabilities", []):
            cve_data = vuln.get("cve", {})
            cve_id = cve_data.get("id", "")
            
            metrics = cve_data.get("metrics", {})
            cvss_v3 = metrics.get("cvssMetricV31", [{}])[0] if metrics.get("cvssMetricV31") else None
            cvss_v2 = metrics.get("cvssMetricV2", [{}])[0] if metrics.get("cvssMetricV2") else None
            
            cvss_score = None
            severity = None
            
            if cvss_v3:
                cvss_score = cvss_v3.get("cvssData", {}).get("baseScore")
                severity = cvss_v3.get("cvssData", {}).get("baseSeverity")
            elif cvss_v2:
                cvss_score = cvss_v2.get("cvssData", {}).get("baseScore")
                severity = cvss_v2.get("baseSeverity")
            
            descriptions = cve_data.get("descriptions", [])
            description = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
            
            refs = cve_data.get("references", [])
            reference_urls = [ref.get("url") for ref in refs[:3] if ref.get("url")]
            
            cves.append({
                "id": cve_id,
                "cvss": cvss_score,
                "severity": severity,
                "description": description[:300] if description else "",
                "published": cve_data.get("published"),
                "references": reference_urls,
                "source": "nvd",
                "url": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
            })
            
    except Exception as e:
        print(f"        [!] NVD API error: {str(e)[:80]}")
    
    return cves


# =============================================================================
# Vulners API Lookup
# =============================================================================

def lookup_cves_vulners(product: str, version: str, api_key: str = None) -> List[Dict]:
    """
    Query Vulners API for CVEs (like Nmap's vulners script).
    
    Args:
        product: Product name
        version: Version string (required for Vulners)
        api_key: Vulners API key
        
    Returns:
        List of CVE dictionaries
    """
    cves = []
    if not version:
        return cves
    
    params = {"software": f"{product} {version}", "version": version, "type": "software"}
    if api_key:
        params["apiKey"] = api_key
    
    try:
        response = requests.get(VULNERS_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("result") == "OK":
            for vuln in data.get("data", {}).get("search", []):
                vuln_id = vuln.get("id", "")
                cvss_data = vuln.get("cvss", {})
                
                cves.append({
                    "id": vuln_id,
                    "cvss": cvss_data.get("score"),
                    "severity": classify_cvss_score(cvss_data.get("score")),
                    "description": vuln.get("description", "")[:300],
                    "published": vuln.get("published"),
                    "references": [vuln.get("href")] if vuln.get("href") else [],
                    "source": "vulners",
                    "url": f"https://vulners.com/{vuln.get('type', 'cve')}/{vuln_id}",
                })
    except Exception as e:
        print(f"        [!] Vulners API error: {str(e)[:80]}")
    
    return cves


# =============================================================================
# Main CVE Lookup Orchestration
# =============================================================================

def run_cve_lookup(
    recon_data: dict,
    enabled: bool = True,
    source: str = "nvd",
    max_cves: int = 20,
    min_cvss: float = 0.0,
    vulners_api_key: str = None,
    nvd_api_key: str = None,
) -> Dict:
    """
    Lookup CVEs for all technologies detected by httpx.
    
    Args:
        recon_data: Reconnaissance data containing httpx results
        enabled: Whether CVE lookup is enabled
        source: API source ('nvd' or 'vulners')
        max_cves: Maximum CVEs per technology
        min_cvss: Minimum CVSS score to include
        vulners_api_key: Vulners API key
        nvd_api_key: NVD API key
        
    Returns:
        Dictionary to add to recon_data
    """
    if not enabled:
        return {}
    
    print(f"\n{'='*60}")
    print("CVE LOOKUP - Technology-Based Vulnerability Discovery")
    print(f"{'='*60}")
    print(f"    Source: {source.upper()}")
    print(f"    Min CVSS: {min_cvss}")
    
    # Extract technologies from httpx
    technologies = set()
    httpx_data = recon_data.get("http_probe", {})
    
    for url_data in httpx_data.get("by_url", {}).values():
        techs = url_data.get("technologies", [])
        technologies.update(techs)
        server = url_data.get("server")
        if server:
            technologies.add(server)
    
    # Filter technologies to lookup
    tech_to_lookup = []
    skip_list = ["ubuntu", "debian", "centos", "linux", "windows", 
                 "dreamweaver", "frontpage", "html", "css", "aws"]
    
    for tech in technologies:
        name, version = parse_technology_string(tech)
        name = normalize_product_name(name)
        if not version or name in skip_list:
            continue
        tech_to_lookup.append(tech)
    
    print(f"\n[*] Technologies with versions: {len(tech_to_lookup)}")
    
    if not tech_to_lookup:
        print("[!] No technologies with versions found")
        return {"technology_cves": {"summary": {"total_cves": 0}}}
    
    # Lookup CVEs
    cve_results = {}
    all_cves = []
    
    for i, tech in enumerate(tech_to_lookup, 1):
        name, version = parse_technology_string(tech)
        name = normalize_product_name(name)
        
        print(f"    [{i}/{len(tech_to_lookup)}] {tech}...", end=" ", flush=True)
        
        if source == "vulners" and vulners_api_key:
            cves = lookup_cves_vulners(name, version, vulners_api_key)
        else:
            cves = lookup_cves_nvd(name, version, max_cves, nvd_api_key)
        
        # Filter by min CVSS
        if min_cvss > 0:
            cves = [c for c in cves if (c.get("cvss") or 0) >= min_cvss]
        
        cves.sort(key=lambda x: x.get("cvss") or 0, reverse=True)
        cves = cves[:max_cves]
        
        if cves:
            cve_results[tech] = {
                "technology": tech,
                "product": name,
                "version": version,
                "cve_count": len(cves),
                "critical": len([c for c in cves if c.get("severity") == "CRITICAL"]),
                "high": len([c for c in cves if c.get("severity") == "HIGH"]),
                "cves": cves,
            }
            all_cves.extend(cves)
            print(f"✓ {len(cves)} CVEs found")
        else:
            print("no CVEs")
        
        # Rate limiting for NVD API
        if source == "nvd" and i < len(tech_to_lookup):
            time.sleep(6)
    
    # Count unique CVEs
    unique_cve_ids = set()
    for tech_data in cve_results.values():
        for cve in tech_data.get("cves", []):
            unique_cve_ids.add(cve["id"])

    # Count severity distribution
    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for tech_data in cve_results.values():
        for cve in tech_data.get("cves", []):
            sev = cve.get("severity", "").upper()
            if sev in severity_counts:
                severity_counts[sev] += 1

    # Build result
    result = {
        "technology_cves": {
            "lookup_timestamp": datetime.now().isoformat(),
            "source": source,
            "technologies_checked": len(tech_to_lookup),
            "technologies_with_cves": len(cve_results),
            "by_technology": cve_results,
            "summary": {
                "total_unique_cves": len(unique_cve_ids),
                "critical": severity_counts["CRITICAL"],
                "high": severity_counts["HIGH"],
                "medium": severity_counts["MEDIUM"],
                "low": severity_counts["LOW"],
            }
        }
    }
    
    # Print summary
    summary = result["technology_cves"]["summary"]
    print(f"\n[+] CVE LOOKUP SUMMARY:")
    print(f"    Total unique CVEs: {summary['total_unique_cves']}")
    if summary['critical'] > 0:
        print(f"    🔴 CRITICAL: {summary['critical']}")
    if summary['high'] > 0:
        print(f"    🟠 HIGH: {summary['high']}")
    if summary['medium'] > 0:
        print(f"    🟡 MEDIUM: {summary['medium']}")
    print(f"{'='*60}")
    
    return result

