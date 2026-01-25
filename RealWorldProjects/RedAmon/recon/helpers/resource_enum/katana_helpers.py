"""
RedAmon - Katana Crawler Helpers for Resource Enumeration
=========================================================
Active URL discovery using Katana web crawler.
"""

import subprocess
import ssl
import urllib.request
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from .form_helpers import parse_forms_from_html


def run_katana_crawler(
    target_urls: List[str],
    docker_image: str,
    depth: int,
    max_urls: int,
    rate_limit: int,
    timeout: int,
    js_crawl: bool,
    params_only: bool,
    scope: str,
    custom_headers: List[str],
    exclude_patterns: List[str],
    use_proxy: bool = False
) -> Tuple[List[str], Dict[str, str]]:
    """
    Run Katana crawler to discover all endpoints.

    Args:
        target_urls: Base URLs to crawl
        docker_image: Katana Docker image name
        depth: Crawl depth
        max_urls: Maximum URLs to discover
        rate_limit: Requests per second limit
        timeout: Request timeout in seconds
        js_crawl: Whether to enable JavaScript crawling
        params_only: Only return URLs with parameters
        scope: Scope setting for crawling
        custom_headers: Custom headers to send
        exclude_patterns: URL patterns to exclude
        use_proxy: Whether to use Tor proxy

    Returns:
        Tuple of (discovered_urls, url_to_response_body)
    """
    print(f"\n[*] Running Katana crawler for endpoint discovery...")
    print(f"    Crawl depth: {depth}")
    print(f"    Max URLs: {max_urls}")
    print(f"    Rate limit: {rate_limit} req/s")
    print(f"    Params only: {params_only}")

    discovered_urls = set()

    for base_url in target_urls:
        if not base_url.startswith(('http://', 'https://')):
            continue

        # Build Katana command
        cmd = ["docker", "run", "--rm"]

        if use_proxy:
            cmd.extend(["--network", "host"])

        cmd.extend(["-v", "/tmp:/tmp"])

        cmd.extend([
            docker_image,
            "-u", base_url,
            "-d", str(depth),
            "-silent",
            "-nc",
            "-rl", str(rate_limit),
            "-timeout", str(timeout),
            "-fs", scope,
        ])

        # JavaScript crawling
        if js_crawl:
            cmd.append("-jc")

        # Custom headers
        if custom_headers:
            for header in custom_headers:
                cmd.extend(["-H", header])

        # Proxy for Tor
        if use_proxy:
            cmd.extend(["-proxy", "socks5://127.0.0.1:9050"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 60
            )

            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    url = line.strip()
                    if url:
                        # Skip URLs matching exclude patterns
                        url_lower = url.lower()
                        if any(pattern.lower() in url_lower for pattern in exclude_patterns):
                            continue

                        # Apply params_only filter
                        if params_only:
                            if '?' in url and '=' in url:
                                discovered_urls.add(url)
                        else:
                            discovered_urls.add(url)

                        if len(discovered_urls) >= max_urls:
                            break

        except subprocess.TimeoutExpired:
            print(f"    [!] Katana timeout for {base_url}")
        except Exception as e:
            print(f"    [!] Katana error for {base_url}: {e}")

        if len(discovered_urls) >= max_urls:
            break

    urls_list = sorted(list(discovered_urls))
    print(f"    [+] Katana discovered {len(urls_list)} URLs")

    return urls_list, {}


def fetch_forms_from_urls(
    urls: List[str],
    use_proxy: bool = False,
    max_urls: int = 50
) -> List[Dict]:
    """
    Fetch HTML from URLs and extract forms.

    Args:
        urls: URLs to fetch (will filter to HTML pages only)
        use_proxy: Whether to use Tor proxy
        max_urls: Maximum URLs to fetch for form extraction

    Returns:
        List of form dictionaries
    """
    all_forms = []

    # Filter to likely HTML pages (exclude static files)
    static_extensions = ['.css', '.js', '.jpg', '.jpeg', '.png', '.gif', '.svg',
                         '.ico', '.woff', '.woff2', '.ttf', '.eot', '.pdf', '.zip',
                         '.mp3', '.mp4', '.webp', '.xml', '.json', '.txt']

    html_urls = []
    for url in urls:
        url_lower = url.lower().split('?')[0]  # Remove query params for extension check
        if not any(url_lower.endswith(ext) for ext in static_extensions):
            html_urls.append(url)

    # Limit to avoid too many requests
    html_urls = html_urls[:max_urls]

    if not html_urls:
        return all_forms

    print(f"    [*] Fetching HTML from {len(html_urls)} URLs to extract forms...")

    # Create SSL context that doesn't verify certificates (for testing)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Setup proxy if needed
    if use_proxy:
        proxy_handler = urllib.request.ProxyHandler({
            'http': 'socks5://127.0.0.1:9050',
            'https': 'socks5://127.0.0.1:9050'
        })
        opener = urllib.request.build_opener(proxy_handler, urllib.request.HTTPSHandler(context=ssl_context))
    else:
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))

    for url in html_urls:
        try:
            request = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                }
            )
            response = opener.open(request, timeout=10)
            content_type = response.headers.get('Content-Type', '')

            # Only process HTML responses
            if 'text/html' in content_type:
                html_content = response.read().decode('utf-8', errors='ignore')
                forms = parse_forms_from_html(html_content, url)
                all_forms.extend(forms)

        except Exception:
            continue

    print(f"    [+] Extracted {len(all_forms)} forms from HTML pages")
    return all_forms


def pull_katana_docker_image(docker_image: str) -> bool:
    """
    Pull the Katana Docker image if not present.
    
    Args:
        docker_image: Docker image name to pull
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"    [*] Pulling Katana image: {docker_image}...")
        result = subprocess.run(
            ["docker", "pull", docker_image],
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0
    except Exception:
        return False

