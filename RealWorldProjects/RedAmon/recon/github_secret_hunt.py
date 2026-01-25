#!/usr/bin/env python3
"""
RedAmon - GitHub Secret Hunter
====================================
Advanced reconnaissance tool for finding leaked secrets, credentials,
and sensitive data in GitHub repositories.

Features:
- 40+ regex patterns for common secrets (AWS, Azure, GCP, Stripe, etc.)
- High-entropy string detection for unknown secret formats
- Commit history scanning to find deleted secrets
- Organization member and gist scanning
- Sensitive filename detection
- Rate limit handling with automatic retry
- JSON output for integration with other tools
"""

import re
import os
import sys
import json
import math
import time
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, List, Set
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from github import Github, Auth
    from github.GithubException import RateLimitExceededException, GithubException
except ImportError:
    print("[!] PyGithub not installed. Run: pip install PyGithub")
    sys.exit(1)

from recon.params import (
    GITHUB_ACCESS_TOKEN,
    GITHUB_TARGET_ORG,
    GITHUB_SCAN_MEMBERS,
    GITHUB_SCAN_GISTS,
    GITHUB_SCAN_COMMITS,
    GITHUB_MAX_COMMITS,
    GITHUB_OUTPUT_JSON,
)

# =============================================================================
# SECRET PATTERNS - Comprehensive regex patterns for secret detection
# =============================================================================

SECRET_PATTERNS = {
    # AWS
    "AWS Access Key ID": r"AKIA[0-9A-Z]{16}",
    "AWS Secret Key": r"(?i)aws(.{0,20})?(?-i)['\"][0-9a-zA-Z/+]{40}['\"]",
    "AWS MWS Key": r"amzn\\.mws\\.[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    
    # Azure
    "Azure Storage Key": r"(?i)DefaultEndpointsProtocol=https;AccountName=[^;]+;AccountKey=[A-Za-z0-9+/=]{88}",
    "Azure Connection String": r"(?i)(AccountKey|SharedAccessKey)=[A-Za-z0-9+/=]{40,}",
    "Azure SAS Token": r"(?i)[?&]sig=[A-Za-z0-9%]{40,}",
    
    # Google Cloud
    "GCP API Key": r"AIza[0-9A-Za-z\\-_]{35}",
    "GCP OAuth": r"[0-9]+-[0-9A-Za-z_]{32}\\.apps\\.googleusercontent\\.com",
    "GCP Service Account": r"\"type\":\\s*\"service_account\"",
    "Firebase URL": r"https://[a-z0-9-]+\\.firebaseio\\.com",
    "Firebase API Key": r"(?i)firebase.*['\"][A-Za-z0-9_]{30,}['\"]",
    
    # GitHub
    "GitHub Token (Classic)": r"ghp_[0-9a-zA-Z]{36}",
    "GitHub Token (Fine-grained)": r"github_pat_[0-9a-zA-Z]{22}_[0-9a-zA-Z]{59}",
    "GitHub OAuth": r"gho_[0-9a-zA-Z]{36}",
    "GitHub App Token": r"(?:ghu|ghs)_[0-9a-zA-Z]{36}",
    "GitHub Refresh Token": r"ghr_[0-9a-zA-Z]{36}",
    
    # GitLab
    "GitLab Token": r"glpat-[0-9a-zA-Z\\-_]{20}",
    "GitLab Runner Token": r"GR1348941[0-9a-zA-Z\\-_]{20}",
    
    # Slack
    "Slack Token": r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*",
    "Slack Webhook": r"https://hooks\\.slack\\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[a-zA-Z0-9]+",
    
    # Stripe
    "Stripe Live Key": r"sk_live_[0-9a-zA-Z]{24,}",
    "Stripe Test Key": r"sk_test_[0-9a-zA-Z]{24,}",
    "Stripe Restricted Key": r"rk_live_[0-9a-zA-Z]{24,}",
    
    # Payment Processors
    "PayPal Client ID": r"(?i)paypal.*client[_-]?id.*['\"][A-Za-z0-9-]{20,}['\"]",
    "Square Access Token": r"sq0atp-[0-9A-Za-z\\-_]{22}",
    "Square OAuth Secret": r"sq0csp-[0-9A-Za-z\\-_]{43}",
    
    # Social Media & APIs
    "Twitter API Key": r"(?i)twitter.*api[_-]?key.*['\"][0-9a-zA-Z]{25}['\"]",
    "Twitter Bearer Token": r"AAAAAAAAAAAAAAAAAAAAAA[0-9A-Za-z%]+",
    "Facebook Access Token": r"EAACEdEose0cBA[0-9A-Za-z]+",
    "Facebook OAuth": r"(?i)facebook.*['\"][0-9]{13,17}['\"]",
    
    # Messaging
    "Twilio API Key": r"SK[0-9a-fA-F]{32}",
    "Twilio Account SID": r"AC[a-zA-Z0-9]{32}",
    "SendGrid API Key": r"SG\\.[a-zA-Z0-9]{22}\\.[a-zA-Z0-9\\-_]{43}",
    "Mailgun API Key": r"key-[0-9a-zA-Z]{32}",
    "Mailchimp API Key": r"[0-9a-f]{32}-us[0-9]{1,2}",
    
    # Databases
    "MongoDB Connection String": r"mongodb(?:\\+srv)?://[^\\s'\"]+",
    "PostgreSQL Connection String": r"postgres(?:ql)?://[^\\s'\"]+",
    "MySQL Connection String": r"mysql://[^\\s'\"]+",
    "Redis URL": r"redis://[^\\s'\"]+",
    
    # CI/CD & DevOps
    "Heroku API Key": r"[h|H][e|E][r|R][o|O][k|K][u|U].*[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}",
    "Travis CI Token": r"(?i)travis.*['\"][a-zA-Z0-9]{20,}['\"]",
    "CircleCI Token": r"(?i)circle.*token.*['\"][a-f0-9]{40}['\"]",
    "NPM Token": r"(?i)//registry\\.npmjs\\.org/:_authToken=[0-9a-f-]{36}",
    "PyPI Token": r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9-_]{50,}",
    "Docker Hub Token": r"dckr_pat_[A-Za-z0-9_-]{27}",
    
    # Cryptographic Keys
    "RSA Private Key": r"-----BEGIN RSA PRIVATE KEY-----",
    "DSA Private Key": r"-----BEGIN DSA PRIVATE KEY-----",
    "EC Private Key": r"-----BEGIN EC PRIVATE KEY-----",
    "OpenSSH Private Key": r"-----BEGIN OPENSSH PRIVATE KEY-----",
    "PGP Private Key": r"-----BEGIN PGP PRIVATE KEY BLOCK-----",
    "Generic Private Key": r"-----BEGIN PRIVATE KEY-----",
    
    # JWT & Auth
    "JWT Token": r"eyJ[A-Za-z0-9-_]+\\.eyJ[A-Za-z0-9-_]+\\.[A-Za-z0-9-_]+",
    "Basic Auth Header": r"(?i)authorization:\\s*basic\\s+[a-zA-Z0-9+/=]+",
    "Bearer Token": r"(?i)bearer\\s+[a-zA-Z0-9\\-_\\.]+",
    
    # Generic Patterns
    "Generic API Key": r"(?i)(api[_-]?key|apikey|api_secret)[\"']?\\s*[:=]\\s*[\"']?[a-zA-Z0-9_\\-]{16,}[\"']?",
    "Generic Secret": r"(?i)(secret|password|passwd|pwd)[\"']?\\s*[:=]\\s*[\"'][^\"']{8,}[\"']",
    "Generic Token": r"(?i)(access[_-]?token|auth[_-]?token)[\"']?\\s*[:=]\\s*[\"']?[a-zA-Z0-9_\\-]{16,}[\"']?",
    "Hardcoded Password": r"(?i)(password|passwd|pwd)\\s*=\\s*[\"'][^\"']{4,}[\"']",
    
    # Cloud & Infrastructure
    "DigitalOcean Token": r"dop_v1_[a-f0-9]{64}",
    "DigitalOcean OAuth": r"doo_v1_[a-f0-9]{64}",
    "Cloudflare API Key": r"(?i)cloudflare.*['\"][a-z0-9]{37}['\"]",
    "Shopify Token": r"shpat_[a-fA-F0-9]{32}",
    "Shopify Shared Secret": r"shpss_[a-fA-F0-9]{32}",
    
    # Misc
    "Telegram Bot Token": r"[0-9]+:AA[0-9A-Za-z\\-_]{33}",
    "Discord Bot Token": r"[MN][A-Za-z\\d]{23,}\\.[\w-]{6}\\.[\w-]{27}",
    "Discord Webhook": r"https://discord(?:app)?\\.com/api/webhooks/[0-9]+/[A-Za-z0-9_-]+",
    "IP Address (Private)": r"(?:^|[^0-9])(10\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}|172\\.(?:1[6-9]|2[0-9]|3[01])\\.[0-9]{1,3}\\.[0-9]{1,3}|192\\.168\\.[0-9]{1,3}\\.[0-9]{1,3})(?:[^0-9]|$)",
}

# Sensitive filenames to flag
SENSITIVE_FILENAMES = {
    # Credentials & Keys
    ".env", ".env.local", ".env.production", ".env.staging", ".env.development",
    ".env.backup", ".env.old", ".env.example", "credentials", "credentials.json",
    "id_rsa", "id_rsa.pub", "id_dsa", "id_ecdsa", "id_ed25519",
    ".pem", ".key", ".p12", ".pfx", ".asc",
    
    # Config files
    "config.json", "config.yaml", "config.yml", "secrets.json", "secrets.yaml",
    "settings.json", "settings.yaml", "application.properties", "application.yml",
    ".htpasswd", ".netrc", ".npmrc", ".pypirc", ".dockercfg",
    "docker-compose.override.yml", "wp-config.php", "database.yml",
    
    # Cloud configs
    "terraform.tfvars", "terraform.tfstate", "*.auto.tfvars",
    "ansible-vault", "vault.yml", "secrets.enc",
    
    # History & Backups
    ".bash_history", ".zsh_history", ".mysql_history", ".psql_history",
    "backup.sql", "dump.sql", "database.sql",
    
    # AWS
    ".aws/credentials", "aws_credentials", ".s3cfg",
    
    # GCP
    "service-account.json", "gcp-credentials.json",
    
    # Kubernetes
    "kubeconfig", ".kube/config",
}

# File extensions to skip (binary/large files)
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".min.js", ".min.css",  # Minified files
    ".map",  # Source maps
    ".lock",  # Lock files
}

# =============================================================================
# ENTROPY DETECTION - Find high-entropy strings (potential secrets)
# =============================================================================

def calculate_shannon_entropy(data: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not data:
        return 0.0
    
    entropy = 0.0
    for x in set(data):
        p_x = data.count(x) / len(data)
        entropy -= p_x * math.log2(p_x)
    return entropy

def find_high_entropy_strings(content: str, threshold: float = 4.5) -> List[Dict]:
    """Find high-entropy strings that might be secrets."""
    findings = []
    
    # Look for quoted strings and assignments
    patterns = [
        r'["\']([A-Za-z0-9+/=_-]{20,})["\']',  # Quoted strings
        r'=\s*([A-Za-z0-9+/=_-]{20,})',  # Assignments
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, content):
            candidate = match.group(1)
            entropy = calculate_shannon_entropy(candidate)
            
            if entropy >= threshold and len(candidate) >= 20:
                # Skip if it looks like a common word or path
                if not re.match(r'^[a-z]+$', candidate, re.I):
                    findings.append({
                        "type": "High Entropy String",
                        "value": candidate[:50] + "..." if len(candidate) > 50 else candidate,
                        "entropy": round(entropy, 2),
                        "length": len(candidate)
                    })
    
    return findings

# =============================================================================
# GITHUB SECRET HUNTER CLASS
# =============================================================================

class GitHubSecretHunter:
    """Advanced GitHub secret scanning tool."""
    
    def __init__(self, token: str, target: str):
        self.token = token
        self.target = target
        self.auth = Auth.Token(token)
        self.github = Github(auth=self.auth)
        
        self.findings: List[Dict] = []
        self.scanned_repos: Set[str] = set()
        self.stats = {
            "repos_scanned": 0,
            "files_scanned": 0,
            "commits_scanned": 0,
            "gists_scanned": 0,
            "secrets_found": 0,
            "sensitive_files": 0,
            "high_entropy": 0,
        }
        
        # Rate limit tracking
        self.rate_limit_hits = 0
        
        # Initialize output file for incremental saving
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Create output filename
        self.scan_start_time = datetime.now()
        self.output_file = self.output_dir / f"github_secrets_{target}.json"
        
        # Initialize the JSON file immediately
        self._init_output_file()
        
    def _init_output_file(self):
        """Initialize the JSON output file at scan start."""
        if not GITHUB_OUTPUT_JSON:
            return
            
        initial_data = {
            "target": self.target,
            "scan_start_time": self.scan_start_time.isoformat(),
            "scan_end_time": None,
            "status": "in_progress",
            "statistics": self.stats,
            "findings": []
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
            
        print(f"[*] Output file initialized: {self.output_file}")
        
    def _save_incremental(self):
        """Save current state to JSON file (called after each finding)."""
        if not GITHUB_OUTPUT_JSON:
            return
            
        data = {
            "target": self.target,
            "scan_start_time": self.scan_start_time.isoformat(),
            "scan_end_time": None,
            "status": "in_progress",
            "last_update": datetime.now().isoformat(),
            "statistics": self.stats,
            "findings": self.findings
        }
        
        # Write to temp file first, then rename (atomic operation)
        temp_file = self.output_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.output_file)
        except Exception as e:
            print(f"    [!] Error saving incremental: {e}")
            # Fallback: write directly
            try:
                with open(self.output_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except:
                pass
            
    def _handle_rate_limit(self):
        """Handle GitHub rate limit with exponential backoff."""
        self.rate_limit_hits += 1
        # Save before waiting (in case user cancels)
        self._save_incremental()
        
        rate_limit = self.github.get_rate_limit()
        reset_time = rate_limit.core.reset
        wait_seconds = (reset_time - datetime.utcnow()).total_seconds() + 10
        
        if wait_seconds > 0:
            print(f"\n[!] Rate limit hit! Waiting {int(wait_seconds)} seconds...")
            print(f"    Reset time: {reset_time}")
            time.sleep(min(wait_seconds, 300))  # Max 5 min wait
        else:
            time.sleep(60)  # Default wait
            
    def _should_skip_file(self, filename: str) -> bool:
        """Check if file should be skipped based on extension."""
        ext = os.path.splitext(filename)[1].lower()
        return ext in SKIP_EXTENSIONS
        
    def _is_sensitive_filename(self, filepath: str) -> bool:
        """Check if filename is in sensitive list."""
        filename = os.path.basename(filepath).lower()
        return filename in SENSITIVE_FILENAMES or any(
            sens.lower() in filepath.lower() for sens in SENSITIVE_FILENAMES
        )
        
    def _add_finding(self, finding_type: str, repo: str, path: str, 
                     secret_type: str, details: Optional[Dict] = None):
        """Add a finding to the results and save incrementally."""
        finding = {
            "timestamp": datetime.now().isoformat(),
            "type": finding_type,
            "repository": repo,
            "path": path,
            "secret_type": secret_type,
            "details": details or {}
        }
        self.findings.append(finding)
        
        # Color-coded output
        if finding_type == "SECRET":
            print(f"\033[91m[!!!] SECRET FOUND: {secret_type}\033[0m")
            self.stats["secrets_found"] += 1
        elif finding_type == "SENSITIVE_FILE":
            print(f"\033[93m[!] SENSITIVE FILE: {path}\033[0m")
            self.stats["sensitive_files"] += 1
        elif finding_type == "HIGH_ENTROPY":
            print(f"\033[95m[~] HIGH ENTROPY: {secret_type}\033[0m")
            self.stats["high_entropy"] += 1
            
        print(f"    Repository: {repo}")
        print(f"    Path: {path}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")
        print()
        
        # Save incrementally after each finding
        self._save_incremental()
        
    def scan_file_content(self, repo_name: str, content: str, path: str):
        """Scan file content for secrets using regex patterns."""
        # Pattern matching
        for secret_type, pattern in SECRET_PATTERNS.items():
            try:
                matches = re.findall(pattern, content)
                if matches:
                    self._add_finding(
                        "SECRET", repo_name, path, secret_type,
                        {"matches": len(matches), "sample": str(matches[0])[:100]}
                    )
            except re.error:
                continue
                
        # Entropy-based detection
        high_entropy = find_high_entropy_strings(content)
        for finding in high_entropy[:5]:  # Limit to top 5 per file
            self._add_finding(
                "HIGH_ENTROPY", repo_name, path,
                f"High Entropy ({finding['entropy']})",
                finding
            )
            
    def scan_repo_contents(self, repo, path: str = ""):
        """Recursively scan repository contents."""
        try:
            contents = repo.get_contents(path)
            if not isinstance(contents, list):
                contents = [contents]
                
            for item in contents:
                if item.type == "dir":
                    self.scan_repo_contents(repo, item.path)
                else:
                    # Check sensitive filename
                    if self._is_sensitive_filename(item.path):
                        self._add_finding(
                            "SENSITIVE_FILE", repo.full_name, item.path,
                            "Sensitive Filename"
                        )
                    
                    # Skip binary/large files
                    if self._should_skip_file(item.name):
                        continue
                        
                    # Scan file content
                    try:
                        if item.size < 500000:  # Skip files > 500KB
                            decoded = item.decoded_content.decode('utf-8', errors='ignore')
                            self.scan_file_content(repo.full_name, decoded, item.path)
                            self.stats["files_scanned"] += 1
                            # Save every 50 files to track progress
                            if self.stats["files_scanned"] % 50 == 0:
                                self._save_incremental()
                    except Exception:
                        continue
                        
        except RateLimitExceededException:
            self._handle_rate_limit()
            self.scan_repo_contents(repo, path)
        except GithubException as e:
            if e.status != 404:  # Ignore not found (empty repos)
                print(f"    [!] Error accessing {path}: {e}")
                
    def scan_commit_history(self, repo):
        """Scan commit history for leaked secrets."""
        if not GITHUB_SCAN_COMMITS:
            return
            
        try:
            commits = repo.get_commits()
            count = 0
            max_commits = GITHUB_MAX_COMMITS if GITHUB_MAX_COMMITS > 0 else float('inf')
            
            for commit in commits:
                if count >= max_commits:
                    break
                    
                try:
                    for file in commit.files:
                        if file.patch and not self._should_skip_file(file.filename):
                            self.scan_file_content(
                                repo.full_name, 
                                file.patch, 
                                f"{file.filename} (commit: {commit.sha[:7]})"
                            )
                    self.stats["commits_scanned"] += 1
                    count += 1
                    # Save every 20 commits to track progress
                    if count % 20 == 0:
                        self._save_incremental()
                except Exception:
                    continue
                    
        except RateLimitExceededException:
            self._handle_rate_limit()
        except Exception as e:
            print(f"    [!] Error scanning commits: {e}")
                
    def scan_repo(self, repo):
        """Scan a single repository."""
        if repo.full_name in self.scanned_repos:
            return
            
        self.scanned_repos.add(repo.full_name)
        print(f"\n[*] Scanning repository: {repo.full_name}")
        print(f"    Stars: {repo.stargazers_count} | Forks: {repo.forks_count}")
        
        # 1. Recursive content scan (thorough)
        self.scan_repo_contents(repo)
        
        # 3. Commit history scan (optional, slow)
        if GITHUB_SCAN_COMMITS:
            self.scan_commit_history(repo)
            
        self.stats["repos_scanned"] += 1
        
        # Save after each repo is completed
        self._save_incremental()
        print(f"    [✓] Repo scan complete. Progress saved.")
        
    def scan_gists(self, user):
        """Scan user gists for secrets."""
        if not GITHUB_SCAN_GISTS:
            return
            
        try:
            for gist in user.get_gists():
                print(f"    [*] Scanning gist: {gist.id}")
                for filename, file in gist.files.items():
                    if not self._should_skip_file(filename):
                        try:
                            content = file.content
                            self.scan_file_content(
                                f"gist:{user.login}",
                                content,
                                f"{gist.id}/{filename}"
                            )
                            self.stats["gists_scanned"] += 1
                        except Exception:
                            continue
                            
        except RateLimitExceededException:
            self._handle_rate_limit()
        except Exception as e:
            print(f"    [!] Error scanning gists: {e}")
            
    def scan_organization(self):
        """Scan an organization and its members."""
        try:
            org = self.github.get_organization(self.target)
            print(f"\n[*] Organization found: {org.login}")
            print(f"    Public repos: {org.public_repos}")
            print(f"    Members: {org.get_members().totalCount if org.get_members() else 'N/A'}")
            
            # Scan organization repos
            for repo in org.get_repos():
                self.scan_repo(repo)
                
            # Scan member repos and gists
            if GITHUB_SCAN_MEMBERS:
                print("\n[*] Scanning organization members...")
                for member in org.get_members():
                    print(f"\n[*] Member: {member.login}")
                    
                    for repo in member.get_repos():
                        self.scan_repo(repo)
                        
                    if GITHUB_SCAN_GISTS:
                        self.scan_gists(member)
                        
        except GithubException as e:
            if e.status == 404:
                # Try as a user instead
                self.scan_user()
            else:
                raise
                
    def scan_user(self):
        """Scan a user's repositories and gists."""
        try:
            user = self.github.get_user(self.target)
            print(f"\n[*] User found: {user.login}")
            print(f"    Public repos: {user.public_repos}")
            
            for repo in user.get_repos():
                self.scan_repo(repo)
                
            if GITHUB_SCAN_GISTS:
                print("\n[*] Scanning user gists...")
                self.scan_gists(user)
                
        except GithubException as e:
            print(f"[!] Error: {e}")
            
    def save_results(self, status: str = "completed"):
        """Save final results to JSON file."""
        if not GITHUB_OUTPUT_JSON:
            return
            
        scan_end_time = datetime.now()
        duration = (scan_end_time - self.scan_start_time).total_seconds()
        
        results = {
            "target": self.target,
            "scan_start_time": self.scan_start_time.isoformat(),
            "scan_end_time": scan_end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "status": status,
            "last_update": scan_end_time.isoformat(),
            "statistics": self.stats,
            "findings": self.findings
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n[*] Final results saved to: {self.output_file}")
        
    def print_summary(self):
        """Print scan summary."""
        print("\n" + "=" * 70)
        print("                         SCAN SUMMARY")
        print("=" * 70)
        print(f"  Target:              {self.target}")
        print(f"  Repos Scanned:       {self.stats['repos_scanned']}")
        print(f"  Files Scanned:       {self.stats['files_scanned']}")
        print(f"  Commits Scanned:     {self.stats['commits_scanned']}")
        print(f"  Gists Scanned:       {self.stats['gists_scanned']}")
        print("-" * 70)
        print(f"\033[91m  Secrets Found:       {self.stats['secrets_found']}\033[0m")
        print(f"\033[93m  Sensitive Files:     {self.stats['sensitive_files']}\033[0m")
        print(f"\033[95m  High Entropy:        {self.stats['high_entropy']}\033[0m")
        print(f"  Rate Limit Hits:     {self.rate_limit_hits}")
        print("=" * 70)
        
        if self.stats['secrets_found'] > 0:
            print("\n\033[91m[!!!] CRITICAL: Secrets were found! Review findings immediately.\033[0m")
            
    def run(self):
        """Run the complete scan."""
        status = "completed"
        
        try:
            # Check if target is org or user
            self.scan_organization()
        except RateLimitExceededException:
            self._handle_rate_limit()
            self.run()
            return self.findings
        except KeyboardInterrupt:
            print("\n\n[!] Scan interrupted by user.")
            status = "interrupted"
        except Exception as e:
            print(f"[!] Error during scan: {e}")
            status = "error"
            
        self.print_summary()
        self.save_results(status)
        
        return self.findings

