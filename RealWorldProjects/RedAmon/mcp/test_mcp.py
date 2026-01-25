#!/usr/bin/env python3
"""
MCP Server Test Client

Tests the MCP servers running in the Docker container.
Uses docker exec to call tools directly (no external dependencies).
"""

import subprocess
import sys
import shlex


def run_docker_exec(command: str, timeout: int = 60, use_shell: bool = False) -> str:
    """
    Run a command inside the redamon-kali container.

    Args:
        command: Command to execute
        timeout: Timeout in seconds
        use_shell: If True, run command through shell (for complex commands with quotes)

    Returns:
        Command output
    """
    try:
        if use_shell:
            # For complex commands with quotes, use shell=True via /bin/sh -c
            full_cmd = ["docker", "exec", "redamon-kali", "/bin/sh", "-c", command]
        else:
            # For simple commands, split properly using shlex
            full_cmd = ["docker", "exec", "redamon-kali"] + shlex.split(command)

        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]: {result.stderr}"
        return output
    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out after {timeout} seconds"
    except Exception as e:
        return f"[ERROR] {e}"


def test_curl(target: str):
    """Test curl HTTP request."""
    print(f"\n{'='*60}")
    print("Testing CURL (HTTP Client)")
    print(f"{'='*60}")
    print(f"Target: {target}")

    # Run curl inside container
    output = run_docker_exec(f"curl -s -i -L --connect-timeout 10 {target}")
    print("\nResponse:")
    print(output[:2000] if len(output) > 2000 else output)


def test_naabu(target: str):
    """Test naabu port scanner."""
    print(f"\n{'='*60}")
    print("Testing NAABU (Port Scanner)")
    print(f"{'='*60}")
    print(f"Target: {target}")

    # Extract hostname from URL
    host = target.replace("https://", "").replace("http://", "").split("/")[0]

    output = run_docker_exec(f"naabu -host {host} -top-ports 100 -silent", timeout=120)
    print("\nOpen Ports:")
    print(output if output.strip() else "[No ports found or scan in progress]")


def test_nuclei(target: str):
    """Test nuclei vulnerability scanner."""
    print(f"\n{'='*60}")
    print("Testing NUCLEI (Vulnerability Scanner)")
    print(f"{'='*60}")
    print(f"Target: {target}")

    output = run_docker_exec(f"nuclei -u {target} -t http/technologies/ -silent", timeout=180)
    print("\nFindings:")
    print(output if output.strip() else "[No findings or scan in progress]")


def test_metasploit(target: str):
    """Test Metasploit Framework."""
    print(f"\n{'='*60}")
    print("Testing METASPLOIT (Exploitation Framework)")
    print(f"{'='*60}")
    print(f"Target: {target}")

    # Extract hostname from URL
    host = target.replace("https://", "").replace("http://", "").split("/")[0]

    print("\n[1] Checking Metasploit version...")
    output = run_docker_exec("msfconsole -v", timeout=30)
    print(output.strip() if output.strip() else "[No output]")

    print("\n[2] Searching for HTTP-related modules...")
    # Use msfconsole with -q (quiet) and -x (execute command)
    # use_shell=True to handle the complex quoting
    output = run_docker_exec(
        'msfconsole -q -x "search type:auxiliary scanner http; exit"',
        timeout=90,
        use_shell=True
    )
    # Show first 30 lines to avoid too much output
    lines = output.split('\n')[:30]
    print('\n'.join(lines))
    if len(output.split('\n')) > 30:
        print(f"... [{len(output.split(chr(10))) - 30} more lines]")

    print(f"\n[3] Getting info on http_version scanner...")
    output = run_docker_exec(
        'msfconsole -q -x "info auxiliary/scanner/http/http_version; exit"',
        timeout=90,
        use_shell=True
    )
    print(output[:1500] if len(output) > 1500 else output)

    print(f"\n[4] Running HTTP version scan on {host}...")
    output = run_docker_exec(
        f'msfconsole -q -x "use auxiliary/scanner/http/http_version; set RHOSTS {host}; run; exit"',
        timeout=120,
        use_shell=True
    )
    print(output if output.strip() else "[No output - scan may still be running]")


def check_container_running() -> bool:
    """Check if the redamon-kali container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", "redamon-kali"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == "true"
    except Exception:
        return False


def simple_http_test(target: str):
    """
    Simple direct HTTP test using system curl.
    Tests if we can reach the target from the host machine.
    """
    print(f"\n{'='*60}")
    print("Simple HTTP Test (from host machine)")
    print(f"{'='*60}")
    print(f"Target: {target}")

    try:
        result = subprocess.run(
            ["curl", "-s", "-I", "-L", "--connect-timeout", "10", target],
            capture_output=True,
            text=True,
            timeout=30
        )
        print("\nResponse Headers:")
        print(result.stdout if result.stdout else "[No response]")
        if result.stderr:
            print(f"[STDERR]: {result.stderr}")
    except FileNotFoundError:
        print("\n[ERROR] curl not found on host. Install curl or skip this test.")
    except Exception as e:
        print(f"\n[ERROR] {e}")


def main():
    """Main test function."""
    target = sys.argv[1] if len(sys.argv) > 1 else "https://www.devergolabs.com"

    print("\n" + "="*60)
    print("MCP Server Test Suite")
    print("="*60)
    print(f"Target: {target}")

    # Check if container is running
    if not check_container_running():
        print("\n[ERROR] Container 'redamon-kali' is not running!")
        print("Start it with: docker-compose up -d")
        sys.exit(1)

    print("\nContainer 'redamon-kali' is running.")

    # Menu
    print("\nAvailable tests:")
    print("  1. Simple HTTP test (from host)")
    print("  2. Curl test (from container)")
    print("  3. Naabu port scan")
    print("  4. Nuclei vulnerability scan")
    print("  5. Metasploit test")
    print("  6. Run all tests")
    print("  0. Exit")

    choice = input("\nSelect test [1-6, 0 to exit]: ").strip()

    if choice == "1":
        simple_http_test(target)
    elif choice == "2":
        test_curl(target)
    elif choice == "3":
        test_naabu(target)
    elif choice == "4":
        test_nuclei(target)
    elif choice == "5":
        test_metasploit(target)
    elif choice == "6":
        simple_http_test(target)
        test_curl(target)
        test_naabu(target)
        test_nuclei(target)
        test_metasploit(target)
    elif choice == "0":
        print("Exiting.")
    else:
        print(f"Invalid choice: {choice}")

    # Show quick commands
    host = target.replace("https://", "").replace("http://", "").split("/")[0]
    print("\n" + "="*60)
    print("Quick Test Commands (run in terminal)")
    print("="*60)
    print(f"""
# Test curl directly in container:
docker exec redamon-kali curl -s -i {target}

# Test naabu port scan:
docker exec redamon-kali naabu -host {host} -top-ports 100

# Test nuclei vulnerability scan:
docker exec redamon-kali nuclei -u {target} -t http/technologies/

# Test metasploit - search modules:
docker exec redamon-kali msfconsole -q -x 'search type:auxiliary http; exit'

# Test metasploit - HTTP version scan:
docker exec redamon-kali msfconsole -q -x 'use auxiliary/scanner/http/http_version; set RHOSTS {host}; run; exit'

# Test metasploit - interactive:
docker exec -it redamon-kali msfconsole

# Interactive shell:
docker exec -it redamon-kali /bin/bash
""")


if __name__ == "__main__":
    main()
