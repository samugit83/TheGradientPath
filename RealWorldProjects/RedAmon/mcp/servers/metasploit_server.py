"""
Metasploit MCP Server - Stateful Exploitation Framework

Exposes Metasploit Framework as a single MCP tool with PERSISTENT state.
Uses a persistent msfconsole process that maintains state between calls.

Architecture:
    - Single persistent msfconsole process per server instance
    - Module context persists between calls
    - Meterpreter/shell sessions persist until explicitly closed
    - Timing-based output detection (universal, no regex parsing)

Tools:
    - metasploit_console: Execute any msfconsole command (stateful)
"""

from fastmcp import FastMCP
import subprocess
import threading
import queue
import time
import os
import re
import atexit
from typing import Optional, Set

# Server configuration
SERVER_NAME = "metasploit"
SERVER_HOST = os.getenv("MCP_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("METASPLOIT_PORT", "8003"))
DEBUG = os.getenv("MSF_DEBUG", "false").lower() == "true"

mcp = FastMCP(SERVER_NAME)


class PersistentMsfConsole:
    """
    Manages a persistent msfconsole process with bidirectional I/O.

    Uses timing-based output detection - waits for output to settle
    rather than parsing specific prompts. This is universal and works
    with any msfconsole output format.
    """

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.output_queue: queue.Queue = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.session_ids: Set[int] = set()
        self._initialized = False

    def start(self) -> bool:
        """Start the persistent msfconsole process."""
        if self.process and self.process.poll() is None:
            return True  # Already running

        try:
            print("[MSF] Starting msfconsole process...")
            self.process = subprocess.Popen(
                ["msfconsole", "-q", "-x", ""],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            print(f"[MSF] Process started with PID: {self.process.pid}")

            # Start background thread to read output
            self.reader_thread = threading.Thread(
                target=self._read_output,
                daemon=True
            )
            self.reader_thread.start()

            # Wait for msfconsole to be ready (can take 60-120s on first start)
            self._wait_for_output(timeout=120, quiet_period=5.0)
            self._initialized = True
            print(f"[MSF] Persistent msfconsole ready (PID: {self.process.pid})")
            return True

        except Exception as e:
            print(f"[MSF] Failed to start msfconsole: {e}")
            return False

    def _read_output(self):
        """Background thread to continuously read msfconsole output."""
        if DEBUG:
            print("[MSF] Reader thread started")
        try:
            while self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self.output_queue.put(line)
                    if DEBUG:
                        print(f"[MSF] OUTPUT: {line.rstrip()[:200]}")
                    self._detect_session_events(line)
        except Exception as e:
            print(f"[MSF] Reader thread error: {e}")
        if DEBUG:
            print("[MSF] Reader thread exited")

    def _detect_session_events(self, line: str):
        """Simple session event detection - just tracks session IDs."""
        line_lower = line.lower()

        # Detect "session X opened"
        if 'session' in line_lower and 'opened' in line_lower:
            try:
                idx = line_lower.index('session')
                rest = line_lower[idx + 7:].strip()
                parts = rest.split()
                if parts and parts[0].isdigit():
                    session_id = int(parts[0])
                    self.session_ids.add(session_id)
                    print(f"[MSF] Session {session_id} opened")
            except (ValueError, IndexError):
                pass

        # Detect "session X closed"
        elif 'session' in line_lower and 'closed' in line_lower:
            try:
                idx = line_lower.index('session')
                rest = line_lower[idx + 7:].strip()
                parts = rest.split()
                if parts and parts[0].isdigit():
                    session_id = int(parts[0])
                    self.session_ids.discard(session_id)
                    print(f"[MSF] Session {session_id} closed")
            except (ValueError, IndexError):
                pass

    def _wait_for_output(self, timeout: float, quiet_period: float) -> str:
        """
        Wait for msfconsole output using timing-based detection.
        Waits until no new output arrives for 'quiet_period' seconds.
        """
        output_lines = []
        end_time = time.time() + timeout
        start_time = time.time()
        last_output_time = time.time()

        min_wait = min(3.0, timeout / 2)

        while time.time() < end_time:
            try:
                line = self.output_queue.get(timeout=0.1)
                output_lines.append(line.rstrip())
                last_output_time = time.time()

            except queue.Empty:
                elapsed = time.time() - start_time
                time_since_last = time.time() - last_output_time

                if output_lines and time_since_last >= quiet_period:
                    if DEBUG:
                        print(f"[MSF] Output complete ({quiet_period}s quiet)")
                    break

                if not output_lines and elapsed < min_wait:
                    continue

        return '\n'.join(output_lines)

    def execute(self, command: str, timeout: float = 120, quiet_period: float = 2.0) -> str:
        """Execute a command in the persistent msfconsole."""
        with self.lock:
            if not self.process or self.process.poll() is not None:
                if not self.start():
                    return "[ERROR] Failed to start msfconsole"

            # Clear any pending output
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break

            # Send command
            try:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
            except Exception as e:
                return f"[ERROR] Failed to send command: {e}"

            # Collect output
            output = self._wait_for_output(timeout=timeout, quiet_period=quiet_period)
            return output if output else "(no output)"

    def stop(self, force: bool = False):
        """Stop the msfconsole process."""
        if self.process and self.process.poll() is None:
            if force:
                print("[MSF] Force killing msfconsole process...")
                try:
                    self.process.kill()
                    self.process.wait(timeout=5)
                except:
                    pass
            else:
                try:
                    self.process.stdin.write("exit -y\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=5)
                except:
                    self.process.kill()
            print("[MSF] msfconsole stopped")
        self.process = None
        self._initialized = False
        self.session_ids.clear()


# Global singleton instance
_msf_console: Optional[PersistentMsfConsole] = None
_msf_lock = threading.Lock()


def get_msf_console() -> PersistentMsfConsole:
    """Get or create the persistent msfconsole instance."""
    global _msf_console
    with _msf_lock:
        if _msf_console is None:
            _msf_console = PersistentMsfConsole()
            _msf_console.start()
            atexit.register(_msf_console.stop)
        elif not _msf_console._initialized:
            _msf_console.start()
    return _msf_console


def _get_timing_for_command(command: str) -> tuple[float, float]:
    """Determine timeout and quiet_period based on command type."""
    cmd_lower = command.lower()

    if any(x in cmd_lower for x in ['exploit', 'run']):
        return (300, 30.0)  # Exploits: 5 min timeout, 30s quiet
    elif 'search' in cmd_lower:
        return (60, 3.0)
    elif 'sessions' in cmd_lower:
        return (60, 5.0)
    elif any(x in cmd_lower for x in ['info', 'show']):
        return (60, 3.0)
    else:
        return (120, 3.0)


def _clean_ansi_output(text: str) -> str:
    """Remove ANSI escape codes and control characters from msfconsole output."""
    # Remove ANSI escape sequences
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    text = re.sub(r'\x1b\][^\x07]*\x07', '', text)
    text = re.sub(r'\x1b[()][AB012]', '', text)

    cleaned_lines = []
    for line in text.split('\n'):
        # Handle carriage returns
        if '\r' in line:
            parts = line.split('\r')
            non_empty_parts = [p for p in parts if p.strip()]
            if non_empty_parts:
                line = non_empty_parts[-1]
            else:
                line = ''

        # Handle backspaces
        while '\x08' in line:
            pos = line.find('\x08')
            if pos > 0:
                line = line[:pos-1] + line[pos+1:]
            else:
                line = line[1:]

        # Remove control characters
        line = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', line)
        line = line.rstrip()

        if line or (cleaned_lines and cleaned_lines[-1]):
            cleaned_lines.append(line)

    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()

    # Remove garbled echo lines
    final_lines = []
    for line in cleaned_lines:
        if line.startswith('<'):
            continue
        if re.match(r'^msf\s+\S+>\S', line):
            continue
        if len(line) < 5 and not line.startswith('[') and '=>' not in line:
            continue
        final_lines.append(line)

    return '\n'.join(final_lines)


# =============================================================================
# MCP TOOL - Single tool for all Metasploit operations
# =============================================================================

@mcp.tool()
def metasploit_console(command: str) -> str:
    """
    Execute Metasploit Framework console commands with PERSISTENT state.

    This is the ONLY tool you need for all Metasploit operations.
    The msfconsole process runs continuously - state persists between calls.

    ## Context Detection (IMPORTANT for post-exploitation!)

    Check the OUTPUT to know where you are:

    | Output ends with      | You are in              | What to do                      |
    |-----------------------|-------------------------|----------------------------------|
    | `msf6 >` or `msf >`   | Main Metasploit console | Configure modules, run exploits  |
    | `meterpreter >`       | Inside Meterpreter      | Run meterpreter commands         |
    | `shell >` or `$ ` `#` | Inside system shell     | Run OS commands (whoami, ls)     |

    ## Exploitation Workflow

    1. Search: "search CVE-2021-41773"
    2. Use module: "use exploit/multi/http/apache_normalize_path_rce"
    3. Configure: "set RHOSTS x.x.x.x" (one option per call)
    4. Exploit: "exploit"
    5. Check output for session or meterpreter prompt

    ## Post-Exploitation Workflow (after session established)

    If output shows `meterpreter >`:
    - You're IN the session, run commands directly: "sysinfo", "getuid", "shell"

    If output shows `msf6 >`:
    - Enter session: "sessions -i 1"
    - Then run meterpreter commands

    To drop to OS shell from meterpreter:
    - Run: "shell"
    - Now you can run: "whoami", "id", "cat /etc/passwd"
    - Exit shell back to meterpreter: "exit"

    To background session (return to msf console):
    - Run: "background"

    ## Common Commands

    Exploitation:
    - "search <term>" - Find modules
    - "use <module>" - Load module
    - "show options" - See required options
    - "set <OPTION> <value>" - Set option
    - "exploit" - Run the exploit

    Session Management:
    - "sessions -l" - List all sessions
    - "sessions -i <id>" - Interact with session
    - "background" - Background current session
    - "sessions -k <id>" - Kill session

    Meterpreter (when in session):
    - "sysinfo" - System information
    - "getuid" - Current user
    - "shell" - Drop to OS shell
    - "download <file>" - Download file
    - "upload <file>" - Upload file

    Args:
        command: The msfconsole command to execute

    Returns:
        The output from msfconsole (check prompt to know your context)
    """
    if DEBUG:
        print(f"[MSF] Executing: {command[:100]}...")

    msf = get_msf_console()
    timeout, quiet_period = _get_timing_for_command(command)

    if DEBUG:
        print(f"[MSF] timeout={timeout}s, quiet_period={quiet_period}s")

    result = msf.execute(command, timeout=timeout, quiet_period=quiet_period)
    result = _clean_ansi_output(result)

    if DEBUG:
        print(f"[MSF] Result ({len(result)} chars)")

    return result


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport == "sse":
        mcp.run(transport="sse", host=SERVER_HOST, port=SERVER_PORT)
    else:
        mcp.run(transport="stdio")
