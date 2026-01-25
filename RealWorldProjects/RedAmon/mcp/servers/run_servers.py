#!/usr/bin/env python3
"""
MCP Server Runner - Launches all MCP servers for RedAmon Agentic AI

This script starts all MCP servers (naabu, nuclei, curl, metasploit) either
in stdio mode (for direct integration) or SSE mode (for network access).

Usage:
    python run_servers.py              # Run in SSE mode (default for container)
    python run_servers.py --stdio      # Run single server in stdio mode
    python run_servers.py --server naabu --stdio  # Run specific server
"""

import asyncio
import os
import sys
import signal
import logging
from multiprocessing import Process
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp-runner")

# Server configurations
SERVERS = {
    "naabu": {
        "module": "naabu_server",
        "port": 8000,
        "description": "Port Scanner"
    },
    "curl": {
        "module": "curl_server",
        "port": 8001,
        "description": "HTTP Client"
    },
    "nuclei": {
        "module": "nuclei_server",
        "port": 8002,
        "description": "Vulnerability Scanner"
    },
    "metasploit": {
        "module": "metasploit_server",
        "port": 8003,
        "description": "Exploitation Framework"
    }
}


def run_server(name: str, config: dict, transport: str = "sse"):
    """Run a single MCP server."""
    import importlib

    logger.info(f"Starting {name} server ({config['description']}) on port {config['port']}")

    # Set environment variables for the server
    os.environ["MCP_TRANSPORT"] = transport
    os.environ[f"{name.upper()}_PORT"] = str(config["port"])

    try:
        # Import and run the server module
        module = importlib.import_module(config["module"])

        if transport == "sse":
            module.mcp.run(
                transport="sse",
                host="0.0.0.0",
                port=config["port"]
            )
        else:
            module.mcp.run(transport="stdio")

    except Exception as e:
        logger.error(f"Error starting {name} server: {e}")
        raise


def run_all_servers_sse():
    """Run all servers in SSE mode using multiprocessing."""
    processes: List[Process] = []

    def shutdown(signum, frame):
        logger.info("Shutting down all servers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start each server in a separate process
    for name, config in SERVERS.items():
        p = Process(
            target=run_server,
            args=(name, config, "sse"),
            name=f"mcp-{name}"
        )
        p.start()
        processes.append(p)
        logger.info(f"Started {name} server (PID: {p.pid})")

    logger.info("All MCP servers started successfully")
    logger.info("Servers available at:")
    for name, config in SERVERS.items():
        logger.info(f"  - {name}: http://0.0.0.0:{config['port']}")

    # Wait for all processes
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        shutdown(None, None)


def run_single_server_stdio(server_name: str):
    """Run a single server in stdio mode."""
    if server_name not in SERVERS:
        logger.error(f"Unknown server: {server_name}")
        logger.info(f"Available servers: {', '.join(SERVERS.keys())}")
        sys.exit(1)

    config = SERVERS[server_name]
    run_server(server_name, config, "stdio")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RedAmon MCP Server Runner"
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run in stdio mode (for direct MCP integration)"
    )
    parser.add_argument(
        "--server",
        choices=list(SERVERS.keys()),
        help="Specific server to run (required for stdio mode)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available servers"
    )

    args = parser.parse_args()

    if args.list:
        print("Available MCP Servers:")
        for name, config in SERVERS.items():
            print(f"  - {name}: {config['description']} (port {config['port']})")
        sys.exit(0)

    if args.stdio:
        if not args.server:
            logger.error("--server is required when using --stdio mode")
            sys.exit(1)
        run_single_server_stdio(args.server)
    else:
        # Default: run all servers in SSE mode
        run_all_servers_sse()


if __name__ == "__main__":
    main()
