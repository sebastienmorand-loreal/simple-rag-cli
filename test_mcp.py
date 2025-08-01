#!/usr/bin/env python3
"""Test script for MCP server functionality."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

async def test_mcp_server():
    """Test MCP server basic functionality."""
    print("Testing MCP server...")
    
    # Start the MCP server process
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "src/simplerag.py", "mcp",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=Path(__file__).parent
    )
    
    try:
        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        # Send request
        proc.stdin.write((json.dumps(init_request) + "\n").encode())
        await proc.stdin.drain()
        
        # Read response with timeout
        try:
            response_data = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
            response = json.loads(response_data.decode())
            print(f"Initialization response: {json.dumps(response, indent=2)}")
            
            # Send list tools request
            tools_request = {
                "jsonrpc": "2.0", 
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            
            proc.stdin.write((json.dumps(tools_request) + "\n").encode())
            await proc.stdin.drain()
            
            tools_response_data = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
            tools_response = json.loads(tools_response_data.decode())
            print(f"Tools list response: {json.dumps(tools_response, indent=2)}")
            
        except asyncio.TimeoutError:
            print("Timeout waiting for server response")
            
    finally:
        # Clean up process
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

if __name__ == "__main__":
    asyncio.run(test_mcp_server())