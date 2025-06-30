# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import importlib.util
import inspect
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from google.adk.agents import BaseAgent

sys.path.append(str(Path(__file__).parent))
from run_a2a import adk_as_a2a # all-in-one relay

import uvicorn

ROOT_AGENT_FILES = [
    "agent.py",
    "__init__.py",
    "main.py"
    "root_agent.py",
]

def _get_root_agent(agent_directory: str) -> BaseAgent:
    """Returns the root agent for the given directory."""
    agent_path = Path(agent_directory)
    if not agent_path.exists() or not agent_path.is_dir():
        raise ValueError(f'Agent directory "{agent_directory}" does not exist.')
    if Path(".env").exists():
        load_dotenv()
    else:
        dotenv_path = Path(".env").relative_to(Path(__file__).parent)
        if dotenv_path.exists():
            load_dotenv(dotenv_path)

    if agent_path.is_absolute():
        agent_path = agent_path.relative_to(os.getcwd())

    file_name = ""
    for f in ROOT_AGENT_FILES:
        if (agent_path / f).exists():
            file_name = f
            break
    if not file_name:
        raise FileNotFoundError(f"Directory {agent_directory} "
                                "doesn't have any of these files: "
                                f"{ROOT_AGENT_FILES}")
    spec = importlib.util.spec_from_file_location(
        "agent_module",
        str(agent_path / file_name)
    )
    agent_module = importlib.util.module_from_spec(spec) # type: ignore
    spec.loader.exec_module(agent_module) # type: ignore
    if getattr(agent_module, "root_agent"):
        root_agent = agent_module.root_agent
    else:
        raise ValueError(f'Unable to find "root_agent".')
    return root_agent

async def _wait(awaitable):
    result, _ = await awaitable
    return result

def _cli_main():
    parser = argparse.ArgumentParser(
        description="A2A Relay"
    )
    parser.add_argument(
        "--debug",
        help="Debugging mode",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--host",
        help="Host address to bind to",
        type=str,
        required=False,
        default="0.0.0.0"
    )
    parser.add_argument(
        "--port",
        help="HTTP Server Port",
        type=int,
        required=False,
        default=8080
    )
    parser.add_argument(
        "--transfer-function-calls",
        help="Makes the relay transfer function calls and responses from ADK.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "agent_directory",
        nargs=1,
        help="Agent Directory",
    )
    options = parser.parse_args(sys.argv[1:])

    agent_directory = options.agent_directory[0]
    agent_directory = str(Path(agent_directory).absolute())
    root_agent = _get_root_agent(agent_directory)
    if inspect.isawaitable(root_agent):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        root_agent = loop.run_until_complete(_wait(root_agent))

    host = options.host
    if (
        host.lower() == "localhost"
        or host.startswith("0.0.0.")
        or host.startswith("127.0.0.")
    ):
        schema = "http"
    else:
        schema = "https"
    agent_url = f"{schema}://{host}:{options.port}/"

    uvicorn.run(
        adk_as_a2a(
            # Same arguments as in AdkApp class:
            # https://cloud.google.com/python/docs/reference/vertexai/latest/vertexai.preview.reasoning_engines.AdkApp
            # + `a2a_agent_version` and `agent_url`

            adk_agent=root_agent,
            enable_tracing=True,
            # session_service_builder = None,
            # artifact_service_builder = None,
            # env_vars = None,
            # a2a_agent_version = None,
            agent_url=agent_url,
            transfer_function_calls=options.transfer_function_calls,
            agent_directory=agent_directory
        ).build(),
        host=options.host,
        port=options.port,
        log_level="debug" if options.debug else "info",
    )

################################################################################
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent))
    load_dotenv()
    _cli_main()