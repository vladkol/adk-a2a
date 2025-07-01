# ADK to A2A Relay

This example is a lightweight relay server that exposes [Agent Development Kit (ADK)](https://google.github.io/adk-docs/) agents as
[Agent2Agent (A2A)](https://a2aproject.github.io/A2A) compliant services.

This library acts as a bridge, allowing any ADK agent to be discovered and communicate within an A2A ecosystem using the standard A2A protocol.

## Features

- Seamless Protocol Bridging: Translates A2A requests into ADK queries and streams ADK responses back as A2A events.
- Dynamic Agent Loading: Run any ADK agent without code modification by simply pointing the CLI to its directory.
- Full Streaming Support: Natively handles the streaming capabilities of both ADK and A2A for real-time, interactive experiences.
- Automatic Type Conversion: Intelligently converts message parts (text, files, data payloads) between A2A and Google GenAI formats.
- Function Call Forwarding: Optionally relays ADK function calls and responses as structured A2A artifacts for advanced tool use cases.
- Artifact Handling: Manages ADK artifacts, making them available within the A2A task context.
- Automatic Agent Card: Generates a standard A2A AgentCard from your ADK agent's metadata, or uses a custom agent.json file if provided.
- Simple CLI Interface: Easy to run from the command line with configurable host, port, and more flags.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)

## Installation

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -U -r requirements.txt
```

## Usage

1. Copy `.env-sample` file to the same directory as `.env`.
2. Specify Id of your Google Cloud Project in `GOOGLE_CLOUD_PROJECT` variable.
The project must have Vertex AI APIs enabled.

To run the sample agent:

```bash
python3 -m adk_a2a samples/adk_agents/dice_roll
```

To run your own agent, replace the last argument with your ADK agent's directory.

### CLI Options

`python3 -m adk_a2a [--debug] [--host HOST_NAME] [--port PORT_NUMBER] [--transfer-function-calls] AGENT_DIRECTORY`

You can customize the server's behavior with the following arguments:

| Argument                    | Description                                                              | Default     |
| --------------------------- | ------------------------------------------------------------------------ | ----------- |
| `AGENT_DIRECTORY`           | (Required) Path to the directory containing the ADK agent.               | N/A      |
| `--host`                    | Host address to bind the server to.                                      | `0.0.0.0`   |
| `--port`                    | Port for the HTTP server.                                                | `8080`      |
| `--transfer-function-calls` | If set, relays ADK function calls/responses as A2A artifacts.            | `False`     |
| `--debug`                   | Enables debug-level logging for detailed output.                         | `False`     |

## Interact with the Agent

Use any A2A-compatible client, such as [A2A Inspector](https://github.com/a2aproject/a2a-inspector).

1. Run the agent server (see above).
2. Connect to `http://127.0.0.1:8080` (if testing locally) or another address you exposed the server with.
3. Make a query to the agent from the client.

## Known limitations

The relay currently doesn't support the following features:

- ADK Long-running tools
- ADK in-task authentication
- Agent-to-agent authentication must be done at the service level when making a request from the client.

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Code and data from this repository are intended for demonstration purposes only. It is not intended for use in a production environment.
