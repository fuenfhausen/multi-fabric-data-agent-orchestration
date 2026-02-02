# Multi-Fabric Data Agent Orchestration

Orchestrate multiple Microsoft Fabric data agents using AI-powered workflows. This repository provides two implementation approaches:

| Implementation | Framework | Documentation |
|----------------|-----------|---------------|
| **Agent Framework** | Microsoft Agent Framework | [ARCHITECTURE-AGENT-FRAMEWORK.md](ARCHITECTURE-AGENT-FRAMEWORK.md) |
| **LangGraph** | LangChain / LangGraph | [ARCHITECTURE-LANGGRAPH.md](ARCHITECTURE-LANGGRAPH.md) |

## Overview

Both implementations provide:
- **Orchestrator Agent** - Routes requests to specialized Fabric agents
- **Lakehouse Agent** - Spark SQL, Delta tables, OneLake operations
- **Warehouse Agent** - T-SQL queries, semantic models, analytics
- **Real-Time Agent** - KQL queries, event streams, time-series analysis

## Project Structure

```
src/
├── shared/                     # Shared utilities (both implementations)
│   ├── __init__.py
│   ├── auth.py                 # Fabric authentication helpers
│   └── config.py               # Configuration management
│
├── agent_framework/            # Microsoft Agent Framework implementation
│   ├── __init__.py
│   ├── agents.py               # Agent definitions
│   ├── tools.py                # @ai_function tool definitions
│   ├── workflow.py             # HandoffBuilder, ConcurrentBuilder
│   └── main.py                 # Entry point and CLI
│
└── langgraph/                  # LangGraph implementation
    ├── __init__.py
    ├── state.py                # TypedDict state schemas
    ├── tools.py                # @tool definitions
    ├── nodes.py                # Graph node functions
    ├── graph.py                # StateGraph builders
    └── main.py                 # Entry point and CLI
```

## Quick Start

### Prerequisites

1. **Azure OpenAI** deployment (GPT-4o recommended)
2. **Microsoft Fabric** workspace with Lakehouse/Warehouse/Eventhouse
3. **Azure CLI** authenticated (`az login`)

### Installation

```bash
# Clone the repository
git clone https://github.com/fuenfhausen/multi-fabric-data-agent-orchestration.git
cd multi-fabric-data-agent-orchestration

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (choose one)

# Option 1: Agent Framework only
pip install -r requirements-agent-framework.txt --pre

# Option 2: LangGraph only
pip install -r requirements-langgraph.txt

# Option 3: Both implementations
pip install -r requirements.txt --pre
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_DEPLOYMENT
# - FABRIC_WORKSPACE_ID
# - etc.
```

## Usage

### Microsoft Agent Framework

```bash
# Interactive mode
python -m src.agent_framework.main

# Single query
python -m src.agent_framework.main -q "List all tables in the lakehouse"

# Different workflow types
python -m src.agent_framework.main -w handoff     # Default: Handoff routing
python -m src.agent_framework.main -w concurrent  # Parallel execution
python -m src.agent_framework.main -w sequential  # Pipeline execution
```

```python
# Python API
import asyncio
from src.agent_framework import run_workflow

result = asyncio.run(run_workflow(
    query="Show me the top 10 customers by revenue",
    workflow_type="handoff"
))
```

### LangGraph

```bash
# Interactive mode
python -m src.langgraph.main

# Single query
python -m src.langgraph.main -q "Query the warehouse for sales data"

# Different graph types
python -m src.langgraph.main -g supervisor   # Default: Supervisor routing
python -m src.langgraph.main -g parallel     # Fan-out/fan-in
python -m src.langgraph.main -g sequential   # Pipeline execution
```

```python
# Python API
import asyncio
from src.langgraph import run_graph

result = asyncio.run(run_graph(
    query="Show me the top 10 customers by revenue",
    graph_type="supervisor"
))
```

## Architecture Comparison

| Aspect | Agent Framework | LangGraph |
|--------|-----------------|-----------|
| **State Management** | Implicit via workflow context | Explicit TypedDict with reducers |
| **Graph Definition** | Builder pattern (HandoffBuilder) | Code-first StateGraph |
| **Tool Decoration** | `@ai_function` | `@tool` |
| **Streaming** | WorkflowEvent streaming | astream / astream_events |
| **Checkpointing** | CheckpointStorage interface | Built-in MemorySaver |
| **Human-in-Loop** | RequestInfoEvent | interrupt_before/after |

## Authentication

Both implementations use `DefaultAzureCredential` for authentication:

1. **Development**: Uses Azure CLI credentials (`az login`)
2. **Production**: Uses Managed Identity

Required permissions:
- `Cognitive Services User` on Azure OpenAI resource
- `Contributor` on Fabric workspace
- `Storage Blob Data Contributor` on OneLake (for direct storage access)

## Development

```bash
# Run tests
pytest

# Format code
black src/
ruff check src/

# Type checking
mypy src/
```

## License

MIT

## References

- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Microsoft Fabric REST APIs](https://learn.microsoft.com/rest/api/fabric/)
- [Azure OpenAI Service](https://learn.microsoft.com/azure/ai-services/openai/)
