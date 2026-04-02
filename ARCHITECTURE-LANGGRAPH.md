# Multi-Fabric Data Agent Orchestration Architecture (LangGraph)

## Overview

This architecture describes a **Foundry Agent** that leverages **Azure AI Agent Service**, **Microsoft Foundry** (formerly Azure AI Foundry), and **LangGraph** to orchestrate multiple **Fabric Data Agents** for unified data operations across Microsoft Fabric workspaces.

> **Note:** For an alternative implementation using Microsoft Agent Framework, see [ARCHITECTURE-AGENT-FRAMEWORK.md](ARCHITECTURE-AGENT-FRAMEWORK.md).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER / APPLICATION                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH ORCHESTRATOR                                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                    Microsoft Foundry Project                              │  │
│  │  • Azure OpenAI Model Deployment (GPT-4o / GPT-4.1)                      │  │
│  │  • LangGraph StateGraph Orchestration                                    │  │
│  │  • Checkpointer for State Persistence                                    │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│   FABRIC DATA AGENT   │ │   FABRIC DATA AGENT   │ │   FABRIC DATA AGENT   │
│      (Lakehouse)      │ │     (Warehouse)       │ │    (Real-Time)        │
│  ─────────────────    │ │  ─────────────────    │ │  ─────────────────    │
│  • Query Execution    │ │  • SQL Analytics      │ │  • Stream Processing  │
│  • Data Exploration   │ │  • Data Modeling      │ │  • Event Handling     │
│  • Delta Lake Ops     │ │  • Semantic Models    │ │  • KQL Queries        │
└───────────────────────┘ └───────────────────────┘ └───────────────────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MICROSOFT FABRIC PLATFORM                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Lakehouse  │  │  Warehouse  │  │  Real-Time  │  │   OneLake   │            │
│  │             │  │             │  │ Intelligence│  │             │            │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Components

### 1. LangGraph Orchestrator

The central coordinator that manages all Fabric data agents using LangGraph's StateGraph for workflow orchestration.

| Component | Description |
|-----------|-------------|
| **Host** | Microsoft Foundry (Azure OpenAI) |
| **Model** | GPT-4o or GPT-4.1 via AzureChatOpenAI |
| **Framework** | LangGraph with StateGraph |
| **Role** | Route, coordinate, and manage state across Fabric agents |

#### LangGraph Advantages
- **Explicit State Management** - TypedDict state schemas with full control
- **Graph-based Workflows** - Visual, debuggable agent flows
- **Conditional Routing** - Dynamic edge routing based on state
- **Checkpointing** - Built-in persistence for long-running workflows
- **Human-in-the-Loop** - Native interrupt/resume capabilities
- **Streaming** - First-class streaming support for real-time responses

### 2. Fabric Data Agents (Nodes)

Specialized agents implemented as LangGraph nodes that interact with specific Microsoft Fabric workloads.

| Agent Node | Fabric Workload | Capabilities |
|------------|-----------------|--------------|
| **lakehouse_node** | Lakehouse | Delta Lake queries, Spark jobs, data exploration |
| **warehouse_node** | Data Warehouse | T-SQL queries, semantic models, data modeling |
| **realtime_node** | Real-Time Intelligence | KQL queries, event streams, real-time analytics |
| **pipeline_node** | Data Factory | Data pipeline orchestration, ETL operations |
| **powerbi_node** | Power BI | Report generation, dataset refresh, DAX queries |

### 3. Microsoft Foundry Integration

```
┌────────────────────────────────────────────────────────────────────┐
│                    MICROSOFT FOUNDRY PROJECT                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                 Azure OpenAI Resource                       │    │
│  │  • Endpoint: https://<resource>.openai.azure.com           │    │
│  │  • Model Deployments: gpt-4o, gpt-4.1                      │    │
│  │  • Used by: AzureChatOpenAI in LangChain                   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                   Azure AI Search (Optional)                │    │
│  │  • Fabric documentation embeddings                         │    │
│  │  • Schema metadata for Fabric workspaces                   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    Connections                              │    │
│  │  • Microsoft Fabric workspace connections                  │    │
│  │  • Azure Storage (OneLake) connections                     │    │
│  └────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

---

## LangGraph State Schema

### Core State Definition

```python
from typing import TypedDict, Annotated, Literal, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class FabricAgentState(TypedDict):
    """State schema for multi-fabric agent orchestration."""
    
    # Conversation messages with reducer for appending
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Current agent handling the request
    current_agent: Literal["orchestrator", "lakehouse", "warehouse", "realtime", "pipeline", "powerbi"]
    
    # Workspace context
    workspace_id: str
    
    # Query metadata
    query_type: Literal["spark", "tsql", "kql", "dax", "pipeline"] | None
    
    # Results from agent execution
    query_results: dict | None
    
    # Error state
    error: str | None
    
    # Routing decision from orchestrator
    next_agent: str | None
    
    # Human-in-the-loop flag
    requires_approval: bool
```

### Extended State for Multi-Source Queries

```python
from typing import TypedDict, Annotated
from operator import add

class MultiSourceState(FabricAgentState):
    """Extended state for concurrent multi-source queries."""
    
    # Parallel execution results (uses add reducer for merging)
    parallel_results: Annotated[list[dict], add]
    
    # Agents to execute in parallel
    parallel_agents: list[str]
    
    # Aggregated final result
    aggregated_result: dict | None
```

---

## Orchestration Patterns

### Pattern 1: Supervisor Orchestration (Recommended)

The orchestrator (supervisor) routes requests to specialized agent nodes based on intent classification.

```
                         ┌─────────────────┐
                         │   Orchestrator  │
                         │   (Supervisor)  │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │ lakehouse_node │  │ warehouse_node │  │  realtime_node │
     └────────────────┘  └────────────────┘  └────────────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │      END        │
                         └─────────────────┘
```

```python
from typing import Literal
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-08-01-preview",
    temperature=0
)

def orchestrator_node(state: FabricAgentState) -> dict:
    """Supervisor node that routes to appropriate Fabric agent."""
    
    system_prompt = """You are a Fabric Data Orchestrator. Analyze the user's request and route to:
    - "lakehouse": For Delta Lake, Spark SQL, PySpark, and lakehouse operations
    - "warehouse": For T-SQL queries, data warehouse analytics, semantic models
    - "realtime": For KQL queries, streaming data, real-time analytics
    - "pipeline": For data pipeline orchestration, ETL operations
    - "powerbi": For reports, dashboards, DAX queries
    - "end": If the request is complete or cannot be handled
    
    Respond with ONLY the agent name."""
    
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    
    next_agent = response.content.strip().lower()
    
    return {
        "next_agent": next_agent,
        "current_agent": "orchestrator"
    }

def route_to_agent(state: FabricAgentState) -> Literal["lakehouse", "warehouse", "realtime", "pipeline", "powerbi", "end"]:
    """Conditional edge function for routing."""
    next_agent = state.get("next_agent", "end")
    if next_agent in ["lakehouse", "warehouse", "realtime", "pipeline", "powerbi"]:
        return next_agent
    return "end"

# Build the graph
workflow = StateGraph(FabricAgentState)

# Add nodes
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("lakehouse", lakehouse_node)
workflow.add_node("warehouse", warehouse_node)
workflow.add_node("realtime", realtime_node)
workflow.add_node("pipeline", pipeline_node)
workflow.add_node("powerbi", powerbi_node)

# Set entry point
workflow.set_entry_point("orchestrator")

# Add conditional edges from orchestrator
workflow.add_conditional_edges(
    "orchestrator",
    route_to_agent,
    {
        "lakehouse": "lakehouse",
        "warehouse": "warehouse",
        "realtime": "realtime",
        "pipeline": "pipeline",
        "powerbi": "powerbi",
        "end": END
    }
)

# All agents return to orchestrator for potential follow-up
for agent in ["lakehouse", "warehouse", "realtime", "pipeline", "powerbi"]:
    workflow.add_edge(agent, "orchestrator")

# Compile the graph
app = workflow.compile()
```

### Pattern 2: Parallel Fan-Out/Fan-In

For queries requiring data from multiple Fabric sources simultaneously.

```
                    ┌─────────────────┐
                    │   Orchestrator  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
   │ lakehouse_node │ │ warehouse_node │ │ realtime_node  │
   └────────┬───────┘ └────────┬───────┘ └────────┬───────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │   Aggregator    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      END        │
                    └─────────────────┘
```

```python
from langgraph.graph import StateGraph, END
from langgraph.constants import Send

def fan_out_node(state: MultiSourceState) -> list[Send]:
    """Fan-out to multiple agents in parallel."""
    parallel_agents = state.get("parallel_agents", [])
    
    return [
        Send(agent, {"messages": state["messages"], "workspace_id": state["workspace_id"]})
        for agent in parallel_agents
    ]

def aggregator_node(state: MultiSourceState) -> dict:
    """Aggregate results from parallel agent executions."""
    results = state.get("parallel_results", [])
    
    # Combine results from all agents
    aggregated = {
        "sources": len(results),
        "data": results,
        "summary": f"Aggregated results from {len(results)} Fabric sources"
    }
    
    return {"aggregated_result": aggregated}

# Build parallel workflow
parallel_workflow = StateGraph(MultiSourceState)

parallel_workflow.add_node("fan_out", fan_out_node)
parallel_workflow.add_node("lakehouse", lakehouse_node)
parallel_workflow.add_node("warehouse", warehouse_node)
parallel_workflow.add_node("realtime", realtime_node)
parallel_workflow.add_node("aggregator", aggregator_node)

parallel_workflow.set_entry_point("fan_out")

# Fan-out edges (using Send for parallel execution)
parallel_workflow.add_conditional_edges("fan_out", fan_out_node)

# All parallel nodes converge to aggregator
for agent in ["lakehouse", "warehouse", "realtime"]:
    parallel_workflow.add_edge(agent, "aggregator")

parallel_workflow.add_edge("aggregator", END)

parallel_app = parallel_workflow.compile()
```

### Pattern 3: Sequential Pipeline

For multi-step data pipelines where output of one agent feeds into another.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Lakehouse  │───►│  Warehouse  │───►│   PowerBI   │───►│     END     │
│   (Extract) │    │ (Transform) │    │   (Report)  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

```python
from langgraph.graph import StateGraph, END

# Build sequential workflow
sequential_workflow = StateGraph(FabricAgentState)

sequential_workflow.add_node("lakehouse", lakehouse_node)
sequential_workflow.add_node("warehouse", warehouse_node)
sequential_workflow.add_node("powerbi", powerbi_node)

sequential_workflow.set_entry_point("lakehouse")

# Linear edges
sequential_workflow.add_edge("lakehouse", "warehouse")
sequential_workflow.add_edge("warehouse", "powerbi")
sequential_workflow.add_edge("powerbi", END)

sequential_app = sequential_workflow.compile()
```

### Pattern 4: Human-in-the-Loop with Checkpointing

For workflows requiring approval before executing sensitive operations.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

def should_continue(state: FabricAgentState) -> Literal["execute", "human_review"]:
    """Check if human approval is required."""
    if state.get("requires_approval", False):
        return "human_review"
    return "execute"

def human_review_node(state: FabricAgentState) -> dict:
    """Interrupt for human review."""
    # This node will pause execution for human input
    return {"current_agent": "human_review"}

# Build workflow with checkpointing
checkpointer = MemorySaver()

hitl_workflow = StateGraph(FabricAgentState)

hitl_workflow.add_node("orchestrator", orchestrator_node)
hitl_workflow.add_node("human_review", human_review_node)
hitl_workflow.add_node("execute", execute_node)

hitl_workflow.set_entry_point("orchestrator")

hitl_workflow.add_conditional_edges(
    "orchestrator",
    should_continue,
    {
        "execute": "execute",
        "human_review": "human_review"
    }
)

# After human review, continue to execute
hitl_workflow.add_edge("human_review", "execute")
hitl_workflow.add_edge("execute", END)

# Compile with checkpointer for state persistence
hitl_app = hitl_workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"]  # Interrupt before human review
)

# Usage with thread for state persistence
config = {"configurable": {"thread_id": "fabric-session-123"}}

# Initial run (will pause at human_review)
result = hitl_app.invoke(initial_state, config)

# Resume after human approval
result = hitl_app.invoke(None, config)  # Continues from checkpoint
```

---

## Fabric Agent Node Implementations

### Lakehouse Agent Node

```python
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

@tool
def execute_spark_sql(workspace_id: str, query: str, max_rows: int = 1000) -> dict:
    """Execute Spark SQL query against a Fabric Lakehouse.
    
    Args:
        workspace_id: The Fabric workspace GUID
        query: Spark SQL query to execute
        max_rows: Maximum rows to return (default 1000)
    
    Returns:
        Query results as dictionary with columns and data
    """
    # Implementation using Fabric REST API
    return fabric_auth.execute_query(workspace_id, "spark", query, max_rows)

@tool
def list_lakehouse_tables(workspace_id: str, lakehouse_name: str) -> list[str]:
    """List all tables in a Fabric Lakehouse.
    
    Args:
        workspace_id: The Fabric workspace GUID
        lakehouse_name: Name of the lakehouse
    
    Returns:
        List of table names
    """
    return onelake_client.list_tables(workspace_id, lakehouse_name)

@tool
def describe_delta_table(workspace_id: str, lakehouse_name: str, table_name: str) -> dict:
    """Get schema and metadata for a Delta table.
    
    Args:
        workspace_id: The Fabric workspace GUID
        lakehouse_name: Name of the lakehouse
        table_name: Name of the Delta table
    
    Returns:
        Table schema and metadata
    """
    return onelake_client.describe_table(workspace_id, lakehouse_name, table_name)

# Create Lakehouse agent using ReAct pattern
lakehouse_tools = [execute_spark_sql, list_lakehouse_tables, describe_delta_table]

lakehouse_agent = create_react_agent(
    llm,
    lakehouse_tools,
    state_modifier="""You are a Microsoft Fabric Lakehouse specialist.

Your capabilities:
- Execute Spark SQL and PySpark queries on Delta tables
- Navigate lakehouse schemas and table structures
- Perform data exploration and profiling
- Read and analyze Delta Lake table metadata

Guidelines:
- Always validate table existence before querying
- Limit results appropriately to avoid memory issues
- Use DESCRIBE and SHOW commands for schema discovery
- Return results in structured, readable format"""
)

def lakehouse_node(state: FabricAgentState) -> dict:
    """Lakehouse agent node for LangGraph."""
    result = lakehouse_agent.invoke({
        "messages": state["messages"]
    })
    
    return {
        "messages": result["messages"],
        "current_agent": "lakehouse",
        "query_type": "spark"
    }
```

### Warehouse Agent Node

```python
@tool
def execute_tsql(workspace_id: str, query: str, max_rows: int = 1000) -> dict:
    """Execute T-SQL query against a Fabric Data Warehouse.
    
    Args:
        workspace_id: The Fabric workspace GUID
        query: T-SQL query to execute
        max_rows: Maximum rows to return (default 1000)
    
    Returns:
        Query results as dictionary with columns and data
    """
    return fabric_sql_client.execute_query(query)

@tool
def list_warehouse_schemas(workspace_id: str, warehouse_name: str) -> list[str]:
    """List all schemas in a Fabric Data Warehouse.
    
    Args:
        workspace_id: The Fabric workspace GUID
        warehouse_name: Name of the warehouse
    
    Returns:
        List of schema names
    """
    query = "SELECT schema_name FROM information_schema.schemata"
    result = fabric_sql_client.execute_query(query)
    return [row["schema_name"] for row in result]

@tool
def get_semantic_model(workspace_id: str, dataset_name: str) -> dict:
    """Get semantic model definition from a Fabric dataset.
    
    Args:
        workspace_id: The Fabric workspace GUID
        dataset_name: Name of the semantic model/dataset
    
    Returns:
        Semantic model definition including measures and relationships
    """
    return fabric_auth.get_semantic_model(workspace_id, dataset_name)

# Create Warehouse agent
warehouse_tools = [execute_tsql, list_warehouse_schemas, get_semantic_model]

warehouse_agent = create_react_agent(
    llm,
    warehouse_tools,
    state_modifier="""You are a Microsoft Fabric Data Warehouse specialist.

Your capabilities:
- Execute T-SQL queries against warehouse tables
- Navigate schemas, tables, and views
- Access and analyze semantic models
- Optimize query performance

Guidelines:
- Use fully qualified names (schema.table)
- Apply TOP or LIMIT clauses to prevent large result sets
- Leverage semantic model relationships when available
- Use CTEs for complex queries"""
)

def warehouse_node(state: FabricAgentState) -> dict:
    """Warehouse agent node for LangGraph."""
    result = warehouse_agent.invoke({
        "messages": state["messages"]
    })
    
    return {
        "messages": result["messages"],
        "current_agent": "warehouse",
        "query_type": "tsql"
    }
```

### Real-Time Intelligence Agent Node

```python
@tool
def execute_kql(workspace_id: str, database_name: str, query: str, max_rows: int = 1000) -> dict:
    """Execute KQL query against a Fabric Eventhouse.
    
    Args:
        workspace_id: The Fabric workspace GUID
        database_name: Name of the KQL database
        query: KQL query to execute
        max_rows: Maximum rows to return (default 1000)
    
    Returns:
        Query results as dictionary
    """
    return fabric_auth.execute_query(workspace_id, "kql", query, max_rows)

@tool
def list_eventhouse_databases(workspace_id: str) -> list[str]:
    """List all KQL databases in a Fabric Eventhouse.
    
    Args:
        workspace_id: The Fabric workspace GUID
    
    Returns:
        List of database names
    """
    return fabric_auth.list_eventhouse_databases(workspace_id)

@tool
def get_stream_status(workspace_id: str, stream_name: str) -> dict:
    """Get status of a real-time data stream.
    
    Args:
        workspace_id: The Fabric workspace GUID
        stream_name: Name of the event stream
    
    Returns:
        Stream status including throughput and latency
    """
    return fabric_auth.get_stream_status(workspace_id, stream_name)

# Create Real-Time agent
realtime_tools = [execute_kql, list_eventhouse_databases, get_stream_status]

realtime_agent = create_react_agent(
    llm,
    realtime_tools,
    state_modifier="""You are a Microsoft Fabric Real-Time Intelligence specialist.

Your capabilities:
- Execute KQL queries against Eventhouse databases
- Monitor real-time event streams
- Analyze time-series and streaming data
- Create real-time analytics and alerts

Guidelines:
- Use summarize and project operators to limit output
- Apply time filters for efficient queries
- Use bin() for time-based aggregations
- Consider ingestion latency in real-time queries"""
)

def realtime_node(state: FabricAgentState) -> dict:
    """Real-Time Intelligence agent node for LangGraph."""
    result = realtime_agent.invoke({
        "messages": state["messages"]
    })
    
    return {
        "messages": result["messages"],
        "current_agent": "realtime",
        "query_type": "kql"
    }
```

---

## Tool Definitions with LangChain

### Core Fabric Tools

```python
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field

class FabricQueryInput(BaseModel):
    """Input schema for Fabric query execution."""
    workspace_id: str = Field(description="The Fabric workspace GUID")
    query_type: str = Field(description="Type: 'spark', 'tsql', or 'kql'")
    query: str = Field(description="The query to execute")
    max_rows: int = Field(default=1000, description="Maximum rows to return")

class TableSchemaInput(BaseModel):
    """Input schema for table schema retrieval."""
    workspace_id: str = Field(description="The Fabric workspace GUID")
    item_type: str = Field(description="Type: 'lakehouse', 'warehouse', or 'eventhouse'")
    table_name: str = Field(description="Fully qualified table name")

# Structured tool with Pydantic validation
execute_fabric_query_tool = StructuredTool.from_function(
    func=execute_fabric_query,
    name="execute_fabric_query",
    description="Execute a query against Microsoft Fabric workloads",
    args_schema=FabricQueryInput
)

get_table_schema_tool = StructuredTool.from_function(
    func=get_table_schema,
    name="get_table_schema",
    description="Retrieve schema information for a Fabric table",
    args_schema=TableSchemaInput
)
```

---

## Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH DATA FLOW                                   │
└──────────────────────────────────────────────────────────────────────────────┘

1. USER REQUEST
   │
   ▼
2. LANGGRAPH ENTRY (StateGraph.invoke)
   │
   ├──► State Initialization (FabricAgentState)
   │    ├── messages: [HumanMessage(user_query)]
   │    ├── workspace_id: from config
   │    └── current_agent: None
   │
   ▼
3. ORCHESTRATOR NODE
   │
   ├──► Intent Classification (LLM call)
   │    ├── Data Query → Route to Fabric agent
   │    ├── Multi-source → Parallel fan-out
   │    └── Pipeline Task → Sequential execution
   │
   ├──► Update State
   │    └── next_agent: "lakehouse" | "warehouse" | "realtime" | ...
   │
   ▼
4. CONDITIONAL EDGE (route_to_agent)
   │
   ├──► lakehouse_node
   │    ├── ReAct agent with Spark tools
   │    └── Update: messages, query_results
   │
   ├──► warehouse_node
   │    ├── ReAct agent with T-SQL tools
   │    └── Update: messages, query_results
   │
   └──► realtime_node
        ├── ReAct agent with KQL tools
        └── Update: messages, query_results
   │
   ▼
5. RETURN TO ORCHESTRATOR (or END)
   │
   ├── Additional routing if needed
   └── Final state returned to user
   │
   ▼
6. RESPONSE (Final State)
   └── messages[-1]: AIMessage with results
```

---

## Authentication & Security

### Identity Flow

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    User     │────►│  Microsoft Entra │────►│   LangGraph     │
│             │     │       ID         │     │   Application   │
└─────────────┘     └──────────────────┘     └─────────────────┘
                                                      │
                            ┌─────────────────────────┼─────────────────────────┐
                            │                         │                         │
                            ▼                         ▼                         ▼
                    ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
                    │ Azure OpenAI  │         │ Fabric REST   │         │ OneLake       │
                    │ (LLM calls)   │         │ (Queries)     │         │ (Storage)     │
                    └───────────────┘         └───────────────┘         └───────────────┘
```

### Credential Configuration for LangChain

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI

# Get token provider for Azure OpenAI
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default"
)

# Initialize AzureChatOpenAI with Entra ID auth
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-08-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_ad_token_provider=token_provider,  # Use Entra ID instead of API key
    temperature=0
)
```

### Fabric Authentication Helper

```python
from azure.identity import DefaultAzureCredential
import requests

class FabricAuthenticator:
    """Handles authentication for Microsoft Fabric APIs."""
    
    FABRIC_SCOPE = "https://api.fabric.microsoft.com/.default"
    
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self._token = None
        self._token_expiry = 0
    
    def get_headers(self) -> dict:
        """Get authenticated headers for Fabric API calls."""
        import time
        
        if self._token is None or time.time() > self._token_expiry - 300:
            token = self.credential.get_token(self.FABRIC_SCOPE)
            self._token = token.token
            self._token_expiry = token.expires_on
        
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json"
        }
    
    def execute_query(self, workspace_id: str, query_type: str, query: str, max_rows: int = 1000) -> dict:
        """Execute query against Fabric APIs."""
        base_url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}"
        
        endpoints = {
            "spark": f"{base_url}/lakehouses/query",
            "tsql": f"{base_url}/warehouses/query",
            "kql": f"{base_url}/eventhouses/query"
        }
        
        response = requests.post(
            endpoints[query_type],
            headers=self.get_headers(),
            json={"query": query, "maxRows": max_rows}
        )
        response.raise_for_status()
        return response.json()

# Global instance
fabric_auth = FabricAuthenticator()
```

### Required Permissions

| Component | Permission | Scope |
|-----------|-----------|-------|
| Azure OpenAI | Cognitive Services User | Azure OpenAI resource |
| Fabric Workspace | Contributor or higher | Fabric workspace |
| OneLake | Storage Blob Data Contributor | OneLake storage |
| Semantic Link | Dataset.Read.All | Power BI workspace |

---

## Agent Interaction Context Management

Managing context across agent interactions is critical for coherent multi-turn conversations, cross-agent coordination, and maintaining system state across the orchestration lifecycle.

### Context Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CONTEXT MANAGEMENT ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Session       │     │   Agent         │     │   System        │
│   Context       │     │   Context       │     │   Context       │
│  ───────────    │     │  ───────────    │     │  ───────────    │
│  • User prefs   │     │  • Agent memory │     │  • Workspace    │
│  • Conv history │     │  • Tool results │     │  • Connections  │
│  • Thread ID    │     │  • Error state  │     │  • Permissions  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │     Context Store            │
                  │  • MemorySaver (in-memory)   │
                  │  • SqliteSaver (persistent)  │
                  │  • PostgresSaver (scalable)  │
                  │  • CosmosDBSaver (cloud)     │
                  └──────────────────────────────┘
```

### Context Types

#### 1. Session Context
Tracks information scoped to a user session or conversation thread.

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from datetime import datetime

class SessionContext(TypedDict):
    """Session-scoped context for multi-turn conversations."""
    
    # Unique session/thread identifier
    thread_id: str
    
    # User identity and preferences
    user_id: str | None
    user_preferences: dict[str, Any]
    
    # Session timing
    session_start: datetime
    last_activity: datetime
    
    # Conversation summary for long sessions
    conversation_summary: str | None
    
    # Token budget tracking
    token_count: int
    max_context_tokens: int
```

#### 2. Agent Context
Maintains state specific to each specialized agent during execution.

```python
class AgentContext(TypedDict):
    """Agent-specific execution context."""
    
    # Agent identification
    agent_name: str
    agent_type: Literal["lakehouse", "warehouse", "realtime", "pipeline", "powerbi"]
    
    # Execution history within this agent
    tool_calls: list[dict[str, Any]]
    intermediate_results: list[dict[str, Any]]
    
    # Agent-specific memory
    discovered_schemas: dict[str, Any]  # Cached schema info
    query_history: list[str]  # Recent queries for this agent
    
    # Performance tracking
    execution_time_ms: int
    retry_count: int
```

#### 3. Cross-Agent Context
Enables context sharing between agents during handoffs and parallel execution.

```python
class CrossAgentContext(TypedDict):
    """Context shared across agents during orchestration."""
    
    # Shared data artifacts
    shared_results: dict[str, Any]
    
    # Handoff context
    handoff_from: str | None
    handoff_reason: str | None
    handoff_context: dict[str, Any]
    
    # Parallel execution coordination
    parallel_execution_id: str | None
    pending_agents: list[str]
    completed_agents: list[str]
    
    # Data lineage tracking
    data_sources_used: list[str]
    transformations_applied: list[str]
```

### Enhanced State Schema with Context Management

```python
from typing import TypedDict, Annotated, Literal, Sequence, Any
from operator import add
from datetime import datetime

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ContextMetadata(TypedDict):
    """Metadata for context lifecycle management."""
    
    created_at: datetime
    updated_at: datetime
    version: int
    checksum: str | None


class InteractionContext(TypedDict):
    """Complete context for agent interactions."""
    
    # Session-level context
    session: SessionContext
    
    # Per-agent context (keyed by agent name)
    agents: dict[str, AgentContext]
    
    # Cross-agent shared context
    shared: CrossAgentContext
    
    # Context metadata
    metadata: ContextMetadata


class EnhancedFabricAgentState(TypedDict):
    """Enhanced state schema with comprehensive context management."""
    
    # Core conversation messages
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Current agent handling the request
    current_agent: Literal[
        "orchestrator", "lakehouse", "warehouse", 
        "realtime", "pipeline", "powerbi"
    ] | None
    
    # Workspace configuration
    workspace_id: str
    
    # Query metadata
    query_type: Literal["spark", "tsql", "kql", "dax", "pipeline"] | None
    query_results: dict[str, Any] | None
    
    # Error tracking
    error: str | None
    
    # Routing
    next_agent: Literal[
        "lakehouse", "warehouse", "realtime", 
        "pipeline", "powerbi", "end"
    ] | None
    
    # Human-in-the-loop
    requires_approval: bool
    
    # === NEW: Interaction Context ===
    interaction_context: InteractionContext
```

### Context Lifecycle Management

#### Context Creation

```python
from datetime import datetime
from uuid import uuid4
import hashlib

def create_interaction_context(
    thread_id: str | None = None,
    user_id: str | None = None,
    max_context_tokens: int = 128000
) -> InteractionContext:
    """Initialize a new interaction context.
    
    Args:
        thread_id: Optional existing thread ID. Generates new if not provided.
        user_id: Optional user identifier for preference tracking.
        max_context_tokens: Maximum tokens allowed in context window.
        
    Returns:
        Initialized InteractionContext.
    """
    now = datetime.utcnow()
    tid = thread_id or str(uuid4())
    
    return InteractionContext(
        session=SessionContext(
            thread_id=tid,
            user_id=user_id,
            user_preferences={},
            session_start=now,
            last_activity=now,
            conversation_summary=None,
            token_count=0,
            max_context_tokens=max_context_tokens,
        ),
        agents={},
        shared=CrossAgentContext(
            shared_results={},
            handoff_from=None,
            handoff_reason=None,
            handoff_context={},
            parallel_execution_id=None,
            pending_agents=[],
            completed_agents=[],
            data_sources_used=[],
            transformations_applied=[],
        ),
        metadata=ContextMetadata(
            created_at=now,
            updated_at=now,
            version=1,
            checksum=None,
        ),
    )
```

#### Context Updates During Agent Handoffs

```python
def update_context_for_handoff(
    state: EnhancedFabricAgentState,
    from_agent: str,
    to_agent: str,
    reason: str,
    context_data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Update context when handing off between agents.
    
    Args:
        state: Current workflow state.
        from_agent: Name of the source agent.
        to_agent: Name of the target agent.
        reason: Explanation for the handoff.
        context_data: Optional data to pass to the target agent.
        
    Returns:
        State update dictionary.
    """
    ctx = state["interaction_context"]
    shared = ctx["shared"]
    
    # Update shared context
    updated_shared = CrossAgentContext(
        **shared,
        handoff_from=from_agent,
        handoff_reason=reason,
        handoff_context=context_data or {},
    )
    
    # Create agent context for source if not exists
    agents = dict(ctx["agents"])
    if from_agent not in agents:
        agents[from_agent] = AgentContext(
            agent_name=from_agent,
            agent_type=from_agent,
            tool_calls=[],
            intermediate_results=[],
            discovered_schemas={},
            query_history=[],
            execution_time_ms=0,
            retry_count=0,
        )
    
    # Update metadata
    updated_metadata = ContextMetadata(
        **ctx["metadata"],
        updated_at=datetime.utcnow(),
        version=ctx["metadata"]["version"] + 1,
    )
    
    return {
        "interaction_context": InteractionContext(
            session=ctx["session"],
            agents=agents,
            shared=updated_shared,
            metadata=updated_metadata,
        )
    }
```

### Context Summarization for Long Conversations

When conversations exceed token limits, context is summarized to preserve essential information.

```python
from langchain_core.messages import SystemMessage, HumanMessage

SUMMARIZATION_PROMPT = """Summarize the following conversation history, preserving:
1. Key user intents and goals
2. Important data sources and tables referenced
3. Query results and findings
4. Any errors encountered and how they were resolved
5. Current state of the analysis

Conversation:
{conversation}

Provide a concise summary that captures the essential context for continuing the analysis."""


async def summarize_context_if_needed(
    state: EnhancedFabricAgentState,
    llm: AzureChatOpenAI,
    summarization_threshold: float = 0.8
) -> dict[str, Any]:
    """Summarize conversation context if approaching token limits.
    
    Args:
        state: Current workflow state.
        llm: Language model for summarization.
        summarization_threshold: Trigger summarization at this % of max tokens.
        
    Returns:
        State update with summarized context if triggered.
    """
    ctx = state["interaction_context"]
    session = ctx["session"]
    
    # Check if summarization is needed
    token_ratio = session["token_count"] / session["max_context_tokens"]
    if token_ratio < summarization_threshold:
        return {}  # No summarization needed
    
    # Build conversation text for summarization
    messages = state["messages"]
    conversation_text = "\n".join([
        f"{msg.type}: {msg.content}" 
        for msg in messages[-20:]  # Last 20 messages
    ])
    
    # Generate summary
    summary_prompt = SUMMARIZATION_PROMPT.format(conversation=conversation_text)
    response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
    
    # Update session with summary
    updated_session = SessionContext(
        **session,
        conversation_summary=response.content,
        last_activity=datetime.utcnow(),
    )
    
    # Trim messages to recent ones plus summary
    from langchain_core.messages import SystemMessage
    summary_message = SystemMessage(
        content=f"[Previous conversation summary]: {response.content}"
    )
    
    return {
        "messages": [summary_message] + list(messages[-5:]),
        "interaction_context": InteractionContext(
            session=updated_session,
            agents=ctx["agents"],
            shared=ctx["shared"],
            metadata=ContextMetadata(
                **ctx["metadata"],
                updated_at=datetime.utcnow(),
                version=ctx["metadata"]["version"] + 1,
            ),
        ),
    }
```

### Persistent Context Storage

#### Using LangGraph Checkpointers

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver

# Development: In-memory checkpointing
dev_checkpointer = MemorySaver()

# Local persistence: SQLite
local_checkpointer = SqliteSaver.from_conn_string("fabric_agent_context.db")

# Production: PostgreSQL with async support
async def create_production_checkpointer():
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    
    conn_string = os.environ["POSTGRES_CONNECTION_STRING"]
    checkpointer = await AsyncPostgresSaver.from_conn_string(conn_string)
    await checkpointer.setup()  # Create tables if needed
    return checkpointer

# Cloud-native: Azure Cosmos DB (custom implementation)
class CosmosDBSaver:
    """Custom checkpointer for Azure Cosmos DB."""
    
    def __init__(self, endpoint: str, database: str, container: str):
        from azure.cosmos import CosmosClient
        from azure.identity import DefaultAzureCredential
        
        credential = DefaultAzureCredential()
        self.client = CosmosClient(endpoint, credential)
        self.container = (
            self.client.get_database_client(database)
            .get_container_client(container)
        )
    
    def get(self, thread_id: str) -> dict | None:
        """Retrieve checkpoint for a thread."""
        try:
            return self.container.read_item(
                item=thread_id,
                partition_key=thread_id
            )
        except Exception:
            return None
    
    def put(self, thread_id: str, checkpoint: dict) -> None:
        """Store checkpoint for a thread."""
        checkpoint["id"] = thread_id
        checkpoint["_partition_key"] = thread_id
        self.container.upsert_item(checkpoint)
```

### Context-Aware Workflow Compilation

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

def build_context_aware_workflow(
    checkpointer: PostgresSaver | MemorySaver | None = None
) -> CompiledStateGraph:
    """Build workflow with context management.
    
    Args:
        checkpointer: Optional checkpointer for state persistence.
        
    Returns:
        Compiled StateGraph with context management.
    """
    workflow = StateGraph(EnhancedFabricAgentState)
    
    # Add context initialization node
    workflow.add_node("init_context", init_context_node)
    
    # Add core orchestration nodes
    workflow.add_node("orchestrator", context_aware_orchestrator_node)
    workflow.add_node("lakehouse", context_aware_lakehouse_node)
    workflow.add_node("warehouse", context_aware_warehouse_node)
    workflow.add_node("realtime", context_aware_realtime_node)
    
    # Add context management nodes
    workflow.add_node("summarize_context", summarize_context_node)
    workflow.add_node("persist_context", persist_context_node)
    
    # Entry point initializes context
    workflow.set_entry_point("init_context")
    workflow.add_edge("init_context", "orchestrator")
    
    # Conditional routing from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        route_with_context,
        {
            "lakehouse": "lakehouse",
            "warehouse": "warehouse",
            "realtime": "realtime",
            "summarize": "summarize_context",
            "end": END
        }
    )
    
    # Agents return through context persistence
    for agent in ["lakehouse", "warehouse", "realtime"]:
        workflow.add_edge(agent, "persist_context")
    
    workflow.add_edge("persist_context", "orchestrator")
    workflow.add_edge("summarize_context", "orchestrator")
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=checkpointer)


def route_with_context(state: EnhancedFabricAgentState) -> str:
    """Route based on next_agent and context state.
    
    Checks if context summarization is needed before routing.
    """
    ctx = state["interaction_context"]
    session = ctx["session"]
    
    # Check if context summarization is needed
    if session["token_count"] > session["max_context_tokens"] * 0.8:
        return "summarize"
    
    next_agent = state.get("next_agent", "end")
    if next_agent in ["lakehouse", "warehouse", "realtime"]:
        return next_agent
    return "end"
```

### Context-Aware Node Implementation

```python
def context_aware_orchestrator_node(
    state: EnhancedFabricAgentState
) -> dict[str, Any]:
    """Orchestrator node with context awareness.
    
    Enriches routing decisions with historical context.
    """
    llm = get_llm()
    ctx = state["interaction_context"]
    
    # Build context-aware system prompt
    context_info = ""
    if ctx["session"]["conversation_summary"]:
        context_info += f"\nPrevious context: {ctx['session']['conversation_summary']}"
    
    if ctx["shared"]["handoff_from"]:
        context_info += f"\nHandoff from {ctx['shared']['handoff_from']}: {ctx['shared']['handoff_reason']}"
    
    if ctx["shared"]["data_sources_used"]:
        context_info += f"\nData sources used: {', '.join(ctx['shared']['data_sources_used'])}"
    
    enhanced_prompt = ORCHESTRATOR_SYSTEM_PROMPT + context_info
    
    messages = [SystemMessage(content=enhanced_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    
    next_agent = response.content.strip().lower()
    valid_agents = ["lakehouse", "warehouse", "realtime", "end"]
    if next_agent not in valid_agents:
        next_agent = "end"
    
    # Update agent context and handoff info
    return {
        "next_agent": next_agent,
        "current_agent": "orchestrator",
        **update_context_for_handoff(
            state, "orchestrator", next_agent, "User request routing"
        )
    }
```

### Thread Management for Multi-User Scenarios

```python
# Usage example with thread isolation
async def handle_user_request(
    user_id: str,
    thread_id: str,
    query: str,
    workflow: CompiledStateGraph
) -> dict:
    """Handle a user request with proper context isolation.
    
    Args:
        user_id: The user's identifier.
        thread_id: The conversation thread ID.
        query: User's query.
        workflow: Compiled workflow with checkpointer.
        
    Returns:
        Agent response with updated context.
    """
    from langchain_core.messages import HumanMessage
    
    # Configuration for thread isolation
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }
    
    # Get existing state or create new
    existing_state = workflow.get_state(config)
    
    if existing_state.values:
        # Continue existing conversation
        input_state = {"messages": [HumanMessage(content=query)]}
    else:
        # New conversation - create full initial state
        input_state = create_initial_enhanced_state(
            workspace_id=os.environ["FABRIC_WORKSPACE_ID"],
            query=query,
            user_id=user_id,
            thread_id=thread_id,
        )
    
    # Invoke workflow
    result = await workflow.ainvoke(input_state, config)
    
    return result
```

---

## Deployment Architecture

### Azure Resources

```bicep
// Core infrastructure for LangGraph-based fabric agent orchestration

resource openAI 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: 'fabric-agent-openai'
  location: location
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: 'fabric-agent-openai'
  }
}

resource gpt4oDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAI
  name: 'gpt-4o'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: '2024-08-06'
    }
  }
  sku: {
    name: 'GlobalStandard'
    capacity: 50
  }
}

// Optional: Azure AI Search for RAG
resource searchService 'Microsoft.Search/searchServices@2024-03-01-preview' = {
  name: 'fabric-agent-search'
  location: location
  sku: { name: 'basic' }
  properties: {
    replicaCount: 1
    partitionCount: 1
  }
}
```

### Environment Configuration

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://fabric-agent-openai.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Microsoft Entra ID (for Service Principal auth, if used)
AZURE_TENANT_ID=<tenant-guid>
AZURE_CLIENT_ID=<client-id>
AZURE_CLIENT_SECRET=<client-secret>

# Microsoft Fabric Configuration
FABRIC_WORKSPACE_ID=<workspace-guid>
FABRIC_SQL_ENDPOINT=<workspace>.datawarehouse.fabric.microsoft.com

# LangGraph Configuration (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<langsmith-api-key>
LANGCHAIN_PROJECT=fabric-agent-orchestration
```

---

## Implementation Guide

### Step 1: Install Dependencies

```bash
pip install langchain langchain-openai langgraph azure-identity
```

### Step 2: Initialize LLM and Tools

```python
import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI

# Setup Azure OpenAI with Entra ID auth
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default"
)

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_ad_token_provider=token_provider,
    temperature=0
)
```

### Step 3: Define State and Agents

```python
from typing import TypedDict, Annotated, Literal, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class FabricAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_agent: str
    workspace_id: str
    next_agent: str | None

# Create specialized agents (see agent implementations above)
```

### Step 4: Build the Graph

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(FabricAgentState)

# Add all nodes
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("lakehouse", lakehouse_node)
workflow.add_node("warehouse", warehouse_node)
workflow.add_node("realtime", realtime_node)

# Set entry and edges
workflow.set_entry_point("orchestrator")
workflow.add_conditional_edges("orchestrator", route_to_agent, {...})

# Compile
app = workflow.compile()
```

### Step 5: Run the Workflow

```python
from langchain_core.messages import HumanMessage

# Initialize state
initial_state = {
    "messages": [HumanMessage(content="Show me the top 10 customers from the lakehouse")],
    "workspace_id": os.environ["FABRIC_WORKSPACE_ID"],
    "current_agent": None,
    "next_agent": None
}

# Run (sync)
result = app.invoke(initial_state)
print(result["messages"][-1].content)

# Run (streaming)
async for event in app.astream_events(initial_state, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

---

## Monitoring & Observability

### LangSmith Integration

```python
import os

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "<your-langsmith-api-key>"
os.environ["LANGCHAIN_PROJECT"] = "fabric-agent-orchestration"

# All LangGraph operations are now traced
```

### Custom Callbacks

```python
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any

class FabricAgentCallback(BaseCallbackHandler):
    """Custom callback for monitoring Fabric agent operations."""
    
    def on_tool_start(self, tool: dict, input_str: str, **kwargs: Any) -> None:
        print(f"🔧 Tool started: {tool['name']}")
        print(f"   Input: {input_str[:100]}...")
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        print(f"✅ Tool completed")
        print(f"   Output: {output[:100]}...")
    
    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        print(f"🤖 Agent action: {action.tool}")

# Use with LLM
llm_with_callbacks = llm.with_config(callbacks=[FabricAgentCallback()])
```

### Metrics to Track

| Metric | Description |
|--------|-------------|
| `langgraph.node.duration` | Time spent in each node |
| `langgraph.edge.transitions` | Number of edge traversals |
| `fabric.query.duration` | Time to execute Fabric query |
| `fabric.query.rows_returned` | Number of rows returned |
| `llm.tokens.total` | Total tokens used per invocation |

---

## Error Handling & Resilience

### Node-Level Error Handling

```python
from langchain_core.messages import AIMessage

def lakehouse_node_with_error_handling(state: FabricAgentState) -> dict:
    """Lakehouse agent node with error handling."""
    try:
        result = lakehouse_agent.invoke({
            "messages": state["messages"]
        })
        return {
            "messages": result["messages"],
            "current_agent": "lakehouse",
            "error": None
        }
    except Exception as e:
        error_message = f"Lakehouse agent error: {str(e)}"
        return {
            "messages": [AIMessage(content=error_message)],
            "current_agent": "lakehouse",
            "error": error_message
        }
```

### Retry with Fallback

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def execute_fabric_query_with_retry(workspace_id: str, query_type: str, query: str) -> dict:
    """Execute Fabric query with retry logic."""
    return fabric_auth.execute_query(workspace_id, query_type, query)
```

### Graph-Level Error Routing

```python
def route_on_error(state: FabricAgentState) -> Literal["retry", "fallback", "end"]:
    """Route based on error state."""
    if state.get("error"):
        if "timeout" in state["error"].lower():
            return "retry"
        elif "permission" in state["error"].lower():
            return "fallback"
    return "end"

workflow.add_conditional_edges(
    "lakehouse",
    route_on_error,
    {
        "retry": "lakehouse",  # Retry same node
        "fallback": "orchestrator",  # Return to orchestrator
        "end": END
    }
)
```

---

## Comparison: LangGraph vs Microsoft Agent Framework

| Aspect | LangGraph | Microsoft Agent Framework |
|--------|-----------|---------------------------|
| **State Management** | Explicit TypedDict with reducers | Implicit via workflow context |
| **Graph Definition** | Code-first StateGraph | Builder pattern (HandoffBuilder) |
| **Checkpointing** | Built-in MemorySaver, SQLite, Postgres | CheckpointStorage interface |
| **Streaming** | Native astream_events | WorkflowEvent streaming |
| **Human-in-Loop** | interrupt_before/after | RequestInfoEvent |
| **Parallel Execution** | Send() for fan-out | ConcurrentBuilder |
| **Observability** | LangSmith integration | OpenTelemetry |
| **LLM Integration** | LangChain ecosystem | Direct model clients |
| **Azure Integration** | Via langchain-openai | Native Azure AI SDK |

---

## Summary

This architecture provides a LangGraph-based approach to orchestrating multiple Microsoft Fabric data agents through:

1. **LangGraph StateGraph** for explicit workflow definition
2. **Azure OpenAI** via LangChain for LLM capabilities
3. **Typed State** with Pydantic validation
4. **ReAct Agents** for tool-using specialist nodes
5. **Checkpointing** for human-in-the-loop and recovery

The supervisor orchestration pattern is recommended for most scenarios, providing intelligent routing with full state visibility across agent transitions.

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Azure OpenAI](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/)
- [Microsoft Fabric REST APIs](https://learn.microsoft.com/rest/api/fabric/)
- [LangSmith Tracing](https://docs.smith.langchain.com/)
