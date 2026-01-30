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
