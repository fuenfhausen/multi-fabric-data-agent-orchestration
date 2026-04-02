# Multi-Fabric Data Agent Orchestration Architecture (Microsoft Agent Framework)

## Overview

This architecture describes a **Foundry Agent** that leverages **Azure AI Agent Service**, **Microsoft Foundry** (formerly Azure AI Foundry), and the **Microsoft Agent Framework** to orchestrate multiple **Fabric Data Agents** for unified data operations across Microsoft Fabric workspaces.

> **Note:** For an alternative implementation using LangGraph, see [ARCHITECTURE-LANGGRAPH.md](ARCHITECTURE-LANGGRAPH.md).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER / APPLICATION                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FOUNDRY ORCHESTRATOR AGENT                               │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                    Microsoft Foundry Project                              │  │
│  │  • Azure AI Agent Service (Hosted Agent Runtime)                         │  │
│  │  • Model Deployment (GPT-4o / GPT-4.1)                                    │  │
│  │  • Agent Framework Orchestration Engine                                  │  │
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

### 1. Foundry Orchestrator Agent

The central coordinator that manages all Fabric data agents using Microsoft Agent Framework orchestration patterns.

| Component | Description |
|-----------|-------------|
| **Host** | Microsoft Foundry (Azure AI Agent Service) |
| **Model** | GPT-4o or GPT-4.1 (recommended for complex reasoning) |
| **Framework** | Microsoft Agent Framework |
| **Role** | Triage, route, and coordinate data operations across Fabric agents |

#### Orchestrator Responsibilities
- Parse user intent and determine which Fabric data agent(s) to invoke
- Coordinate multi-agent workflows (sequential, concurrent, handoff)
- Aggregate results from multiple data agents
- Maintain conversation context across agent handoffs

### 2. Fabric Data Agents (Specialists)

Specialized agents that interact with specific Microsoft Fabric workloads.

| Agent | Fabric Workload | Capabilities |
|-------|-----------------|--------------|
| **Lakehouse Agent** | Lakehouse | Delta Lake queries, Spark jobs, data exploration |
| **Warehouse Agent** | Data Warehouse | T-SQL queries, semantic models, data modeling |
| **Real-Time Agent** | Real-Time Intelligence | KQL queries, event streams, real-time analytics |
| **Pipeline Agent** | Data Factory | Data pipeline orchestration, ETL operations |
| **Power BI Agent** | Power BI | Report generation, dataset refresh, DAX queries |

### 3. Microsoft Foundry Integration

```
┌────────────────────────────────────────────────────────────────────┐
│                    MICROSOFT FOUNDRY PROJECT                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                 Azure AI Services Resource                  │    │
│  │  • Endpoint: https://<resource>.services.ai.azure.com      │    │
│  │  • Model Deployments: gpt-4o, gpt-4.1                      │    │
│  │  • Agent Service: Hosted agent runtime                     │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                   Knowledge Indexes                         │    │
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

## Orchestration Patterns

### Pattern 1: Handoff Orchestration (Recommended)

Best for scenarios where the orchestrator triages requests to the most appropriate specialist.

```
User Query → Orchestrator → [Triage] → Specialist Agent → Response
                                ↓
                    (handoff_to_lakehouse_agent)
                    (handoff_to_warehouse_agent)
                    (handoff_to_realtime_agent)
```

```python
from agent_framework import ChatAgent, HandoffBuilder
from agent_framework.azure import AzureOpenAIChatClient

# Create the orchestrator agent
orchestrator = ChatAgent(
    chat_client=client,
    name="fabric_orchestrator",
    instructions="""You are a Fabric Data Orchestrator. Route queries to:
    - lakehouse_agent: For Delta Lake, Spark, and lakehouse operations
    - warehouse_agent: For SQL analytics, data modeling, semantic models
    - realtime_agent: For streaming data, KQL queries, real-time analytics
    Always handoff to the most appropriate specialist."""
)

# Build handoff workflow
workflow = (
    HandoffBuilder(
        name="fabric_data_orchestration",
        participants=[orchestrator, lakehouse_agent, warehouse_agent, realtime_agent]
    )
    .with_start_agent(orchestrator)
    .add_handoff(orchestrator, [lakehouse_agent, warehouse_agent, realtime_agent])
    .add_handoff(lakehouse_agent, [orchestrator])
    .add_handoff(warehouse_agent, [orchestrator])
    .add_handoff(realtime_agent, [orchestrator])
    .build()
)
```

### Pattern 2: Concurrent Orchestration

Best for queries that require data from multiple Fabric sources simultaneously.

```
                    ┌─► Lakehouse Agent ──┐
User Query → Fan-Out ├─► Warehouse Agent ──├─► Aggregator → Response
                    └─► Real-Time Agent ──┘
```

```python
from agent_framework import ConcurrentBuilder

# Fan-out to multiple agents, fan-in with aggregator
workflow = (
    ConcurrentBuilder()
    .participants([lakehouse_agent, warehouse_agent, realtime_agent])
    .with_aggregator(aggregator_callback)
    .build()
)
```

### Pattern 3: Sequential Orchestration

Best for multi-step data pipelines where output of one agent feeds into another.

```
User Query → Lakehouse Agent → Warehouse Agent → Power BI Agent → Response
```

```python
from agent_framework import AgentWorkflowBuilder

workflow = AgentWorkflowBuilder.BuildSequential(
    workflowName="fabric_pipeline",
    agents=[lakehouse_agent, warehouse_agent, powerbi_agent]
)
```

### Pattern 4: Autonomous Mode with Iteration

Best for complex analytical tasks requiring multiple iterations.

```python
workflow = (
    HandoffBuilder(participants=[orchestrator, analyst_agent])
    .with_autonomous_mode(
        agents=[analyst_agent],
        turn_limits={"analyst_agent": 10}  # Allow 10 iterations
    )
    .build()
)
```

---

## Fabric Data Agent Specifications

### Lakehouse Agent

```python
lakehouse_agent = ChatAgent(
    chat_client=client,
    name="lakehouse_agent",
    instructions="""You are a Microsoft Fabric Lakehouse specialist.
    
    Capabilities:
    - Execute Spark SQL and PySpark queries
    - Navigate Delta Lake tables and schemas
    - Perform data exploration and profiling
    - Manage lakehouse files and folders
    
    Always validate table names exist before querying.
    Return results in structured format.""",
    tools=[
        execute_spark_sql_tool,
        list_tables_tool,
        describe_table_tool,
        read_delta_table_tool
    ]
)
```

### Warehouse Agent

```python
warehouse_agent = ChatAgent(
    chat_client=client,
    name="warehouse_agent",
    instructions="""You are a Microsoft Fabric Data Warehouse specialist.
    
    Capabilities:
    - Execute T-SQL queries against warehouse tables
    - Manage semantic models and relationships
    - Create and modify views and procedures
    - Optimize query performance
    
    Always use fully qualified table names (schema.table).
    Limit results to 1000 rows unless specified.""",
    tools=[
        execute_tsql_tool,
        list_schemas_tool,
        get_semantic_model_tool,
        create_view_tool
    ]
)
```

### Real-Time Agent

```python
realtime_agent = ChatAgent(
    chat_client=client,
    name="realtime_agent",
    instructions="""You are a Microsoft Fabric Real-Time Intelligence specialist.
    
    Capabilities:
    - Execute KQL queries against Eventhouse databases
    - Monitor real-time event streams
    - Create real-time dashboards and alerts
    - Analyze time-series data
    
    Optimize KQL queries for performance.
    Use summarize and project to limit output.""",
    tools=[
        execute_kql_tool,
        list_databases_tool,
        get_stream_status_tool,
        create_alert_tool
    ]
)
```

---

## Tool Definitions

### Core Fabric Tools

```python
from agent_framework import ai_function
from typing import Annotated

@ai_function
def execute_fabric_query(
    workspace_id: Annotated[str, "The Fabric workspace GUID"],
    query_type: Annotated[str, "Type: 'spark', 'tsql', or 'kql'"],
    query: Annotated[str, "The query to execute"],
    max_rows: Annotated[int, "Maximum rows to return"] = 1000
) -> dict:
    """Execute a query against Microsoft Fabric workloads."""
    # Implementation connects to Fabric REST APIs
    pass

@ai_function
def get_table_schema(
    workspace_id: Annotated[str, "The Fabric workspace GUID"],
    item_type: Annotated[str, "Type: 'lakehouse', 'warehouse', or 'eventhouse'"],
    table_name: Annotated[str, "Fully qualified table name"]
) -> dict:
    """Retrieve schema information for a Fabric table."""
    pass

@ai_function
def list_workspace_items(
    workspace_id: Annotated[str, "The Fabric workspace GUID"],
    item_types: Annotated[list[str], "Filter by item types"] = None
) -> list:
    """List items in a Microsoft Fabric workspace."""
    pass
```

---

## Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                        │
└──────────────────────────────────────────────────────────────────────────────┘

1. USER REQUEST
   │
   ▼
2. FOUNDRY ORCHESTRATOR receives request
   │
   ├──► Intent Classification (What does user want?)
   │    ├── Data Query → Route to appropriate Fabric agent
   │    ├── Multi-source Analysis → Concurrent execution
   │    └── Pipeline Task → Sequential execution
   │
   ▼
3. AGENT SELECTION & HANDOFF
   │
   ├──► Lakehouse Agent
   │    └── Spark/Delta operations via Fabric REST API
   │
   ├──► Warehouse Agent
   │    └── T-SQL operations via Fabric SQL endpoint
   │
   └──► Real-Time Agent
        └── KQL operations via Eventhouse endpoint
   │
   ▼
4. RESULT AGGREGATION
   │
   ├── Single agent → Direct response
   └── Multiple agents → Aggregator combines results
   │
   ▼
5. RESPONSE TO USER
```

---

## Authentication & Security

### Identity Flow Overview

The multi-fabric agent orchestration uses a layered authentication model where credentials flow from the user through Microsoft Entra ID to various Fabric endpoints.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         AUTHENTICATION FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    User     │────►│  Microsoft Entra │────►│ Foundry Project │
│             │     │       ID         │     │                 │
└─────────────┘     └──────────────────┘     └─────────────────┘
                                                      │
                            ┌─────────────────────────┼─────────────────────────┐
                            │                         │                         │
                            ▼                         ▼                         ▼
                    ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
                    │ Fabric API    │         │ OneLake       │         │ Fabric REST   │
                    │ (Delegated)   │         │ (Managed ID)  │         │ (Service)     │
                    └───────────────┘         └───────────────┘         └───────────────┘
```

### Authentication Methods

| Method | Use Case | Token Audience | Best For |
|--------|----------|----------------|----------|
| **Delegated (User)** | Interactive queries | `https://api.fabric.microsoft.com/.default` | User-initiated operations |
| **Managed Identity** | Background processing | `https://storage.azure.com/.default` | Automated pipelines |
| **Service Principal** | Application access | `https://api.fabric.microsoft.com/.default` | CI/CD, scheduled jobs |

### Credential Chain Implementation

The agents use `DefaultAzureCredential` which provides a seamless authentication experience across development and production environments:

```python
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential

# Option 1: Default credential chain (recommended)
# Tries: Environment → Managed Identity → Azure CLI → Interactive Browser
credential = DefaultAzureCredential()

# Option 2: Explicit chain for production
credential = ChainedTokenCredential(
    ManagedIdentityCredential(),  # First: Try managed identity (production)
    AzureCliCredential()           # Fallback: Azure CLI (development)
)
```

### Fabric API Authentication

#### Token Acquisition for Fabric REST APIs

```python
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AccessToken
import requests

class FabricAuthenticator:
    """Handles authentication for Microsoft Fabric APIs."""
    
    FABRIC_SCOPE = "https://api.fabric.microsoft.com/.default"
    STORAGE_SCOPE = "https://storage.azure.com/.default"
    
    def __init__(self, credential: DefaultAzureCredential = None):
        self.credential = credential or DefaultAzureCredential()
        self._token_cache: dict[str, AccessToken] = {}
    
    def get_fabric_token(self) -> str:
        """Get access token for Fabric REST API calls."""
        if self._is_token_expired(self.FABRIC_SCOPE):
            self._token_cache[self.FABRIC_SCOPE] = self.credential.get_token(self.FABRIC_SCOPE)
        return self._token_cache[self.FABRIC_SCOPE].token
    
    def get_storage_token(self) -> str:
        """Get access token for OneLake storage operations."""
        if self._is_token_expired(self.STORAGE_SCOPE):
            self._token_cache[self.STORAGE_SCOPE] = self.credential.get_token(self.STORAGE_SCOPE)
        return self._token_cache[self.STORAGE_SCOPE].token
    
    def _is_token_expired(self, scope: str) -> bool:
        """Check if cached token is expired or missing."""
        import time
        if scope not in self._token_cache:
            return True
        # Refresh 5 minutes before expiry
        return self._token_cache[scope].expires_on < time.time() + 300
    
    def get_headers(self) -> dict:
        """Get HTTP headers for Fabric API requests."""
        return {
            "Authorization": f"Bearer {self.get_fabric_token()}",
            "Content-Type": "application/json"
        }
```

#### Using Authentication in Fabric Tools

```python
from agent_framework import ai_function
from typing import Annotated

# Global authenticator instance (initialized once)
fabric_auth = FabricAuthenticator()

@ai_function
def execute_fabric_query(
    workspace_id: Annotated[str, "The Fabric workspace GUID"],
    query_type: Annotated[str, "Type: 'spark', 'tsql', or 'kql'"],
    query: Annotated[str, "The query to execute"],
    max_rows: Annotated[int, "Maximum rows to return"] = 1000
) -> dict:
    """Execute a query against Microsoft Fabric workloads."""
    import requests
    
    base_url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}"
    headers = fabric_auth.get_headers()
    
    if query_type == "tsql":
        # SQL Analytics endpoint
        endpoint = f"{base_url}/warehouses/query"
        payload = {"query": query, "maxRows": max_rows}
    elif query_type == "spark":
        # Lakehouse Spark endpoint
        endpoint = f"{base_url}/lakehouses/query"
        payload = {"query": query, "maxRows": max_rows}
    elif query_type == "kql":
        # Eventhouse KQL endpoint
        endpoint = f"{base_url}/eventhouses/query"
        payload = {"query": query, "maxRows": max_rows}
    else:
        raise ValueError(f"Unsupported query type: {query_type}")
    
    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()
```

### OneLake Authentication

OneLake uses Azure Storage authentication with ABFS (Azure Blob File System) protocol:

```python
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

class OneLakeClient:
    """Client for OneLake data operations."""
    
    def __init__(self, workspace_id: str, credential: DefaultAzureCredential = None):
        self.workspace_id = workspace_id
        self.credential = credential or DefaultAzureCredential()
        
        # OneLake endpoint format
        account_url = f"https://onelake.dfs.fabric.microsoft.com"
        self.service_client = DataLakeServiceClient(
            account_url=account_url,
            credential=self.credential
        )
    
    def read_delta_table(self, lakehouse_name: str, table_name: str) -> list[dict]:
        """Read data from a Delta table in OneLake."""
        # Path format: <workspace_id>/<lakehouse_name>/Tables/<table_name>
        file_system = self.service_client.get_file_system_client(self.workspace_id)
        directory = file_system.get_directory_client(f"{lakehouse_name}/Tables/{table_name}")
        
        # Implementation would use delta-rs or similar library
        # to read the Delta table format
        pass
    
    def list_tables(self, lakehouse_name: str) -> list[str]:
        """List all tables in a lakehouse."""
        file_system = self.service_client.get_file_system_client(self.workspace_id)
        tables_dir = file_system.get_directory_client(f"{lakehouse_name}/Tables")
        
        return [path.name for path in tables_dir.get_paths() if path.is_directory]
```

### SQL Endpoint Authentication

For warehouse and lakehouse SQL endpoints, use pyodbc with Microsoft Entra ID authentication:

```python
import pyodbc
from azure.identity import DefaultAzureCredential
import struct

class FabricSQLClient:
    """Client for Fabric SQL endpoint queries."""
    
    def __init__(self, server: str, database: str, credential: DefaultAzureCredential = None):
        self.server = server
        self.database = database
        self.credential = credential or DefaultAzureCredential()
    
    def _get_connection(self) -> pyodbc.Connection:
        """Create authenticated connection to Fabric SQL endpoint."""
        # Get access token
        token = self.credential.get_token("https://database.windows.net/.default")
        token_bytes = token.token.encode("UTF-16-LE")
        token_struct = struct.pack(f'<I{len(token_bytes)}s', len(token_bytes), token_bytes)
        
        # Connection string for Fabric SQL endpoint
        conn_str = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
        )
        
        conn = pyodbc.connect(conn_str, attrs_before={
            1256: token_struct  # SQL_COPT_SS_ACCESS_TOKEN
        })
        return conn
    
    def execute_query(self, query: str) -> list[dict]:
        """Execute a T-SQL query and return results."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()
```

### Required Permissions Matrix

| Component | Role/Permission | Scope | Purpose |
|-----------|----------------|-------|---------|
| **Azure AI Services** | `Cognitive Services User` | AI Services resource | Agent model invocation |
| **Azure AI Services** | `Cognitive Services Contributor` | AI Services resource | Agent management |
| **Fabric Workspace** | `Contributor` | Workspace | Query execution, item access |
| **Fabric Workspace** | `Admin` | Workspace | Manage connections, security |
| **OneLake** | `Storage Blob Data Contributor` | Workspace storage | Read/write Delta tables |
| **OneLake** | `Storage Blob Data Reader` | Workspace storage | Read-only access |
| **Power BI** | `Dataset.Read.All` | Power BI workspace | Semantic model access |
| **Power BI** | `Report.ReadWrite.All` | Power BI workspace | Report generation |

### Service Principal Setup

For production deployments, create a service principal with appropriate permissions:

```bash
# Create service principal
az ad sp create-for-rbac --name "fabric-agent-orchestrator" --role "Contributor" \
    --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group>

# Grant Fabric workspace access (via Fabric Admin Portal or API)
# 1. Add service principal to Fabric workspace as Contributor
# 2. Enable "Service principals can use Fabric APIs" in tenant settings
```

### Environment Variables

```env
# Microsoft Entra ID (for Service Principal auth)
AZURE_TENANT_ID=<tenant-guid>
AZURE_CLIENT_ID=<client-id>
AZURE_CLIENT_SECRET=<client-secret>

# Microsoft Foundry Configuration
AZURE_AI_SERVICES_ENDPOINT=https://fabric-agent-ai.services.ai.azure.com
AZURE_OPENAI_ENDPOINT=https://fabric-agent-ai.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Microsoft Fabric Configuration
FABRIC_WORKSPACE_ID=<workspace-guid>
FABRIC_SQL_ENDPOINT=<workspace>.datawarehouse.fabric.microsoft.com
FABRIC_LAKEHOUSE_SQL_ENDPOINT=<workspace>.dfs.fabric.microsoft.com
```

### Security Best Practices

1. **Use Managed Identity** in production Azure deployments
2. **Rotate secrets** regularly for service principal authentication
3. **Apply least privilege** - grant only required permissions per agent
4. **Enable audit logging** on Fabric workspace for compliance
5. **Use private endpoints** for sensitive data workloads
6. **Implement token caching** to reduce authentication overhead

```python
# Example: Secure credential handling with Azure Key Vault
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

def get_secure_credential():
    """Retrieve credentials from Azure Key Vault."""
    credential = DefaultAzureCredential()
    vault_url = "https://fabric-agent-vault.vault.azure.net"
    
    secret_client = SecretClient(vault_url=vault_url, credential=credential)
    
    # For scenarios requiring explicit credentials
    client_secret = secret_client.get_secret("fabric-agent-client-secret")
    return client_secret.value
```

---

## Agent Interaction Context Management

Managing context across agent interactions is essential for maintaining coherent multi-turn conversations, enabling effective agent handoffs, and preserving system state throughout the orchestration lifecycle.

### Context Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CONTEXT MANAGEMENT ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Session       │     │   Agent         │     │   System        │
│   Context       │     │   Context       │     │   Context       │
│  ───────────    │     │  ───────────    │     │  ───────────    │
│  • Thread state │     │  • Agent memory │     │  • Workspace    │
│  • Conv history │     │  • Tool results │     │  • Connections  │
│  • User prefs   │     │  • Handoff data │     │  • Permissions  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │    Context Store             │
                  │  • In-Memory (dev)           │
                  │  • Azure Cosmos DB (prod)    │
                  │  • Azure Redis Cache         │
                  │  • Thread-based isolation    │
                  └──────────────────────────────┘
```

### Context Types

#### 1. Session Context
Tracks information scoped to a user session or conversation thread.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class SessionContext:
    """Session-scoped context for multi-turn conversations."""
    
    # Unique session/thread identifier
    thread_id: str
    
    # User identity and preferences
    user_id: str | None = None
    user_preferences: dict[str, Any] = field(default_factory=dict)
    
    # Session timing
    session_start: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    # Conversation summary for long sessions
    conversation_summary: str | None = None
    
    # Message history tracking
    message_count: int = 0
    max_messages_before_summary: int = 20
```

#### 2. Agent Context
Maintains state specific to each specialized agent during execution.

```python
from dataclasses import dataclass, field
from typing import Literal, Any

@dataclass
class AgentContext:
    """Agent-specific execution context."""
    
    # Agent identification
    agent_name: str
    agent_type: Literal["lakehouse", "warehouse", "realtime", "pipeline", "powerbi"]
    
    # Execution history
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    intermediate_results: list[dict[str, Any]] = field(default_factory=list)
    
    # Agent-specific memory
    discovered_schemas: dict[str, Any] = field(default_factory=dict)
    query_history: list[str] = field(default_factory=list)
    
    # Performance tracking
    execution_time_ms: int = 0
    retry_count: int = 0
    
    def add_tool_call(self, tool_name: str, args: dict, result: Any) -> None:
        """Record a tool invocation."""
        self.tool_calls.append({
            "tool": tool_name,
            "args": args,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
```

#### 3. Handoff Context
Enables context passing between agents during orchestration handoffs.

```python
@dataclass
class HandoffContext:
    """Context passed during agent handoffs."""
    
    # Source and target agents
    from_agent: str
    to_agent: str
    
    # Handoff reason and metadata
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Data being passed
    shared_data: dict[str, Any] = field(default_factory=dict)
    
    # Query/task continuation info
    pending_task: str | None = None
    task_context: dict[str, Any] = field(default_factory=dict)
    
    # Data lineage
    data_sources_used: list[str] = field(default_factory=list)
    transformations_applied: list[str] = field(default_factory=list)
```

### Interaction Context Manager

A comprehensive manager class for handling all context types:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

@dataclass
class InteractionContext:
    """Complete context manager for agent interactions."""
    
    # Core contexts
    session: SessionContext
    agents: dict[str, AgentContext] = field(default_factory=dict)
    
    # Handoff tracking
    handoff_history: list[HandoffContext] = field(default_factory=list)
    current_handoff: HandoffContext | None = None
    
    # System context
    workspace_id: str = ""
    active_connections: list[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    
    @classmethod
    def create(
        cls,
        thread_id: str | None = None,
        user_id: str | None = None,
        workspace_id: str = ""
    ) -> "InteractionContext":
        """Factory method to create a new interaction context."""
        return cls(
            session=SessionContext(
                thread_id=thread_id or str(uuid4()),
                user_id=user_id,
            ),
            workspace_id=workspace_id,
        )
    
    def get_or_create_agent_context(
        self,
        agent_name: str,
        agent_type: str
    ) -> AgentContext:
        """Get existing agent context or create new one."""
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentContext(
                agent_name=agent_name,
                agent_type=agent_type,
            )
        return self.agents[agent_name]
    
    def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
        reason: str,
        shared_data: dict[str, Any] | None = None
    ) -> HandoffContext:
        """Record a handoff between agents."""
        handoff = HandoffContext(
            from_agent=from_agent,
            to_agent=to_agent,
            reason=reason,
            shared_data=shared_data or {},
        )
        
        # Archive current handoff if exists
        if self.current_handoff:
            self.handoff_history.append(self.current_handoff)
        
        self.current_handoff = handoff
        self.updated_at = datetime.utcnow()
        self.version += 1
        
        return handoff
    
    def get_context_for_agent(self, agent_name: str) -> dict[str, Any]:
        """Get relevant context for a specific agent."""
        context = {
            "thread_id": self.session.thread_id,
            "workspace_id": self.workspace_id,
            "conversation_summary": self.session.conversation_summary,
        }
        
        # Include handoff context if this agent is the target
        if self.current_handoff and self.current_handoff.to_agent == agent_name:
            context["handoff"] = {
                "from": self.current_handoff.from_agent,
                "reason": self.current_handoff.reason,
                "data": self.current_handoff.shared_data,
                "pending_task": self.current_handoff.pending_task,
            }
        
        # Include agent's own history
        if agent_name in self.agents:
            agent_ctx = self.agents[agent_name]
            context["previous_queries"] = agent_ctx.query_history[-5:]
            context["discovered_schemas"] = agent_ctx.discovered_schemas
        
        return context
```

### Context-Aware Agent Implementation

#### Enhanced Orchestrator with Context

```python
from agent_framework import ChatAgent, HandoffBuilder
from agent_framework.azure import AzureOpenAIChatClient

class ContextAwareOrchestrator:
    """Orchestrator that maintains and uses interaction context."""
    
    def __init__(
        self,
        chat_client: AzureOpenAIChatClient,
        context_store: "ContextStore"
    ):
        self.chat_client = chat_client
        self.context_store = context_store
        
        self.agent = ChatAgent(
            chat_client=chat_client,
            name="fabric_orchestrator",
            description="Context-aware Fabric data orchestrator",
            instructions=self._build_instructions(),
        )
    
    def _build_instructions(self) -> str:
        return """You are a Microsoft Fabric Data Orchestrator with full context awareness.

When routing requests:
1. Consider the conversation history and previous agent interactions
2. Use handoff context to understand what the previous agent accomplished
3. Track data sources used across the conversation
4. Maintain continuity when users refer to previous results

Route to specialists based on:
- **Lakehouse Agent**: Delta Lake, Spark SQL, PySpark, lakehouse files
- **Warehouse Agent**: T-SQL, data warehouse, semantic models
- **Real-Time Agent**: KQL, streaming, real-time analytics

Always include relevant context when handing off to specialists."""
    
    async def process_with_context(
        self,
        thread_id: str,
        user_message: str
    ) -> str:
        """Process a message with full context awareness."""
        
        # Load or create context
        context = await self.context_store.get_or_create(thread_id)
        
        # Update session activity
        context.session.last_activity = datetime.utcnow()
        context.session.message_count += 1
        
        # Check if summarization needed
        if context.session.message_count >= context.session.max_messages_before_summary:
            await self._summarize_conversation(context)
        
        # Build context-aware prompt
        system_context = self._build_context_prompt(context)
        
        # Process with agent (context injected into system prompt)
        response = await self.agent.process(
            message=user_message,
            system_context=system_context,
        )
        
        # Save updated context
        await self.context_store.save(context)
        
        return response
    
    def _build_context_prompt(self, context: InteractionContext) -> str:
        """Build context information for the system prompt."""
        parts = []
        
        if context.session.conversation_summary:
            parts.append(f"Conversation Summary: {context.session.conversation_summary}")
        
        if context.current_handoff:
            h = context.current_handoff
            parts.append(f"Previous Agent: {h.from_agent}")
            parts.append(f"Handoff Reason: {h.reason}")
            if h.data_sources_used:
                parts.append(f"Data Sources Used: {', '.join(h.data_sources_used)}")
        
        if context.workspace_id:
            parts.append(f"Workspace: {context.workspace_id}")
        
        return "\n".join(parts) if parts else ""
    
    async def _summarize_conversation(self, context: InteractionContext) -> None:
        """Summarize conversation when it gets too long."""
        # Implementation would call LLM to summarize
        pass
```

#### Context-Aware Handoff Pattern

```python
from agent_framework import HandoffBuilder

def build_context_aware_workflow(
    orchestrator: ChatAgent,
    lakehouse_agent: ChatAgent,
    warehouse_agent: ChatAgent,
    realtime_agent: ChatAgent,
    context: InteractionContext,
) -> "AgentWorkflow":
    """Build workflow with context passing on handoffs."""
    
    # Create context-aware handoff callbacks
    def on_handoff(from_agent: str, to_agent: str, reason: str):
        """Callback executed on each handoff."""
        context.record_handoff(
            from_agent=from_agent,
            to_agent=to_agent,
            reason=reason,
            shared_data={
                "workspace_id": context.workspace_id,
                "summary": context.session.conversation_summary,
            }
        )
    
    def get_agent_context(agent_name: str) -> dict[str, Any]:
        """Get context for injecting into agent."""
        return context.get_context_for_agent(agent_name)
    
    # Build workflow with context hooks
    workflow = (
        HandoffBuilder(
            name="context_aware_fabric_orchestration",
            participants=[orchestrator, lakehouse_agent, warehouse_agent, realtime_agent]
        )
        .with_start_agent(orchestrator)
        .add_handoff(orchestrator, [lakehouse_agent, warehouse_agent, realtime_agent])
        .add_handoff(lakehouse_agent, [orchestrator])
        .add_handoff(warehouse_agent, [orchestrator])
        .add_handoff(realtime_agent, [orchestrator])
        .with_handoff_callback(on_handoff)
        .with_context_provider(get_agent_context)
        .build()
    )
    
    return workflow
```

### Context Storage Implementations

#### In-Memory Store (Development)

```python
from threading import Lock

class InMemoryContextStore:
    """Thread-safe in-memory context store for development."""
    
    def __init__(self):
        self._store: dict[str, InteractionContext] = {}
        self._lock = Lock()
    
    async def get_or_create(
        self,
        thread_id: str,
        user_id: str | None = None,
        workspace_id: str = ""
    ) -> InteractionContext:
        """Get existing context or create new one."""
        with self._lock:
            if thread_id not in self._store:
                self._store[thread_id] = InteractionContext.create(
                    thread_id=thread_id,
                    user_id=user_id,
                    workspace_id=workspace_id,
                )
            return self._store[thread_id]
    
    async def save(self, context: InteractionContext) -> None:
        """Save context to store."""
        with self._lock:
            context.updated_at = datetime.utcnow()
            self._store[context.session.thread_id] = context
    
    async def delete(self, thread_id: str) -> None:
        """Delete context from store."""
        with self._lock:
            self._store.pop(thread_id, None)
```

#### Azure Cosmos DB Store (Production)

```python
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential
import json

class CosmosDBContextStore:
    """Production context store using Azure Cosmos DB."""
    
    def __init__(
        self,
        endpoint: str,
        database_name: str,
        container_name: str
    ):
        self.endpoint = endpoint
        self.database_name = database_name
        self.container_name = container_name
        self._client: CosmosClient | None = None
        self._container = None
    
    async def _ensure_client(self):
        """Lazy initialization of Cosmos client."""
        if self._client is None:
            credential = DefaultAzureCredential()
            self._client = CosmosClient(self.endpoint, credential)
            database = self._client.get_database_client(self.database_name)
            self._container = database.get_container_client(self.container_name)
    
    async def get_or_create(
        self,
        thread_id: str,
        user_id: str | None = None,
        workspace_id: str = ""
    ) -> InteractionContext:
        """Get existing context or create new one."""
        await self._ensure_client()
        
        try:
            item = await self._container.read_item(
                item=thread_id,
                partition_key=thread_id
            )
            return self._deserialize(item)
        except Exception:
            # Create new context
            return InteractionContext.create(
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
            )
    
    async def save(self, context: InteractionContext) -> None:
        """Save context to Cosmos DB."""
        await self._ensure_client()
        
        item = self._serialize(context)
        item["id"] = context.session.thread_id
        item["_partitionKey"] = context.session.thread_id
        item["ttl"] = 86400 * 7  # 7 day TTL
        
        await self._container.upsert_item(item)
    
    def _serialize(self, context: InteractionContext) -> dict:
        """Serialize context to JSON-compatible dict."""
        # Implementation converts dataclasses to dicts
        pass
    
    def _deserialize(self, item: dict) -> InteractionContext:
        """Deserialize context from Cosmos DB item."""
        # Implementation reconstructs dataclasses from dicts
        pass
```

#### Azure Redis Cache Store (High-Performance)

```python
import redis.asyncio as redis
import json

class RedisContextStore:
    """High-performance context store using Azure Redis Cache."""
    
    def __init__(self, connection_string: str, ttl_seconds: int = 3600):
        self.connection_string = connection_string
        self.ttl_seconds = ttl_seconds
        self._client: redis.Redis | None = None
    
    async def _ensure_client(self):
        """Lazy initialization of Redis client."""
        if self._client is None:
            self._client = redis.from_url(self.connection_string)
    
    async def get_or_create(
        self,
        thread_id: str,
        user_id: str | None = None,
        workspace_id: str = ""
    ) -> InteractionContext:
        """Get existing context or create new one."""
        await self._ensure_client()
        
        key = f"context:{thread_id}"
        data = await self._client.get(key)
        
        if data:
            return self._deserialize(json.loads(data))
        
        return InteractionContext.create(
            thread_id=thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
    
    async def save(self, context: InteractionContext) -> None:
        """Save context with TTL."""
        await self._ensure_client()
        
        key = f"context:{context.session.thread_id}"
        data = json.dumps(self._serialize(context))
        
        await self._client.setex(key, self.ttl_seconds, data)
```

### Context Summarization

For long-running conversations, summarize context to stay within token limits:

```python
async def summarize_context(
    context: InteractionContext,
    chat_client: AzureOpenAIChatClient,
    messages: list[str]
) -> str:
    """Generate a summary of the conversation for context compression."""
    
    prompt = f"""Summarize the following conversation, preserving:
1. Key user intents and goals
2. Important data sources and tables referenced
3. Query results and key findings
4. Any errors encountered and resolutions
5. Current state of the analysis

Conversation:
{chr(10).join(messages[-20:])}

Provide a concise summary (max 500 words) that captures essential context."""
    
    response = await chat_client.generate(prompt)
    
    # Update context with summary
    context.session.conversation_summary = response
    context.session.message_count = 0  # Reset counter
    
    return response
```

### Multi-User Thread Isolation

```python
async def handle_user_request(
    user_id: str,
    thread_id: str,
    message: str,
    orchestrator: ContextAwareOrchestrator,
    context_store: "ContextStore"
) -> str:
    """Handle a user request with proper context isolation.
    
    Args:
        user_id: The user's identifier.
        thread_id: The conversation thread ID.
        message: User's message.
        orchestrator: The context-aware orchestrator.
        context_store: Storage for interaction contexts.
        
    Returns:
        Agent response.
    """
    # Load or create context for this thread
    context = await context_store.get_or_create(
        thread_id=thread_id,
        user_id=user_id,
        workspace_id=os.environ.get("FABRIC_WORKSPACE_ID", ""),
    )
    
    # Process with context
    response = await orchestrator.process_with_context(
        thread_id=thread_id,
        user_message=message,
    )
    
    return response
```

---

## Deployment Architecture

### Azure Resources

```bicep
// Core infrastructure for multi-fabric agent orchestration

resource aiServices 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: 'fabric-agent-ai'
  location: location
  kind: 'AIServices'
  sku: { name: 'S0' }
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: 'fabric-agent-ai'
  }
}

resource gpt4oDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: aiServices
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
```

### Environment Configuration

```env
# Microsoft Foundry Configuration
AZURE_AI_SERVICES_ENDPOINT=https://fabric-agent-ai.services.ai.azure.com
AZURE_AI_PROJECT_ENDPOINT=https://fabric-agent-ai.services.ai.azure.com/api/projects/fabric-orchestration

# Azure OpenAI Configuration (for agents)
AZURE_OPENAI_ENDPOINT=https://fabric-agent-ai.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Microsoft Fabric Configuration
FABRIC_WORKSPACE_ID=<workspace-guid>
FABRIC_TENANT_ID=<tenant-guid>
```

---

## Implementation Guide

### Step 1: Install Microsoft Agent Framework

```bash
# Python (recommended) - Note: --pre flag required during preview
pip install agent-framework-azure-ai --pre
```

```bash
# .NET - Note: --prerelease flag required during preview
dotnet add package Microsoft.Agents.AI.AzureAI --prerelease
dotnet add package Microsoft.Agents.AI.Workflows --prerelease
```

### Step 2: Initialize Foundry Client

```python
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential

# Initialize chat client connected to Foundry
client = AzureOpenAIChatClient(
    credential=DefaultAzureCredential(),
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"]
)
```

### Step 3: Create Fabric Data Agents

```python
# Create specialized agents for each Fabric workload
orchestrator = create_orchestrator_agent(client)
lakehouse_agent = create_lakehouse_agent(client)
warehouse_agent = create_warehouse_agent(client)
realtime_agent = create_realtime_agent(client)
```

### Step 4: Build Orchestration Workflow

```python
from agent_framework import HandoffBuilder

workflow = (
    HandoffBuilder(
        name="fabric_multi_agent_orchestration",
        participants=[orchestrator, lakehouse_agent, warehouse_agent, realtime_agent]
    )
    .with_start_agent(orchestrator)
    .add_handoff(orchestrator, [lakehouse_agent, warehouse_agent, realtime_agent])
    .add_handoff(lakehouse_agent, [orchestrator, warehouse_agent])
    .add_handoff(warehouse_agent, [orchestrator, lakehouse_agent])
    .add_handoff(realtime_agent, [orchestrator])
    .build()
)
```

### Step 5: Run the Workflow

```python
import asyncio

async def main():
    user_query = "Show me the top 10 customers by revenue from the lakehouse, then create a warehouse view for reporting"
    
    async for event in workflow.run_stream(user_query):
        if hasattr(event, 'text'):
            print(event.text)

asyncio.run(main())
```

---

## Monitoring & Observability

### Telemetry Integration

```python
from opentelemetry import trace
from agent_framework import WorkflowEvent

tracer = trace.get_tracer("fabric-agent-orchestration")

async for event in workflow.run_stream(query):
    with tracer.start_as_current_span(f"agent_event_{type(event).__name__}"):
        # Process event
        if isinstance(event, HandoffSentEvent):
            span.set_attribute("source_agent", event.source)
            span.set_attribute("target_agent", event.target)
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `agent.handoff.count` | Number of handoffs between agents |
| `agent.response.latency` | Time to receive agent response |
| `fabric.query.duration` | Time to execute Fabric query |
| `workflow.completion.rate` | Percentage of workflows completed successfully |

---

## Error Handling & Resilience

```python
from agent_framework import WorkflowStatusEvent, WorkflowRunState

async for event in workflow.run_stream(query):
    if isinstance(event, WorkflowStatusEvent):
        if event.state == WorkflowRunState.FAILED:
            # Handle workflow failure
            logger.error(f"Workflow failed: {event.error}")
            # Implement retry or fallback logic
```

### Retry Strategy

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def execute_with_retry(workflow, query):
    return await workflow.run(query)
```

---

## Summary

This architecture provides a scalable, maintainable approach to orchestrating multiple Microsoft Fabric data agents through:

1. **Microsoft Foundry** as the AI hosting platform
2. **Azure AI Agent Service** for managed agent runtime
3. **Microsoft Agent Framework** for multi-agent orchestration patterns
4. **Specialized Fabric agents** for each data workload

The handoff orchestration pattern is recommended for most scenarios, providing intelligent routing while maintaining conversation context across agent transitions.

---

## References

- [Microsoft Agent Framework Documentation](https://github.com/microsoft/agent-framework)
- [Microsoft Foundry Documentation](https://learn.microsoft.com/azure/ai-studio/)
- [Microsoft Fabric REST APIs](https://learn.microsoft.com/rest/api/fabric/)
- [Azure AI Agent Service](https://learn.microsoft.com/azure/ai-services/agents/)
