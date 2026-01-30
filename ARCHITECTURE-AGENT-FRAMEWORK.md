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
