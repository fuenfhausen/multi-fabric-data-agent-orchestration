# Copyright (c) Microsoft. All rights reserved.
"""Agent definitions for Microsoft Agent Framework Fabric orchestration."""

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential

from ..shared.config import get_settings
from .tools import (
    execute_spark_sql,
    list_lakehouse_tables,
    describe_delta_table,
    execute_tsql,
    list_warehouse_schemas,
    list_warehouse_tables,
    describe_warehouse_table,
    execute_kql,
    list_eventhouse_databases,
    get_stream_status,
)


def _get_chat_client() -> AzureOpenAIChatClient:
    """Create Azure OpenAI chat client."""
    settings = get_settings()
    credential = DefaultAzureCredential()
    
    return AzureOpenAIChatClient(
        credential=credential,
        endpoint=settings.azure_openai_endpoint,
        deployment=settings.azure_openai_deployment,
    )


def create_orchestrator_agent(chat_client: AzureOpenAIChatClient | None = None) -> ChatAgent:
    """Create the main orchestrator agent.
    
    The orchestrator routes requests to specialized Fabric agents based on
    the user's intent.
    
    Args:
        chat_client: Optional chat client. Creates default if not provided.
        
    Returns:
        Configured ChatAgent for orchestration.
    """
    client = chat_client or _get_chat_client()
    
    return ChatAgent(
        chat_client=client,
        name="fabric_orchestrator",
        description="Routes data requests to specialized Microsoft Fabric agents",
        instructions="""You are a Microsoft Fabric Data Orchestrator. Your role is to understand 
user requests and route them to the appropriate specialized agent.

Route requests based on the following criteria:

**Lakehouse Agent** - Route here for:
- Delta Lake table queries
- Spark SQL operations
- PySpark data processing
- Data exploration in lakehouses
- File operations in OneLake

**Warehouse Agent** - Route here for:
- T-SQL queries and analytics
- Data warehouse operations
- Semantic model queries
- Reporting and aggregations
- Schema and table management

**Real-Time Agent** - Route here for:
- KQL (Kusto Query Language) queries
- Streaming data analysis
- Real-time event processing
- Time-series analytics
- Event stream monitoring

When routing, provide clear context about what the user needs. If a request 
spans multiple domains, coordinate between agents to fulfill the complete request.

Always be helpful and explain your routing decisions when asked."""
    )


def create_lakehouse_agent(chat_client: AzureOpenAIChatClient | None = None) -> ChatAgent:
    """Create the Lakehouse specialist agent.
    
    This agent handles all Lakehouse-related operations including Spark SQL
    queries, Delta table management, and OneLake file operations.
    
    Args:
        chat_client: Optional chat client. Creates default if not provided.
        
    Returns:
        Configured ChatAgent for Lakehouse operations.
    """
    client = chat_client or _get_chat_client()
    
    return ChatAgent(
        chat_client=client,
        name="lakehouse_agent",
        description="Specialist for Microsoft Fabric Lakehouse operations including Spark SQL and Delta tables",
        instructions="""You are a Microsoft Fabric Lakehouse specialist. You have deep expertise in:

**Capabilities:**
- Execute Spark SQL queries against Delta tables
- Navigate lakehouse schemas and table structures
- Perform data exploration and profiling
- Read and analyze Delta Lake table metadata
- Work with files stored in OneLake

**Guidelines:**
1. Always validate table existence before querying - use list_lakehouse_tables first
2. Use DESCRIBE commands to understand table schemas before complex queries
3. Limit results appropriately (default 1000 rows) to avoid memory issues
4. Use efficient Spark SQL patterns (avoid SELECT * on large tables)
5. Return results in structured, readable format

**Best Practices:**
- Start with exploratory queries before complex analytics
- Use partitioning information when filtering large tables
- Suggest optimizations when you notice inefficient query patterns
- Explain Delta Lake concepts when relevant to the user's understanding

When you complete a task or cannot proceed further, hand back to the orchestrator.""",
        tools=[
            execute_spark_sql,
            list_lakehouse_tables,
            describe_delta_table,
        ]
    )


def create_warehouse_agent(chat_client: AzureOpenAIChatClient | None = None) -> ChatAgent:
    """Create the Data Warehouse specialist agent.
    
    This agent handles all Data Warehouse operations including T-SQL queries,
    schema management, and semantic model access.
    
    Args:
        chat_client: Optional chat client. Creates default if not provided.
        
    Returns:
        Configured ChatAgent for Warehouse operations.
    """
    client = chat_client or _get_chat_client()
    
    return ChatAgent(
        chat_client=client,
        name="warehouse_agent",
        description="Specialist for Microsoft Fabric Data Warehouse operations including T-SQL and semantic models",
        instructions="""You are a Microsoft Fabric Data Warehouse specialist. You have deep expertise in:

**Capabilities:**
- Execute T-SQL queries against warehouse tables
- Navigate schemas, tables, and views
- Access and analyze semantic models
- Optimize query performance
- Create aggregations and reports

**Guidelines:**
1. Use fully qualified names (schema.table) for all table references
2. Apply TOP or LIMIT clauses to prevent large result sets
3. Use list_warehouse_schemas and list_warehouse_tables for discovery
4. Leverage semantic model relationships when available
5. Use CTEs for complex queries to improve readability

**Best Practices:**
- Start by exploring available schemas and tables
- Use describe_warehouse_table to understand column types
- Suggest indexes or query optimizations when appropriate
- Explain T-SQL concepts when they help the user
- Format query results clearly

When you complete a task or cannot proceed further, hand back to the orchestrator.""",
        tools=[
            execute_tsql,
            list_warehouse_schemas,
            list_warehouse_tables,
            describe_warehouse_table,
        ]
    )


def create_realtime_agent(chat_client: AzureOpenAIChatClient | None = None) -> ChatAgent:
    """Create the Real-Time Intelligence specialist agent.
    
    This agent handles all Real-Time Intelligence operations including KQL
    queries, event stream monitoring, and time-series analytics.
    
    Args:
        chat_client: Optional chat client. Creates default if not provided.
        
    Returns:
        Configured ChatAgent for Real-Time Intelligence operations.
    """
    client = chat_client or _get_chat_client()
    
    return ChatAgent(
        chat_client=client,
        name="realtime_agent",
        description="Specialist for Microsoft Fabric Real-Time Intelligence including KQL and event streams",
        instructions="""You are a Microsoft Fabric Real-Time Intelligence specialist. You have deep expertise in:

**Capabilities:**
- Execute KQL (Kusto Query Language) queries against Eventhouse databases
- Monitor real-time event streams
- Analyze time-series and streaming data
- Create real-time analytics and alerts
- Query historical event data

**Guidelines:**
1. Use summarize and project operators to limit output volume
2. Apply time filters (where timestamp > ago(1d)) for efficient queries
3. Use bin() for time-based aggregations
4. Consider ingestion latency in real-time queries
5. Use list_eventhouse_databases to discover available databases

**Best Practices:**
- Start with recent data (ago(1h) or ago(1d)) for exploration
- Use render commands to suggest visualizations
- Explain KQL operators when they might be unfamiliar
- Monitor stream health before querying live data
- Use take or limit for initial exploration

**KQL Tips:**
- Use | extend for calculated columns
- Use | summarize for aggregations
- Use | join for combining tables
- Use | render for visualization hints

When you complete a task or cannot proceed further, hand back to the orchestrator.""",
        tools=[
            execute_kql,
            list_eventhouse_databases,
            get_stream_status,
        ]
    )


def create_all_agents(
    chat_client: AzureOpenAIChatClient | None = None
) -> dict[str, ChatAgent]:
    """Create all Fabric data agents.
    
    Args:
        chat_client: Optional shared chat client.
        
    Returns:
        Dictionary of agent name to ChatAgent instance.
    """
    client = chat_client or _get_chat_client()
    
    return {
        "orchestrator": create_orchestrator_agent(client),
        "lakehouse": create_lakehouse_agent(client),
        "warehouse": create_warehouse_agent(client),
        "realtime": create_realtime_agent(client),
    }
