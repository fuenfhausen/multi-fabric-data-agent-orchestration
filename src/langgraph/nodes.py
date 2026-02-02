# Copyright (c) Microsoft. All rights reserved.
"""Node definitions for LangGraph Fabric agent orchestration."""

from typing import Literal, Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

from .state import FabricAgentState, MultiSourceState
from .tools import LAKEHOUSE_TOOLS, WAREHOUSE_TOOLS, REALTIME_TOOLS


def get_llm() -> AzureChatOpenAI:
    """Create Azure OpenAI chat model from settings."""
    import os
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from ..shared.config import get_settings
    
    settings = get_settings()
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default"
    )
    
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        azure_ad_token_provider=token_provider,
        temperature=0,
    )


# =============================================================================
# Orchestrator Node
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are a Microsoft Fabric Data Orchestrator. Your role is to analyze 
user requests and route them to the appropriate specialized agent.

Analyze the user's request and respond with ONLY one of these agent names:
- "lakehouse": For Delta Lake, Spark SQL, PySpark, and lakehouse operations
- "warehouse": For T-SQL queries, data warehouse analytics, semantic models
- "realtime": For KQL queries, streaming data, real-time analytics
- "end": If the request is complete, unclear, or cannot be handled

Decision criteria:
- Mention of Delta tables, Spark, PySpark, lakehouse → "lakehouse"
- Mention of T-SQL, warehouse, SQL queries, semantic models → "warehouse"  
- Mention of KQL, streaming, real-time, events, Kusto → "realtime"
- General questions, greetings, or unclear requests → "end"

Respond with ONLY the agent name, nothing else."""


def orchestrator_node(state: FabricAgentState) -> dict[str, Any]:
    """Supervisor node that routes to appropriate Fabric agent.
    
    Analyzes the user's request and determines which specialized agent
    should handle it.
    
    Args:
        state: Current workflow state.
        
    Returns:
        State update with next_agent and current_agent.
    """
    llm = get_llm()
    
    messages = [SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT)] + list(state["messages"])
    response = llm.invoke(messages)
    
    next_agent = response.content.strip().lower()
    
    # Validate the response
    valid_agents = ["lakehouse", "warehouse", "realtime", "end"]
    if next_agent not in valid_agents:
        # Default to end if response is invalid
        next_agent = "end"
    
    return {
        "next_agent": next_agent,
        "current_agent": "orchestrator"
    }


# =============================================================================
# Lakehouse Node
# =============================================================================

LAKEHOUSE_SYSTEM_PROMPT = """You are a Microsoft Fabric Lakehouse specialist.

Your capabilities:
- Execute Spark SQL queries against Delta tables
- Navigate lakehouse schemas and table structures
- Perform data exploration and profiling
- Read and analyze Delta Lake table metadata

Guidelines:
1. Always validate table existence before querying - use list_lakehouse_tables first
2. Use DESCRIBE commands to understand table schemas before complex queries
3. Limit results appropriately (default 1000 rows) to avoid memory issues
4. Use efficient Spark SQL patterns (avoid SELECT * on large tables)
5. Return results in structured, readable format

When you complete your task, provide a clear summary of results."""


def lakehouse_node(state: FabricAgentState) -> dict[str, Any]:
    """Lakehouse agent node using ReAct pattern.
    
    Handles all Lakehouse-related operations including Spark SQL
    queries and Delta table management.
    
    Args:
        state: Current workflow state.
        
    Returns:
        State update with messages and metadata.
    """
    llm = get_llm()
    
    # Create ReAct agent with lakehouse tools
    agent = create_react_agent(
        llm,
        LAKEHOUSE_TOOLS,
        state_modifier=LAKEHOUSE_SYSTEM_PROMPT
    )
    
    try:
        result = agent.invoke({
            "messages": list(state["messages"])
        })
        
        return {
            "messages": result["messages"],
            "current_agent": "lakehouse",
            "query_type": "spark",
            "error": None
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Lakehouse agent error: {str(e)}")],
            "current_agent": "lakehouse",
            "error": str(e)
        }


# =============================================================================
# Warehouse Node
# =============================================================================

WAREHOUSE_SYSTEM_PROMPT = """You are a Microsoft Fabric Data Warehouse specialist.

Your capabilities:
- Execute T-SQL queries against warehouse tables
- Navigate schemas, tables, and views
- Access and analyze semantic models
- Optimize query performance

Guidelines:
1. Use fully qualified names (schema.table) for all table references
2. Apply TOP or LIMIT clauses to prevent large result sets
3. Use list_warehouse_schemas and list_warehouse_tables for discovery
4. Leverage semantic model relationships when available
5. Use CTEs for complex queries to improve readability

When you complete your task, provide a clear summary of results."""


def warehouse_node(state: FabricAgentState) -> dict[str, Any]:
    """Warehouse agent node using ReAct pattern.
    
    Handles all Data Warehouse operations including T-SQL queries
    and semantic model access.
    
    Args:
        state: Current workflow state.
        
    Returns:
        State update with messages and metadata.
    """
    llm = get_llm()
    
    # Create ReAct agent with warehouse tools
    agent = create_react_agent(
        llm,
        WAREHOUSE_TOOLS,
        state_modifier=WAREHOUSE_SYSTEM_PROMPT
    )
    
    try:
        result = agent.invoke({
            "messages": list(state["messages"])
        })
        
        return {
            "messages": result["messages"],
            "current_agent": "warehouse",
            "query_type": "tsql",
            "error": None
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Warehouse agent error: {str(e)}")],
            "current_agent": "warehouse",
            "error": str(e)
        }


# =============================================================================
# Real-Time Intelligence Node
# =============================================================================

REALTIME_SYSTEM_PROMPT = """You are a Microsoft Fabric Real-Time Intelligence specialist.

Your capabilities:
- Execute KQL (Kusto Query Language) queries against Eventhouse databases
- Monitor real-time event streams
- Analyze time-series and streaming data
- Create real-time analytics and alerts

Guidelines:
1. Use summarize and project operators to limit output volume
2. Apply time filters (where timestamp > ago(1d)) for efficient queries
3. Use bin() for time-based aggregations
4. Consider ingestion latency in real-time queries
5. Use list_eventhouse_databases to discover available databases

KQL Tips:
- Use | extend for calculated columns
- Use | summarize for aggregations
- Use | join for combining tables
- Use | render for visualization hints

When you complete your task, provide a clear summary of results."""


def realtime_node(state: FabricAgentState) -> dict[str, Any]:
    """Real-Time Intelligence agent node using ReAct pattern.
    
    Handles all Real-Time Intelligence operations including KQL
    queries and event stream monitoring.
    
    Args:
        state: Current workflow state.
        
    Returns:
        State update with messages and metadata.
    """
    llm = get_llm()
    
    # Create ReAct agent with real-time tools
    agent = create_react_agent(
        llm,
        REALTIME_TOOLS,
        state_modifier=REALTIME_SYSTEM_PROMPT
    )
    
    try:
        result = agent.invoke({
            "messages": list(state["messages"])
        })
        
        return {
            "messages": result["messages"],
            "current_agent": "realtime",
            "query_type": "kql",
            "error": None
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Real-Time agent error: {str(e)}")],
            "current_agent": "realtime",
            "error": str(e)
        }


# =============================================================================
# Aggregator Node (for parallel workflows)
# =============================================================================

def aggregator_node(state: MultiSourceState) -> dict[str, Any]:
    """Aggregate results from parallel agent executions.
    
    Combines results from multiple Fabric agents into a unified response.
    
    Args:
        state: Current workflow state with parallel results.
        
    Returns:
        State update with aggregated result.
    """
    results = state.get("parallel_results", [])
    
    aggregated = {
        "sources": len(results),
        "data": results,
        "summary": f"Aggregated results from {len(results)} Fabric sources"
    }
    
    # Create summary message
    summary_parts = [f"## Aggregated Results from {len(results)} Sources\n"]
    
    for i, result in enumerate(results):
        source = result.get("source", f"Source {i+1}")
        summary_parts.append(f"### {source}")
        if "data" in result:
            summary_parts.append(f"- Records: {len(result.get('data', []))}")
        if "error" in result:
            summary_parts.append(f"- Error: {result['error']}")
        summary_parts.append("")
    
    return {
        "messages": [AIMessage(content="\n".join(summary_parts))],
        "aggregated_result": aggregated,
        "current_agent": "aggregator"
    }


# =============================================================================
# Routing Functions
# =============================================================================

def route_to_agent(
    state: FabricAgentState
) -> Literal["lakehouse", "warehouse", "realtime", "end"]:
    """Conditional edge function for routing from orchestrator.
    
    Args:
        state: Current workflow state.
        
    Returns:
        Next node to route to.
    """
    next_agent = state.get("next_agent", "end")
    
    if next_agent in ["lakehouse", "warehouse", "realtime"]:
        return next_agent
    
    return "end"


def route_on_error(
    state: FabricAgentState
) -> Literal["retry", "orchestrator", "end"]:
    """Route based on error state.
    
    Args:
        state: Current workflow state.
        
    Returns:
        Next node based on error condition.
    """
    error = state.get("error")
    
    if not error:
        return "end"
    
    error_lower = error.lower()
    
    if "timeout" in error_lower or "retry" in error_lower:
        return "retry"
    elif "permission" in error_lower or "unauthorized" in error_lower:
        return "orchestrator"
    
    return "end"


def should_continue(
    state: FabricAgentState
) -> Literal["continue", "end"]:
    """Determine if workflow should continue or end.
    
    Args:
        state: Current workflow state.
        
    Returns:
        'continue' or 'end'.
    """
    # Check for errors
    if state.get("error"):
        return "end"
    
    # Check if we have results
    if state.get("query_results"):
        return "end"
    
    # Check message count (prevent infinite loops)
    if len(state.get("messages", [])) > 20:
        return "end"
    
    return "continue"
