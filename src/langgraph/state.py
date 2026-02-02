# Copyright (c) Microsoft. All rights reserved.
"""State definitions for LangGraph Fabric agent orchestration."""

from typing import TypedDict, Annotated, Literal, Sequence, Any
from operator import add

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class FabricAgentState(TypedDict):
    """State schema for multi-fabric agent orchestration.
    
    This state is passed between nodes in the LangGraph workflow and
    maintains the conversation context and routing information.
    """
    
    # Conversation messages with reducer for appending
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Current agent handling the request
    current_agent: Literal[
        "orchestrator", 
        "lakehouse", 
        "warehouse", 
        "realtime", 
        "pipeline", 
        "powerbi"
    ] | None
    
    # Workspace context from configuration
    workspace_id: str
    
    # Query metadata
    query_type: Literal["spark", "tsql", "kql", "dax", "pipeline"] | None
    
    # Results from agent execution
    query_results: dict[str, Any] | None
    
    # Error state
    error: str | None
    
    # Routing decision from orchestrator
    next_agent: Literal[
        "lakehouse", 
        "warehouse", 
        "realtime", 
        "pipeline", 
        "powerbi",
        "end"
    ] | None
    
    # Human-in-the-loop flag
    requires_approval: bool


class MultiSourceState(FabricAgentState):
    """Extended state for concurrent multi-source queries.
    
    Adds support for parallel execution results and aggregation.
    """
    
    # Parallel execution results (uses add reducer for merging)
    parallel_results: Annotated[list[dict[str, Any]], add]
    
    # Agents to execute in parallel
    parallel_agents: list[str]
    
    # Aggregated final result
    aggregated_result: dict[str, Any] | None


class ConversationState(TypedDict):
    """Simplified state for basic conversation tracking."""
    
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_agent: str | None


def create_initial_state(
    workspace_id: str,
    query: str | None = None
) -> FabricAgentState:
    """Create initial state for a new workflow execution.
    
    Args:
        workspace_id: The Fabric workspace GUID.
        query: Optional initial query (will be added as HumanMessage).
        
    Returns:
        Initialized FabricAgentState.
    """
    from langchain_core.messages import HumanMessage
    
    messages = []
    if query:
        messages.append(HumanMessage(content=query))
    
    return FabricAgentState(
        messages=messages,
        current_agent=None,
        workspace_id=workspace_id,
        query_type=None,
        query_results=None,
        error=None,
        next_agent=None,
        requires_approval=False,
    )


def create_multi_source_state(
    workspace_id: str,
    parallel_agents: list[str],
    query: str | None = None
) -> MultiSourceState:
    """Create initial state for parallel execution.
    
    Args:
        workspace_id: The Fabric workspace GUID.
        parallel_agents: List of agents to run in parallel.
        query: Optional initial query.
        
    Returns:
        Initialized MultiSourceState.
    """
    from langchain_core.messages import HumanMessage
    
    messages = []
    if query:
        messages.append(HumanMessage(content=query))
    
    return MultiSourceState(
        messages=messages,
        current_agent=None,
        workspace_id=workspace_id,
        query_type=None,
        query_results=None,
        error=None,
        next_agent=None,
        requires_approval=False,
        parallel_results=[],
        parallel_agents=parallel_agents,
        aggregated_result=None,
    )
