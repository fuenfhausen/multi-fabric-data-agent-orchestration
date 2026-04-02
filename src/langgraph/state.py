# Copyright (c) Microsoft. All rights reserved.
"""State definitions for LangGraph Fabric agent orchestration."""

from typing import TypedDict, Annotated, Literal, Sequence, Any
from operator import add
from datetime import datetime

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# =============================================================================
# Context Management Types
# =============================================================================

class SessionContext(TypedDict):
    """Session-scoped context for multi-turn conversations.
    
    Tracks information scoped to a user session or conversation thread.
    """
    
    # Unique session/thread identifier
    thread_id: str
    
    # User identity and preferences
    user_id: str | None
    user_preferences: dict[str, Any]
    
    # Session timing
    session_start: str  # ISO format datetime
    last_activity: str  # ISO format datetime
    
    # Conversation summary for long sessions
    conversation_summary: str | None
    
    # Token budget tracking
    token_count: int
    max_context_tokens: int


class AgentContext(TypedDict):
    """Agent-specific execution context.
    
    Maintains state specific to each specialized agent during execution.
    """
    
    # Agent identification
    agent_name: str
    agent_type: Literal["lakehouse", "warehouse", "realtime", "pipeline", "powerbi"]
    
    # Execution history
    tool_calls: list[dict[str, Any]]
    intermediate_results: list[dict[str, Any]]
    
    # Agent-specific memory
    discovered_schemas: dict[str, Any]
    query_history: list[str]
    
    # Performance tracking
    execution_time_ms: int
    retry_count: int


class CrossAgentContext(TypedDict):
    """Context shared across agents during orchestration.
    
    Enables context sharing between agents during handoffs and parallel execution.
    """
    
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


class ContextMetadata(TypedDict):
    """Metadata for context lifecycle management."""
    
    created_at: str  # ISO format datetime
    updated_at: str  # ISO format datetime
    version: int
    checksum: str | None


class InteractionContext(TypedDict):
    """Complete context for agent interactions.
    
    Combines session, agent, and cross-agent contexts into a unified
    structure for comprehensive context management.
    """
    
    # Session-level context
    session: SessionContext
    
    # Per-agent context (keyed by agent name)
    agents: dict[str, AgentContext]
    
    # Cross-agent shared context
    shared: CrossAgentContext
    
    # Context metadata
    metadata: ContextMetadata


# =============================================================================
# Core State Types
# =============================================================================

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


# =============================================================================
# Enhanced State with Context Management
# =============================================================================

class EnhancedFabricAgentState(TypedDict):
    """Enhanced state schema with comprehensive context management.
    
    Extends the base FabricAgentState with full interaction context
    for managing multi-turn conversations, agent handoffs, and
    session persistence.
    """
    
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
    
    # Interaction Context (NEW)
    interaction_context: InteractionContext


# =============================================================================
# Context Factory Functions
# =============================================================================

def create_session_context(
    thread_id: str,
    user_id: str | None = None,
    max_context_tokens: int = 128000
) -> SessionContext:
    """Create a new session context.
    
    Args:
        thread_id: Unique thread/session identifier.
        user_id: Optional user identifier for preference tracking.
        max_context_tokens: Maximum tokens allowed in context window.
        
    Returns:
        Initialized SessionContext.
    """
    now = datetime.utcnow().isoformat()
    
    return SessionContext(
        thread_id=thread_id,
        user_id=user_id,
        user_preferences={},
        session_start=now,
        last_activity=now,
        conversation_summary=None,
        token_count=0,
        max_context_tokens=max_context_tokens,
    )


def create_agent_context(
    agent_name: str,
    agent_type: Literal["lakehouse", "warehouse", "realtime", "pipeline", "powerbi"]
) -> AgentContext:
    """Create a new agent context.
    
    Args:
        agent_name: Name of the agent.
        agent_type: Type of the agent.
        
    Returns:
        Initialized AgentContext.
    """
    return AgentContext(
        agent_name=agent_name,
        agent_type=agent_type,
        tool_calls=[],
        intermediate_results=[],
        discovered_schemas={},
        query_history=[],
        execution_time_ms=0,
        retry_count=0,
    )


def create_cross_agent_context() -> CrossAgentContext:
    """Create a new cross-agent context.
    
    Returns:
        Initialized CrossAgentContext.
    """
    return CrossAgentContext(
        shared_results={},
        handoff_from=None,
        handoff_reason=None,
        handoff_context={},
        parallel_execution_id=None,
        pending_agents=[],
        completed_agents=[],
        data_sources_used=[],
        transformations_applied=[],
    )


def create_context_metadata() -> ContextMetadata:
    """Create new context metadata.
    
    Returns:
        Initialized ContextMetadata.
    """
    now = datetime.utcnow().isoformat()
    
    return ContextMetadata(
        created_at=now,
        updated_at=now,
        version=1,
        checksum=None,
    )


def create_interaction_context(
    thread_id: str,
    user_id: str | None = None,
    max_context_tokens: int = 128000
) -> InteractionContext:
    """Create a complete interaction context.
    
    Args:
        thread_id: Unique thread/session identifier.
        user_id: Optional user identifier.
        max_context_tokens: Maximum tokens allowed in context window.
        
    Returns:
        Initialized InteractionContext.
    """
    return InteractionContext(
        session=create_session_context(thread_id, user_id, max_context_tokens),
        agents={},
        shared=create_cross_agent_context(),
        metadata=create_context_metadata(),
    )


def create_enhanced_initial_state(
    workspace_id: str,
    thread_id: str,
    query: str | None = None,
    user_id: str | None = None,
    max_context_tokens: int = 128000
) -> EnhancedFabricAgentState:
    """Create initial state with full context management.
    
    Args:
        workspace_id: The Fabric workspace GUID.
        thread_id: Unique thread/session identifier.
        query: Optional initial query (will be added as HumanMessage).
        user_id: Optional user identifier.
        max_context_tokens: Maximum tokens allowed in context window.
        
    Returns:
        Initialized EnhancedFabricAgentState.
    """
    from langchain_core.messages import HumanMessage
    
    messages = []
    if query:
        messages.append(HumanMessage(content=query))
    
    return EnhancedFabricAgentState(
        messages=messages,
        current_agent=None,
        workspace_id=workspace_id,
        query_type=None,
        query_results=None,
        error=None,
        next_agent=None,
        requires_approval=False,
        interaction_context=create_interaction_context(
            thread_id=thread_id,
            user_id=user_id,
            max_context_tokens=max_context_tokens,
        ),
    )


# =============================================================================
# Context Update Helpers
# =============================================================================

def update_session_activity(context: InteractionContext) -> InteractionContext:
    """Update session last activity timestamp.
    
    Args:
        context: Current interaction context.
        
    Returns:
        Updated InteractionContext.
    """
    now = datetime.utcnow().isoformat()
    
    updated_session = SessionContext(
        **context["session"],
        last_activity=now,
    )
    
    updated_metadata = ContextMetadata(
        **context["metadata"],
        updated_at=now,
        version=context["metadata"]["version"] + 1,
    )
    
    return InteractionContext(
        session=updated_session,
        agents=context["agents"],
        shared=context["shared"],
        metadata=updated_metadata,
    )


def record_handoff(
    context: InteractionContext,
    from_agent: str,
    to_agent: str,
    reason: str,
    handoff_data: dict[str, Any] | None = None
) -> InteractionContext:
    """Record an agent handoff in the context.
    
    Args:
        context: Current interaction context.
        from_agent: Name of the source agent.
        to_agent: Name of the target agent.
        reason: Explanation for the handoff.
        handoff_data: Optional data to pass to the target agent.
        
    Returns:
        Updated InteractionContext with handoff recorded.
    """
    now = datetime.utcnow().isoformat()
    
    updated_shared = CrossAgentContext(
        **context["shared"],
        handoff_from=from_agent,
        handoff_reason=reason,
        handoff_context=handoff_data or {},
    )
    
    updated_metadata = ContextMetadata(
        **context["metadata"],
        updated_at=now,
        version=context["metadata"]["version"] + 1,
    )
    
    return InteractionContext(
        session=context["session"],
        agents=context["agents"],
        shared=updated_shared,
        metadata=updated_metadata,
    )


def get_or_create_agent_context(
    context: InteractionContext,
    agent_name: str,
    agent_type: Literal["lakehouse", "warehouse", "realtime", "pipeline", "powerbi"]
) -> tuple[InteractionContext, AgentContext]:
    """Get existing agent context or create a new one.
    
    Args:
        context: Current interaction context.
        agent_name: Name of the agent.
        agent_type: Type of the agent.
        
    Returns:
        Tuple of (updated context, agent context).
    """
    agents = dict(context["agents"])
    
    if agent_name not in agents:
        agents[agent_name] = create_agent_context(agent_name, agent_type)
        
        now = datetime.utcnow().isoformat()
        updated_metadata = ContextMetadata(
            **context["metadata"],
            updated_at=now,
            version=context["metadata"]["version"] + 1,
        )
        
        context = InteractionContext(
            session=context["session"],
            agents=agents,
            shared=context["shared"],
            metadata=updated_metadata,
        )
    
    return context, agents[agent_name]


def add_data_source(
    context: InteractionContext,
    source: str
) -> InteractionContext:
    """Add a data source to the lineage tracking.
    
    Args:
        context: Current interaction context.
        source: Data source identifier (e.g., "lakehouse.sales_data").
        
    Returns:
        Updated InteractionContext.
    """
    sources = list(context["shared"]["data_sources_used"])
    if source not in sources:
        sources.append(source)
    
    now = datetime.utcnow().isoformat()
    
    updated_shared = CrossAgentContext(
        **context["shared"],
        data_sources_used=sources,
    )
    
    updated_metadata = ContextMetadata(
        **context["metadata"],
        updated_at=now,
        version=context["metadata"]["version"] + 1,
    )
    
    return InteractionContext(
        session=context["session"],
        agents=context["agents"],
        shared=updated_shared,
        metadata=updated_metadata,
    )
