# Copyright (c) Microsoft. All rights reserved.
"""Context management for Microsoft Agent Framework Fabric orchestration.

This module provides context lifecycle management, persistence, and
coordination capabilities for multi-turn conversations and cross-agent
handoffs using the Microsoft Agent Framework.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Protocol, Literal
from uuid import uuid4
import json


# =============================================================================
# Context Data Classes
# =============================================================================

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
    
    # Message tracking
    message_count: int = 0
    max_messages_before_summary: int = 20
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "user_preferences": self.user_preferences,
            "session_start": self.session_start.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "conversation_summary": self.conversation_summary,
            "message_count": self.message_count,
            "max_messages_before_summary": self.max_messages_before_summary,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionContext":
        """Create from dictionary."""
        return cls(
            thread_id=data["thread_id"],
            user_id=data.get("user_id"),
            user_preferences=data.get("user_preferences", {}),
            session_start=datetime.fromisoformat(data["session_start"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            conversation_summary=data.get("conversation_summary"),
            message_count=data.get("message_count", 0),
            max_messages_before_summary=data.get("max_messages_before_summary", 20),
        )


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
    
    def add_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any
    ) -> None:
        """Record a tool invocation."""
        self.tool_calls.append({
            "tool": tool_name,
            "args": args,
            "result": str(result)[:1000],  # Truncate large results
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def add_query(self, query: str) -> None:
        """Add a query to history, keeping last 10."""
        self.query_history.append(query)
        if len(self.query_history) > 10:
            self.query_history = self.query_history[-10:]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "tool_calls": self.tool_calls,
            "intermediate_results": self.intermediate_results,
            "discovered_schemas": self.discovered_schemas,
            "query_history": self.query_history,
            "execution_time_ms": self.execution_time_ms,
            "retry_count": self.retry_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentContext":
        """Create from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            agent_type=data["agent_type"],
            tool_calls=data.get("tool_calls", []),
            intermediate_results=data.get("intermediate_results", []),
            discovered_schemas=data.get("discovered_schemas", {}),
            query_history=data.get("query_history", []),
            execution_time_ms=data.get("execution_time_ms", 0),
            retry_count=data.get("retry_count", 0),
        )


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
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "shared_data": self.shared_data,
            "pending_task": self.pending_task,
            "task_context": self.task_context,
            "data_sources_used": self.data_sources_used,
            "transformations_applied": self.transformations_applied,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HandoffContext":
        """Create from dictionary."""
        return cls(
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            reason=data["reason"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            shared_data=data.get("shared_data", {}),
            pending_task=data.get("pending_task"),
            task_context=data.get("task_context", {}),
            data_sources_used=data.get("data_sources_used", []),
            transformations_applied=data.get("transformations_applied", []),
        )


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
        agent_type: Literal["lakehouse", "warehouse", "realtime", "pipeline", "powerbi"]
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
    
    def add_data_source(self, source: str) -> None:
        """Add a data source to lineage tracking."""
        if self.current_handoff:
            if source not in self.current_handoff.data_sources_used:
                self.current_handoff.data_sources_used.append(source)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "session": self.session.to_dict(),
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "handoff_history": [h.to_dict() for h in self.handoff_history],
            "current_handoff": self.current_handoff.to_dict() if self.current_handoff else None,
            "workspace_id": self.workspace_id,
            "active_connections": self.active_connections,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionContext":
        """Create from dictionary."""
        return cls(
            session=SessionContext.from_dict(data["session"]),
            agents={k: AgentContext.from_dict(v) for k, v in data.get("agents", {}).items()},
            handoff_history=[HandoffContext.from_dict(h) for h in data.get("handoff_history", [])],
            current_handoff=HandoffContext.from_dict(data["current_handoff"]) if data.get("current_handoff") else None,
            workspace_id=data.get("workspace_id", ""),
            active_connections=data.get("active_connections", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            version=data.get("version", 1),
        )


# =============================================================================
# Context Store Protocol
# =============================================================================

class ContextStore(Protocol):
    """Protocol for context storage implementations."""
    
    async def get(self, thread_id: str) -> InteractionContext | None:
        """Retrieve context for a thread."""
        ...
    
    async def save(self, context: InteractionContext) -> None:
        """Save context to storage."""
        ...
    
    async def delete(self, thread_id: str) -> None:
        """Delete context for a thread."""
        ...
    
    async def get_or_create(
        self,
        thread_id: str,
        user_id: str | None = None,
        workspace_id: str = ""
    ) -> InteractionContext:
        """Get existing context or create new one."""
        ...


# =============================================================================
# In-Memory Context Store (Development)
# =============================================================================

class InMemoryContextStore:
    """Thread-safe in-memory context store for development."""
    
    def __init__(self):
        self._store: dict[str, InteractionContext] = {}
    
    async def get(self, thread_id: str) -> InteractionContext | None:
        """Retrieve context for a thread."""
        return self._store.get(thread_id)
    
    async def save(self, context: InteractionContext) -> None:
        """Save context to store."""
        context.updated_at = datetime.utcnow()
        self._store[context.session.thread_id] = context
    
    async def delete(self, thread_id: str) -> None:
        """Delete context for a thread."""
        self._store.pop(thread_id, None)
    
    async def get_or_create(
        self,
        thread_id: str,
        user_id: str | None = None,
        workspace_id: str = ""
    ) -> InteractionContext:
        """Get existing context or create new one."""
        if thread_id not in self._store:
            self._store[thread_id] = InteractionContext.create(
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
            )
        return self._store[thread_id]
    
    def clear(self) -> None:
        """Clear all stored contexts."""
        self._store.clear()


# =============================================================================
# Azure Cosmos DB Context Store (Production)
# =============================================================================

class CosmosDBContextStore:
    """Production context store using Azure Cosmos DB."""
    
    def __init__(
        self,
        endpoint: str,
        database_name: str,
        container_name: str,
        ttl_days: int = 7
    ):
        self.endpoint = endpoint
        self.database_name = database_name
        self.container_name = container_name
        self.ttl_seconds = ttl_days * 86400
        self._container = None
    
    async def _ensure_client(self):
        """Lazy initialization of Cosmos client."""
        if self._container is None:
            from azure.cosmos.aio import CosmosClient
            from azure.identity.aio import DefaultAzureCredential
            
            credential = DefaultAzureCredential()
            client = CosmosClient(self.endpoint, credential)
            database = client.get_database_client(self.database_name)
            self._container = database.get_container_client(self.container_name)
    
    async def get(self, thread_id: str) -> InteractionContext | None:
        """Retrieve context for a thread."""
        await self._ensure_client()
        
        try:
            item = await self._container.read_item(
                item=thread_id,
                partition_key=thread_id
            )
            return InteractionContext.from_dict(item["context"])
        except Exception:
            return None
    
    async def save(self, context: InteractionContext) -> None:
        """Save context to Cosmos DB."""
        await self._ensure_client()
        
        context.updated_at = datetime.utcnow()
        
        item = {
            "id": context.session.thread_id,
            "_partitionKey": context.session.thread_id,
            "context": context.to_dict(),
            "ttl": self.ttl_seconds,
        }
        
        await self._container.upsert_item(item)
    
    async def delete(self, thread_id: str) -> None:
        """Delete context for a thread."""
        await self._ensure_client()
        
        try:
            await self._container.delete_item(
                item=thread_id,
                partition_key=thread_id
            )
        except Exception:
            pass  # Item may not exist
    
    async def get_or_create(
        self,
        thread_id: str,
        user_id: str | None = None,
        workspace_id: str = ""
    ) -> InteractionContext:
        """Get existing context or create new one."""
        existing = await self.get(thread_id)
        if existing:
            return existing
        
        context = InteractionContext.create(
            thread_id=thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        await self.save(context)
        return context


# =============================================================================
# Redis Context Store (High Performance)
# =============================================================================

class RedisContextStore:
    """High-performance context store using Azure Redis Cache."""
    
    def __init__(
        self,
        connection_string: str,
        ttl_seconds: int = 3600
    ):
        self.connection_string = connection_string
        self.ttl_seconds = ttl_seconds
        self._client = None
    
    async def _ensure_client(self):
        """Lazy initialization of Redis client."""
        if self._client is None:
            import redis.asyncio as redis
            self._client = redis.from_url(self.connection_string)
    
    async def get(self, thread_id: str) -> InteractionContext | None:
        """Retrieve context for a thread."""
        await self._ensure_client()
        
        key = f"context:{thread_id}"
        data = await self._client.get(key)
        
        if data:
            return InteractionContext.from_dict(json.loads(data))
        return None
    
    async def save(self, context: InteractionContext) -> None:
        """Save context with TTL."""
        await self._ensure_client()
        
        context.updated_at = datetime.utcnow()
        
        key = f"context:{context.session.thread_id}"
        data = json.dumps(context.to_dict())
        
        await self._client.setex(key, self.ttl_seconds, data)
    
    async def delete(self, thread_id: str) -> None:
        """Delete context for a thread."""
        await self._ensure_client()
        
        key = f"context:{thread_id}"
        await self._client.delete(key)
    
    async def get_or_create(
        self,
        thread_id: str,
        user_id: str | None = None,
        workspace_id: str = ""
    ) -> InteractionContext:
        """Get existing context or create new one."""
        existing = await self.get(thread_id)
        if existing:
            return existing
        
        context = InteractionContext.create(
            thread_id=thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        await self.save(context)
        return context


# =============================================================================
# Context Manager
# =============================================================================

class ContextManager:
    """High-level context management for agent interactions."""
    
    def __init__(self, store: ContextStore):
        self.store = store
    
    async def start_session(
        self,
        thread_id: str,
        user_id: str | None = None,
        workspace_id: str = ""
    ) -> InteractionContext:
        """Start a new session or resume an existing one."""
        context = await self.store.get_or_create(
            thread_id=thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        
        # Update session activity
        context.session.last_activity = datetime.utcnow()
        context.session.message_count += 1
        await self.store.save(context)
        
        return context
    
    async def record_agent_handoff(
        self,
        thread_id: str,
        from_agent: str,
        to_agent: str,
        reason: str,
        data: dict[str, Any] | None = None
    ) -> InteractionContext:
        """Record a handoff between agents."""
        context = await self.store.get(thread_id)
        if not context:
            raise ValueError(f"No context found for thread: {thread_id}")
        
        context.record_handoff(from_agent, to_agent, reason, data)
        await self.store.save(context)
        
        return context
    
    async def add_tool_execution(
        self,
        thread_id: str,
        agent_name: str,
        agent_type: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any
    ) -> InteractionContext:
        """Record a tool execution in agent context."""
        context = await self.store.get(thread_id)
        if not context:
            raise ValueError(f"No context found for thread: {thread_id}")
        
        agent_ctx = context.get_or_create_agent_context(agent_name, agent_type)
        agent_ctx.add_tool_call(tool_name, tool_args, tool_result)
        
        await self.store.save(context)
        return context
    
    async def get_context_summary(self, thread_id: str) -> dict[str, Any]:
        """Get a summary of the current context state."""
        context = await self.store.get(thread_id)
        if not context:
            return {"exists": False}
        
        return {
            "exists": True,
            "thread_id": context.session.thread_id,
            "user_id": context.session.user_id,
            "message_count": context.session.message_count,
            "conversation_summary": context.session.conversation_summary,
            "agents_used": list(context.agents.keys()),
            "current_handoff": {
                "from": context.current_handoff.from_agent,
                "to": context.current_handoff.to_agent,
                "reason": context.current_handoff.reason,
            } if context.current_handoff else None,
            "version": context.version,
            "last_activity": context.session.last_activity.isoformat(),
        }
    
    async def end_session(self, thread_id: str, preserve: bool = True) -> None:
        """End a session."""
        if not preserve:
            await self.store.delete(thread_id)


# =============================================================================
# Factory Functions
# =============================================================================

def create_context_store(store_type: str = "memory", **kwargs) -> ContextStore:
    """Create a context store based on type.
    
    Args:
        store_type: Type of store ("memory", "cosmos", "redis").
        **kwargs: Additional arguments for store initialization.
        
    Returns:
        Configured context store.
    """
    if store_type == "memory":
        return InMemoryContextStore()
    elif store_type == "cosmos":
        return CosmosDBContextStore(
            endpoint=kwargs["endpoint"],
            database_name=kwargs["database_name"],
            container_name=kwargs["container_name"],
            ttl_days=kwargs.get("ttl_days", 7),
        )
    elif store_type == "redis":
        return RedisContextStore(
            connection_string=kwargs["connection_string"],
            ttl_seconds=kwargs.get("ttl_seconds", 3600),
        )
    else:
        raise ValueError(f"Unknown store type: {store_type}")


def create_context_manager(
    store_type: str = "memory",
    **store_kwargs
) -> ContextManager:
    """Create a context manager with the specified store.
    
    Args:
        store_type: Type of backing store.
        **store_kwargs: Additional arguments for store.
        
    Returns:
        Configured context manager.
    """
    store = create_context_store(store_type, **store_kwargs)
    return ContextManager(store=store)
