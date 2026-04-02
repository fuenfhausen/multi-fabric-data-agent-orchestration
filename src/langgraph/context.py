# Copyright (c) Microsoft. All rights reserved.
"""Context management for LangGraph Fabric agent orchestration.

This module provides context lifecycle management, persistence, and
summarization capabilities for multi-turn conversations and cross-agent
coordination.
"""

from typing import Any, Protocol
from datetime import datetime
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI

from .state import (
    InteractionContext,
    EnhancedFabricAgentState,
    SessionContext,
    AgentContext,
    CrossAgentContext,
    ContextMetadata,
    create_interaction_context,
    create_agent_context,
    update_session_activity,
    record_handoff,
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
        max_context_tokens: int = 128000
    ) -> InteractionContext:
        """Get existing context or create new one."""
        ...


# =============================================================================
# In-Memory Context Store (Development)
# =============================================================================

class InMemoryContextStore:
    """Thread-safe in-memory context store for development.
    
    This store keeps all context in memory and is suitable for
    development and testing. Data is lost when the process ends.
    """
    
    def __init__(self):
        self._store: dict[str, InteractionContext] = {}
    
    async def get(self, thread_id: str) -> InteractionContext | None:
        """Retrieve context for a thread."""
        return self._store.get(thread_id)
    
    async def save(self, context: InteractionContext) -> None:
        """Save context to store."""
        context = update_session_activity(context)
        self._store[context["session"]["thread_id"]] = context
    
    async def delete(self, thread_id: str) -> None:
        """Delete context for a thread."""
        self._store.pop(thread_id, None)
    
    async def get_or_create(
        self,
        thread_id: str,
        user_id: str | None = None,
        max_context_tokens: int = 128000
    ) -> InteractionContext:
        """Get existing context or create new one."""
        if thread_id not in self._store:
            self._store[thread_id] = create_interaction_context(
                thread_id=thread_id,
                user_id=user_id,
                max_context_tokens=max_context_tokens,
            )
        return self._store[thread_id]
    
    def clear(self) -> None:
        """Clear all stored contexts."""
        self._store.clear()


# =============================================================================
# SQLite Context Store (Local Persistence)
# =============================================================================

class SqliteContextStore:
    """SQLite-based context store for local persistence.
    
    Provides persistent storage using SQLite, suitable for
    single-instance deployments and local development with
    persistence requirements.
    """
    
    def __init__(self, db_path: str = "fabric_agent_context.db"):
        self.db_path = db_path
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        """Ensure database tables exist."""
        if self._initialized:
            return
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS contexts (
                    thread_id TEXT PRIMARY KEY,
                    context_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER DEFAULT 1
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_contexts_updated 
                ON contexts(updated_at)
            """)
            await db.commit()
        
        self._initialized = True
    
    async def get(self, thread_id: str) -> InteractionContext | None:
        """Retrieve context for a thread."""
        await self._ensure_initialized()
        
        import aiosqlite
        import json
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT context_json FROM contexts WHERE thread_id = ?",
                (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return json.loads(row[0])
        return None
    
    async def save(self, context: InteractionContext) -> None:
        """Save context to SQLite."""
        await self._ensure_initialized()
        
        import aiosqlite
        import json
        
        context = update_session_activity(context)
        thread_id = context["session"]["thread_id"]
        context_json = json.dumps(context)
        now = datetime.utcnow().isoformat()
        version = context["metadata"]["version"]
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO contexts (thread_id, context_json, created_at, updated_at, version)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    context_json = excluded.context_json,
                    updated_at = excluded.updated_at,
                    version = excluded.version
            """, (thread_id, context_json, now, now, version))
            await db.commit()
    
    async def delete(self, thread_id: str) -> None:
        """Delete context for a thread."""
        await self._ensure_initialized()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM contexts WHERE thread_id = ?",
                (thread_id,)
            )
            await db.commit()
    
    async def get_or_create(
        self,
        thread_id: str,
        user_id: str | None = None,
        max_context_tokens: int = 128000
    ) -> InteractionContext:
        """Get existing context or create new one."""
        existing = await self.get(thread_id)
        if existing:
            return existing
        
        context = create_interaction_context(
            thread_id=thread_id,
            user_id=user_id,
            max_context_tokens=max_context_tokens,
        )
        await self.save(context)
        return context
    
    async def cleanup_old_contexts(self, days: int = 7) -> int:
        """Delete contexts older than specified days.
        
        Args:
            days: Number of days after which to delete contexts.
            
        Returns:
            Number of deleted contexts.
        """
        await self._ensure_initialized()
        
        import aiosqlite
        from datetime import timedelta
        
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM contexts WHERE updated_at < ?",
                (cutoff,)
            )
            await db.commit()
            return cursor.rowcount


# =============================================================================
# Context Summarization
# =============================================================================

SUMMARIZATION_PROMPT = """Summarize the following conversation history, preserving:
1. Key user intents and goals
2. Important data sources and tables referenced
3. Query results and findings
4. Any errors encountered and how they were resolved
5. Current state of the analysis

Conversation:
{conversation}

Provide a concise summary (max 300 words) that captures the essential context for continuing the analysis."""


async def summarize_context(
    state: EnhancedFabricAgentState,
    llm: AzureChatOpenAI,
    summarization_threshold: float = 0.8
) -> dict[str, Any]:
    """Summarize conversation context if approaching token limits.
    
    This function checks if the current token count is approaching
    the maximum allowed and generates a summary if needed.
    
    Args:
        state: Current workflow state.
        llm: Language model for summarization.
        summarization_threshold: Trigger summarization at this % of max tokens.
        
    Returns:
        State update with summarized context if triggered, empty dict otherwise.
    """
    ctx = state["interaction_context"]
    session = ctx["session"]
    
    # Check if summarization is needed
    token_ratio = session["token_count"] / session["max_context_tokens"]
    if token_ratio < summarization_threshold:
        return {}  # No summarization needed
    
    # Build conversation text for summarization
    messages = state["messages"]
    conversation_parts = []
    for msg in messages[-20:]:  # Last 20 messages
        if isinstance(msg, HumanMessage):
            conversation_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation_parts.append(f"Assistant: {msg.content}")
        elif isinstance(msg, SystemMessage):
            conversation_parts.append(f"System: {msg.content}")
    
    conversation_text = "\n".join(conversation_parts)
    
    # Generate summary
    summary_prompt = SUMMARIZATION_PROMPT.format(conversation=conversation_text)
    response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
    
    # Update session with summary
    now = datetime.utcnow().isoformat()
    updated_session = SessionContext(
        **session,
        conversation_summary=response.content,
        last_activity=now,
        token_count=0,  # Reset token count after summarization
    )
    
    updated_metadata = ContextMetadata(
        **ctx["metadata"],
        updated_at=now,
        version=ctx["metadata"]["version"] + 1,
    )
    
    # Create summary message and trim conversation
    summary_message = SystemMessage(
        content=f"[Previous conversation summary]: {response.content}"
    )
    
    return {
        "messages": [summary_message] + list(messages[-5:]),
        "interaction_context": InteractionContext(
            session=updated_session,
            agents=ctx["agents"],
            shared=ctx["shared"],
            metadata=updated_metadata,
        ),
    }


# =============================================================================
# Context Manager
# =============================================================================

class ContextManager:
    """High-level context management for agent interactions.
    
    Provides a unified interface for managing context throughout
    the agent orchestration lifecycle.
    """
    
    def __init__(
        self,
        store: ContextStore,
        llm: AzureChatOpenAI | None = None,
        summarization_threshold: float = 0.8
    ):
        self.store = store
        self.llm = llm
        self.summarization_threshold = summarization_threshold
    
    async def start_session(
        self,
        thread_id: str,
        user_id: str | None = None,
        workspace_id: str = ""
    ) -> InteractionContext:
        """Start a new session or resume an existing one.
        
        Args:
            thread_id: Unique thread/session identifier.
            user_id: Optional user identifier.
            workspace_id: Fabric workspace ID.
            
        Returns:
            Session context.
        """
        context = await self.store.get_or_create(
            thread_id=thread_id,
            user_id=user_id,
        )
        
        # Update session activity
        context = update_session_activity(context)
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
        """Record a handoff between agents.
        
        Args:
            thread_id: Session thread ID.
            from_agent: Source agent name.
            to_agent: Target agent name.
            reason: Reason for the handoff.
            data: Optional data to pass to target agent.
            
        Returns:
            Updated context.
        """
        context = await self.store.get(thread_id)
        if not context:
            raise ValueError(f"No context found for thread: {thread_id}")
        
        context = record_handoff(context, from_agent, to_agent, reason, data)
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
        """Record a tool execution in agent context.
        
        Args:
            thread_id: Session thread ID.
            agent_name: Name of the agent.
            agent_type: Type of the agent.
            tool_name: Name of the tool executed.
            tool_args: Arguments passed to the tool.
            tool_result: Result from the tool.
            
        Returns:
            Updated context.
        """
        context = await self.store.get(thread_id)
        if not context:
            raise ValueError(f"No context found for thread: {thread_id}")
        
        # Get or create agent context
        agents = dict(context["agents"])
        if agent_name not in agents:
            agents[agent_name] = create_agent_context(agent_name, agent_type)
        
        # Add tool call
        agent_ctx = agents[agent_name]
        tool_calls = list(agent_ctx["tool_calls"])
        tool_calls.append({
            "tool": tool_name,
            "args": tool_args,
            "result": str(tool_result)[:1000],  # Truncate large results
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        agents[agent_name] = AgentContext(
            **agent_ctx,
            tool_calls=tool_calls,
        )
        
        now = datetime.utcnow().isoformat()
        updated_context = InteractionContext(
            session=context["session"],
            agents=agents,
            shared=context["shared"],
            metadata=ContextMetadata(
                **context["metadata"],
                updated_at=now,
                version=context["metadata"]["version"] + 1,
            ),
        )
        
        await self.store.save(updated_context)
        return updated_context
    
    async def get_context_summary(self, thread_id: str) -> dict[str, Any]:
        """Get a summary of the current context state.
        
        Args:
            thread_id: Session thread ID.
            
        Returns:
            Summary dictionary with key context information.
        """
        context = await self.store.get(thread_id)
        if not context:
            return {"exists": False}
        
        session = context["session"]
        shared = context["shared"]
        
        return {
            "exists": True,
            "thread_id": session["thread_id"],
            "user_id": session["user_id"],
            "message_count": session["token_count"],
            "conversation_summary": session["conversation_summary"],
            "agents_used": list(context["agents"].keys()),
            "data_sources": shared["data_sources_used"],
            "last_handoff": {
                "from": shared["handoff_from"],
                "reason": shared["handoff_reason"],
            } if shared["handoff_from"] else None,
            "version": context["metadata"]["version"],
            "last_activity": session["last_activity"],
        }
    
    async def end_session(self, thread_id: str, preserve: bool = True) -> None:
        """End a session.
        
        Args:
            thread_id: Session thread ID.
            preserve: If True, keep context for future reference.
                     If False, delete the context.
        """
        if not preserve:
            await self.store.delete(thread_id)


# =============================================================================
# Factory Functions
# =============================================================================

def create_context_store(store_type: str = "memory", **kwargs) -> ContextStore:
    """Create a context store based on type.
    
    Args:
        store_type: Type of store ("memory", "sqlite").
        **kwargs: Additional arguments for store initialization.
        
    Returns:
        Configured context store.
    """
    if store_type == "memory":
        return InMemoryContextStore()
    elif store_type == "sqlite":
        return SqliteContextStore(
            db_path=kwargs.get("db_path", "fabric_agent_context.db")
        )
    else:
        raise ValueError(f"Unknown store type: {store_type}")


def create_context_manager(
    store_type: str = "memory",
    llm: AzureChatOpenAI | None = None,
    **store_kwargs
) -> ContextManager:
    """Create a context manager with the specified store.
    
    Args:
        store_type: Type of backing store.
        llm: Optional LLM for summarization.
        **store_kwargs: Additional arguments for store.
        
    Returns:
        Configured context manager.
    """
    store = create_context_store(store_type, **store_kwargs)
    return ContextManager(store=store, llm=llm)
