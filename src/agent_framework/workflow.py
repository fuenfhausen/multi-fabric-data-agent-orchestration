# Copyright (c) Microsoft. All rights reserved.
"""Workflow builders for Microsoft Agent Framework Fabric orchestration."""

from agent_framework import (
    ChatAgent,
    HandoffBuilder,
    ConcurrentBuilder,
    Workflow,
)
from agent_framework.azure import AzureOpenAIChatClient

from .agents import (
    create_orchestrator_agent,
    create_lakehouse_agent,
    create_warehouse_agent,
    create_realtime_agent,
)


def build_handoff_workflow(
    chat_client: AzureOpenAIChatClient | None = None,
    enable_autonomous_mode: bool = False,
    autonomous_turn_limits: dict[str, int] | None = None,
) -> Workflow:
    """Build a handoff workflow for Fabric agent orchestration.
    
    The handoff pattern routes requests from the orchestrator to specialized
    agents, which can hand back or transfer to other specialists.
    
    Args:
        chat_client: Optional shared chat client for all agents.
        enable_autonomous_mode: Whether agents can iterate autonomously.
        autonomous_turn_limits: Per-agent turn limits for autonomous mode.
        
    Returns:
        Compiled Workflow ready for execution.
    """
    # Create agents
    orchestrator = create_orchestrator_agent(chat_client)
    lakehouse_agent = create_lakehouse_agent(chat_client)
    warehouse_agent = create_warehouse_agent(chat_client)
    realtime_agent = create_realtime_agent(chat_client)
    
    # Build the handoff workflow
    builder = HandoffBuilder(
        name="fabric_data_orchestration",
        participants=[orchestrator, lakehouse_agent, warehouse_agent, realtime_agent],
        description="Multi-fabric data agent orchestration with handoff routing"
    )
    
    # Set orchestrator as the starting agent
    builder.with_start_agent(orchestrator)
    
    # Configure handoff routes
    # Orchestrator can route to any specialist
    builder.add_handoff(
        orchestrator,
        [lakehouse_agent, warehouse_agent, realtime_agent],
        description="Route to specialized Fabric agent based on request type"
    )
    
    # Specialists can hand back to orchestrator or transfer to other specialists
    builder.add_handoff(
        lakehouse_agent,
        [orchestrator, warehouse_agent],
        description="Hand back to orchestrator or transfer to warehouse for SQL operations"
    )
    
    builder.add_handoff(
        warehouse_agent,
        [orchestrator, lakehouse_agent],
        description="Hand back to orchestrator or transfer to lakehouse for Delta operations"
    )
    
    builder.add_handoff(
        realtime_agent,
        [orchestrator],
        description="Hand back to orchestrator after completing real-time analysis"
    )
    
    # Enable autonomous mode if requested
    if enable_autonomous_mode:
        turn_limits = autonomous_turn_limits or {
            "lakehouse_agent": 5,
            "warehouse_agent": 5,
            "realtime_agent": 5,
        }
        builder.with_autonomous_mode(
            agents=[lakehouse_agent, warehouse_agent, realtime_agent],
            turn_limits=turn_limits
        )
    
    return builder.build()


def build_concurrent_workflow(
    chat_client: AzureOpenAIChatClient | None = None,
    agents_to_include: list[str] | None = None,
) -> Workflow:
    """Build a concurrent workflow for parallel Fabric queries.
    
    The concurrent pattern fans out the same request to multiple agents
    simultaneously and aggregates their results.
    
    Args:
        chat_client: Optional shared chat client for all agents.
        agents_to_include: List of agent types to include ('lakehouse', 'warehouse', 'realtime').
                          Defaults to all three.
        
    Returns:
        Compiled Workflow ready for execution.
    """
    agents_to_include = agents_to_include or ["lakehouse", "warehouse", "realtime"]
    
    # Create only the requested agents
    participants = []
    
    if "lakehouse" in agents_to_include:
        participants.append(create_lakehouse_agent(chat_client))
    
    if "warehouse" in agents_to_include:
        participants.append(create_warehouse_agent(chat_client))
    
    if "realtime" in agents_to_include:
        participants.append(create_realtime_agent(chat_client))
    
    if not participants:
        raise ValueError("At least one agent type must be included")
    
    # Build concurrent workflow
    builder = ConcurrentBuilder()
    builder.participants(participants)
    
    # Custom aggregator to combine results from all agents
    async def aggregate_results(results: list) -> dict:
        """Aggregate results from multiple Fabric agents."""
        aggregated = {
            "sources": len(results),
            "results": [],
            "summary": []
        }
        
        for i, result in enumerate(results):
            agent_name = participants[i].name if i < len(participants) else f"agent_{i}"
            aggregated["results"].append({
                "agent": agent_name,
                "data": result
            })
            aggregated["summary"].append(f"{agent_name}: completed")
        
        return aggregated
    
    builder.with_aggregator(aggregate_results)
    
    return builder.build()


def build_sequential_workflow(
    chat_client: AzureOpenAIChatClient | None = None,
    pipeline: list[str] | None = None,
) -> Workflow:
    """Build a sequential workflow for data pipelines.
    
    The sequential pattern executes agents one after another, passing
    context between them.
    
    Args:
        chat_client: Optional shared chat client for all agents.
        pipeline: Ordered list of agent types ('lakehouse', 'warehouse', 'realtime').
                 Defaults to ['lakehouse', 'warehouse'].
        
    Returns:
        Compiled Workflow ready for execution.
    """
    pipeline = pipeline or ["lakehouse", "warehouse"]
    
    agent_creators = {
        "lakehouse": create_lakehouse_agent,
        "warehouse": create_warehouse_agent,
        "realtime": create_realtime_agent,
    }
    
    participants = []
    for agent_type in pipeline:
        if agent_type not in agent_creators:
            raise ValueError(f"Unknown agent type: {agent_type}")
        participants.append(agent_creators[agent_type](chat_client))
    
    # For sequential workflow, we use HandoffBuilder with linear handoffs
    builder = HandoffBuilder(
        name="fabric_sequential_pipeline",
        participants=participants,
        description="Sequential data pipeline across Fabric agents"
    )
    
    builder.with_start_agent(participants[0])
    
    # Chain agents sequentially
    for i in range(len(participants) - 1):
        builder.add_handoff(
            participants[i],
            [participants[i + 1]],
            description=f"Pass to next stage: {participants[i + 1].name}"
        )
    
    return builder.build()
