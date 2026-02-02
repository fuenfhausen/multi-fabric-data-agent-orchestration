# Copyright (c) Microsoft. All rights reserved.
"""Graph builders for LangGraph Fabric agent orchestration."""

from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import FabricAgentState, MultiSourceState
from .nodes import (
    orchestrator_node,
    lakehouse_node,
    warehouse_node,
    realtime_node,
    aggregator_node,
    route_to_agent,
)


def build_supervisor_graph(
    checkpointer: MemorySaver | None = None,
    interrupt_before: list[str] | None = None,
) -> StateGraph:
    """Build a supervisor orchestration graph.
    
    The orchestrator (supervisor) routes requests to specialized agent nodes
    based on intent classification.
    
    Flow:
        orchestrator → [lakehouse | warehouse | realtime] → orchestrator → END
    
    Args:
        checkpointer: Optional checkpointer for state persistence.
        interrupt_before: Optional list of nodes to interrupt before.
        
    Returns:
        Compiled StateGraph ready for execution.
    """
    # Create the graph
    workflow = StateGraph(FabricAgentState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("lakehouse", lakehouse_node)
    workflow.add_node("warehouse", warehouse_node)
    workflow.add_node("realtime", realtime_node)
    
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
            "end": END
        }
    )
    
    # All agents return to END after completing
    # (For multi-turn, they could return to orchestrator)
    workflow.add_edge("lakehouse", END)
    workflow.add_edge("warehouse", END)
    workflow.add_edge("realtime", END)
    
    # Compile with optional checkpointer
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    
    return workflow.compile(**compile_kwargs)


def build_supervisor_graph_multi_turn(
    checkpointer: MemorySaver | None = None,
    max_iterations: int = 5,
) -> StateGraph:
    """Build a multi-turn supervisor graph.
    
    Similar to supervisor graph but agents return to orchestrator
    for potential follow-up actions.
    
    Flow:
        orchestrator ↔ [lakehouse | warehouse | realtime] (loop until END)
    
    Args:
        checkpointer: Optional checkpointer for state persistence.
        max_iterations: Maximum number of routing iterations.
        
    Returns:
        Compiled StateGraph ready for execution.
    """
    workflow = StateGraph(FabricAgentState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("lakehouse", lakehouse_node)
    workflow.add_node("warehouse", warehouse_node)
    workflow.add_node("realtime", realtime_node)
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Conditional routing from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        route_to_agent,
        {
            "lakehouse": "lakehouse",
            "warehouse": "warehouse",
            "realtime": "realtime",
            "end": END
        }
    )
    
    # Agents return to orchestrator for potential follow-up
    workflow.add_edge("lakehouse", "orchestrator")
    workflow.add_edge("warehouse", "orchestrator")
    workflow.add_edge("realtime", "orchestrator")
    
    # Compile
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    
    return workflow.compile(**compile_kwargs)


def build_parallel_graph(
    agents_to_include: list[str] | None = None,
    checkpointer: MemorySaver | None = None,
) -> StateGraph:
    """Build a parallel fan-out/fan-in graph.
    
    Executes multiple agents in parallel and aggregates their results.
    
    Flow:
        fan_out → [lakehouse, warehouse, realtime] (parallel) → aggregator → END
    
    Args:
        agents_to_include: List of agents to include. Defaults to all.
        checkpointer: Optional checkpointer for state persistence.
        
    Returns:
        Compiled StateGraph ready for execution.
    """
    from langgraph.constants import Send
    
    agents_to_include = agents_to_include or ["lakehouse", "warehouse", "realtime"]
    
    # Map agent names to node functions
    agent_nodes = {
        "lakehouse": lakehouse_node,
        "warehouse": warehouse_node,
        "realtime": realtime_node,
    }
    
    def fan_out_node(state: MultiSourceState) -> list[Send]:
        """Fan-out to multiple agents in parallel."""
        parallel_agents = state.get("parallel_agents", agents_to_include)
        
        return [
            Send(agent, {
                "messages": state["messages"],
                "workspace_id": state["workspace_id"],
                "current_agent": None,
                "query_type": None,
                "query_results": None,
                "error": None,
                "next_agent": None,
                "requires_approval": False,
            })
            for agent in parallel_agents
            if agent in agent_nodes
        ]
    
    # Create the graph
    workflow = StateGraph(MultiSourceState)
    
    # Add fan-out node
    workflow.add_node("fan_out", fan_out_node)
    
    # Add agent nodes
    for agent_name in agents_to_include:
        if agent_name in agent_nodes:
            workflow.add_node(agent_name, agent_nodes[agent_name])
    
    # Add aggregator
    workflow.add_node("aggregator", aggregator_node)
    
    # Set entry point
    workflow.set_entry_point("fan_out")
    
    # Fan-out edges (using Send for parallel execution)
    workflow.add_conditional_edges("fan_out", fan_out_node)
    
    # All parallel nodes converge to aggregator
    for agent_name in agents_to_include:
        if agent_name in agent_nodes:
            workflow.add_edge(agent_name, "aggregator")
    
    # Aggregator to END
    workflow.add_edge("aggregator", END)
    
    # Compile
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    
    return workflow.compile(**compile_kwargs)


def build_sequential_graph(
    pipeline: list[str] | None = None,
    checkpointer: MemorySaver | None = None,
) -> StateGraph:
    """Build a sequential pipeline graph.
    
    Executes agents one after another, passing context between them.
    
    Flow:
        agent1 → agent2 → agent3 → END
    
    Args:
        pipeline: Ordered list of agent types. Defaults to ['lakehouse', 'warehouse'].
        checkpointer: Optional checkpointer for state persistence.
        
    Returns:
        Compiled StateGraph ready for execution.
    """
    pipeline = pipeline or ["lakehouse", "warehouse"]
    
    # Map agent names to node functions
    agent_nodes = {
        "lakehouse": lakehouse_node,
        "warehouse": warehouse_node,
        "realtime": realtime_node,
    }
    
    # Validate pipeline
    for agent_name in pipeline:
        if agent_name not in agent_nodes:
            raise ValueError(f"Unknown agent type: {agent_name}")
    
    # Create the graph
    workflow = StateGraph(FabricAgentState)
    
    # Add nodes
    for agent_name in pipeline:
        workflow.add_node(agent_name, agent_nodes[agent_name])
    
    # Set entry point
    workflow.set_entry_point(pipeline[0])
    
    # Chain agents sequentially
    for i in range(len(pipeline) - 1):
        workflow.add_edge(pipeline[i], pipeline[i + 1])
    
    # Last agent to END
    workflow.add_edge(pipeline[-1], END)
    
    # Compile
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    
    return workflow.compile(**compile_kwargs)


def build_human_in_loop_graph(
    checkpointer: MemorySaver | None = None,
) -> StateGraph:
    """Build a graph with human-in-the-loop approval.
    
    Interrupts before executing queries that require approval.
    
    Flow:
        orchestrator → [approval_check] → agent → END
    
    Args:
        checkpointer: Checkpointer for state persistence (required for HITL).
        
    Returns:
        Compiled StateGraph ready for execution.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    def approval_check_node(state: FabricAgentState) -> dict:
        """Node that checks if approval is required."""
        # This node will be interrupted before execution
        # User can modify state or approve continuation
        return {"requires_approval": False}
    
    # Create the graph
    workflow = StateGraph(FabricAgentState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("approval_check", approval_check_node)
    workflow.add_node("lakehouse", lakehouse_node)
    workflow.add_node("warehouse", warehouse_node)
    workflow.add_node("realtime", realtime_node)
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Orchestrator routes through approval check
    def route_through_approval(state: FabricAgentState) -> str:
        next_agent = state.get("next_agent", "end")
        if next_agent in ["lakehouse", "warehouse", "realtime"]:
            return "approval_check"
        return "end"
    
    workflow.add_conditional_edges(
        "orchestrator",
        route_through_approval,
        {
            "approval_check": "approval_check",
            "end": END
        }
    )
    
    # After approval, route to appropriate agent
    workflow.add_conditional_edges(
        "approval_check",
        route_to_agent,
        {
            "lakehouse": "lakehouse",
            "warehouse": "warehouse",
            "realtime": "realtime",
            "end": END
        }
    )
    
    # Agents to END
    workflow.add_edge("lakehouse", END)
    workflow.add_edge("warehouse", END)
    workflow.add_edge("realtime", END)
    
    # Compile with interrupt before approval_check
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["approval_check"]
    )
