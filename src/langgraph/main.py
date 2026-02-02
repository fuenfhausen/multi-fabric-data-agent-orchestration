# Copyright (c) Microsoft. All rights reserved.
"""Main entry point for LangGraph Fabric agent orchestration."""

import asyncio
from typing import AsyncIterator, Literal, Any

from langchain_core.messages import HumanMessage, BaseMessage

from ..shared.config import get_settings
from .state import FabricAgentState, create_initial_state
from .graph import (
    build_supervisor_graph,
    build_parallel_graph,
    build_sequential_graph,
    build_human_in_loop_graph,
)


async def run_graph(
    query: str,
    graph_type: Literal["supervisor", "parallel", "sequential", "hitl"] = "supervisor",
    verbose: bool = True,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run a Fabric orchestration graph.
    
    Args:
        query: The user's query to process.
        graph_type: Type of graph to use.
        verbose: Whether to print events during execution.
        thread_id: Optional thread ID for stateful execution.
        
    Returns:
        Final state from the graph execution.
    """
    settings = get_settings()
    
    # Build the appropriate graph
    if graph_type == "supervisor":
        graph = build_supervisor_graph()
    elif graph_type == "parallel":
        graph = build_parallel_graph()
    elif graph_type == "sequential":
        graph = build_sequential_graph()
    elif graph_type == "hitl":
        graph = build_human_in_loop_graph()
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # Create initial state
    initial_state = create_initial_state(
        workspace_id=settings.fabric_workspace_id,
        query=query
    )
    
    # Configuration
    config = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    
    # Run the graph
    if verbose:
        print(f"\n🚀 Running {graph_type} graph...")
        print(f"📝 Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print("-" * 50)
    
    # Stream events
    final_state = None
    async for event in graph.astream(initial_state, config):
        if verbose:
            _print_event(event)
        final_state = event
    
    if verbose:
        print("-" * 50)
        print("✅ Graph execution complete")
    
    return final_state


def run_graph_sync(
    query: str,
    graph_type: Literal["supervisor", "parallel", "sequential", "hitl"] = "supervisor",
    verbose: bool = True,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for run_graph."""
    return asyncio.run(run_graph(query, graph_type, verbose, thread_id))


async def run_graph_interactive(
    graph_type: Literal["supervisor", "parallel", "sequential"] = "supervisor",
) -> None:
    """Run an interactive session with the Fabric orchestration graph.
    
    Args:
        graph_type: Type of graph to use.
    """
    settings = get_settings()
    
    # Build the graph
    if graph_type == "supervisor":
        graph = build_supervisor_graph()
    elif graph_type == "parallel":
        graph = build_parallel_graph()
    elif graph_type == "sequential":
        graph = build_sequential_graph()
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    print(f"\n🚀 Fabric Agent Orchestration (LangGraph - {graph_type} mode)")
    print("=" * 50)
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["quit", "exit"]:
                print("\nGoodbye! 👋")
                break
            
            print("\n" + "-" * 40)
            
            # Create initial state
            initial_state = create_initial_state(
                workspace_id=settings.fabric_workspace_id,
                query=query
            )
            
            # Run the graph
            async for event in graph.astream(initial_state):
                _print_event(event)
            
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


async def stream_graph(
    query: str,
    graph_type: Literal["supervisor", "parallel", "sequential"] = "supervisor",
    thread_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream events from a Fabric orchestration graph.
    
    Args:
        query: The user's query to process.
        graph_type: Type of graph to use.
        thread_id: Optional thread ID for stateful execution.
        
    Yields:
        State updates as they occur.
    """
    settings = get_settings()
    
    # Build the appropriate graph
    if graph_type == "supervisor":
        graph = build_supervisor_graph()
    elif graph_type == "parallel":
        graph = build_parallel_graph()
    elif graph_type == "sequential":
        graph = build_sequential_graph()
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # Create initial state
    initial_state = create_initial_state(
        workspace_id=settings.fabric_workspace_id,
        query=query
    )
    
    # Configuration
    config = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    
    async for event in graph.astream(initial_state, config):
        yield event


async def stream_graph_events(
    query: str,
    graph_type: Literal["supervisor", "parallel", "sequential"] = "supervisor",
    thread_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream detailed events from a Fabric orchestration graph.
    
    Uses astream_events for more granular event streaming including
    tool calls and LLM tokens.
    
    Args:
        query: The user's query to process.
        graph_type: Type of graph to use.
        thread_id: Optional thread ID for stateful execution.
        
    Yields:
        Detailed events as they occur.
    """
    settings = get_settings()
    
    # Build the appropriate graph
    if graph_type == "supervisor":
        graph = build_supervisor_graph()
    elif graph_type == "parallel":
        graph = build_parallel_graph()
    elif graph_type == "sequential":
        graph = build_sequential_graph()
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # Create initial state
    initial_state = create_initial_state(
        workspace_id=settings.fabric_workspace_id,
        query=query
    )
    
    # Configuration
    config = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    
    async for event in graph.astream_events(initial_state, config, version="v2"):
        yield event


def _print_event(event: dict[str, Any]) -> None:
    """Print a graph event to the console."""
    # Event is a dict with node name as key
    for node_name, node_output in event.items():
        if node_name == "__end__":
            continue
            
        print(f"\n🔹 Node: {node_name}")
        
        # Print current agent
        if "current_agent" in node_output and node_output["current_agent"]:
            print(f"   Agent: {node_output['current_agent']}")
        
        # Print routing decision
        if "next_agent" in node_output and node_output["next_agent"]:
            print(f"   → Routing to: {node_output['next_agent']}")
        
        # Print error if any
        if "error" in node_output and node_output["error"]:
            print(f"   ❌ Error: {node_output['error']}")
        
        # Print messages
        if "messages" in node_output:
            messages = node_output["messages"]
            if messages:
                last_message = messages[-1] if isinstance(messages, list) else messages
                if hasattr(last_message, "content"):
                    content = last_message.content
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"   💬 {content}")


def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Fabric Data Agent Orchestration (LangGraph)"
    )
    parser.add_argument(
        "--graph",
        "-g",
        type=str,
        choices=["supervisor", "parallel", "sequential", "hitl"],
        default="supervisor",
        help="Graph type to use (default: supervisor)"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Query to execute (omit for interactive mode)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--thread-id",
        "-t",
        type=str,
        help="Thread ID for stateful execution"
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        result = asyncio.run(run_graph(
            query=args.query,
            graph_type=args.graph,
            verbose=not args.quiet,
            thread_id=args.thread_id
        ))
        
        if not args.quiet:
            print("\n" + "=" * 50)
            print("Final State:")
            if result:
                for key, value in result.items():
                    if key != "messages":
                        print(f"  {key}: {value}")
    else:
        # Interactive mode
        asyncio.run(run_graph_interactive(
            graph_type=args.graph
        ))


if __name__ == "__main__":
    main()
