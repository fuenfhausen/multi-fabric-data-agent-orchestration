# Copyright (c) Microsoft. All rights reserved.
"""Main entry point for Microsoft Agent Framework Fabric orchestration."""

import asyncio
from typing import AsyncIterator, Literal

from agent_framework import (
    WorkflowEvent,
    WorkflowOutputEvent,
    AgentRunUpdateEvent,
    RequestInfoEvent,
    HandoffSentEvent,
    ChatMessage,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential

from ..shared.config import get_settings
from .workflow import (
    build_handoff_workflow,
    build_concurrent_workflow,
    build_sequential_workflow,
)


def get_chat_client() -> AzureOpenAIChatClient:
    """Create Azure OpenAI chat client from settings."""
    settings = get_settings()
    credential = DefaultAzureCredential()
    
    return AzureOpenAIChatClient(
        credential=credential,
        endpoint=settings.azure_openai_endpoint,
        deployment=settings.azure_openai_deployment,
    )


async def run_workflow(
    query: str,
    workflow_type: Literal["handoff", "concurrent", "sequential"] = "handoff",
    chat_client: AzureOpenAIChatClient | None = None,
    verbose: bool = True,
) -> list[ChatMessage]:
    """Run a Fabric orchestration workflow.
    
    Args:
        query: The user's query to process.
        workflow_type: Type of workflow to use.
        chat_client: Optional chat client.
        verbose: Whether to print events during execution.
        
    Returns:
        List of ChatMessage responses from the workflow.
    """
    client = chat_client or get_chat_client()
    
    # Build the appropriate workflow
    if workflow_type == "handoff":
        workflow = build_handoff_workflow(client)
    elif workflow_type == "concurrent":
        workflow = build_concurrent_workflow(client)
    elif workflow_type == "sequential":
        workflow = build_sequential_workflow(client)
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    # Collect results
    results: list[ChatMessage] = []
    
    # Run the workflow
    async for event in workflow.run_stream(query):
        if verbose:
            _print_event(event)
        
        # Collect output events
        if isinstance(event, WorkflowOutputEvent):
            if isinstance(event.output, list):
                results.extend(event.output)
            else:
                results.append(event.output)
    
    return results


async def run_workflow_interactive(
    workflow_type: Literal["handoff", "concurrent", "sequential"] = "handoff",
    chat_client: AzureOpenAIChatClient | None = None,
) -> None:
    """Run an interactive session with the Fabric orchestration workflow.
    
    Args:
        workflow_type: Type of workflow to use.
        chat_client: Optional chat client.
    """
    client = chat_client or get_chat_client()
    
    # Build the workflow
    if workflow_type == "handoff":
        workflow = build_handoff_workflow(client)
    elif workflow_type == "concurrent":
        workflow = build_concurrent_workflow(client)
    elif workflow_type == "sequential":
        workflow = build_sequential_workflow(client)
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    print(f"\n🚀 Fabric Agent Orchestration ({workflow_type} mode)")
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
            
            async for event in workflow.run_stream(query):
                _print_event(event)
            
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def _print_event(event: WorkflowEvent) -> None:
    """Print a workflow event to the console."""
    if isinstance(event, AgentRunUpdateEvent):
        # Print streaming text
        if hasattr(event, 'text') and event.text:
            print(event.text, end="", flush=True)
    
    elif isinstance(event, HandoffSentEvent):
        print(f"\n🔄 Handoff: {event.source} → {event.target}")
    
    elif isinstance(event, RequestInfoEvent):
        print(f"\n💬 Agent requests input")
    
    elif isinstance(event, WorkflowOutputEvent):
        if isinstance(event.output, list):
            for msg in event.output:
                if hasattr(msg, 'text') and msg.text:
                    print(f"\n📤 Output: {msg.text[:200]}...")
        else:
            print(f"\n📤 Output received")


async def stream_workflow(
    query: str,
    workflow_type: Literal["handoff", "concurrent", "sequential"] = "handoff",
    chat_client: AzureOpenAIChatClient | None = None,
) -> AsyncIterator[WorkflowEvent]:
    """Stream events from a Fabric orchestration workflow.
    
    Args:
        query: The user's query to process.
        workflow_type: Type of workflow to use.
        chat_client: Optional chat client.
        
    Yields:
        WorkflowEvent instances as they occur.
    """
    client = chat_client or get_chat_client()
    
    # Build the appropriate workflow
    if workflow_type == "handoff":
        workflow = build_handoff_workflow(client)
    elif workflow_type == "concurrent":
        workflow = build_concurrent_workflow(client)
    elif workflow_type == "sequential":
        workflow = build_sequential_workflow(client)
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    async for event in workflow.run_stream(query):
        yield event


def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Fabric Data Agent Orchestration (Agent Framework)"
    )
    parser.add_argument(
        "--workflow",
        "-w",
        type=str,
        choices=["handoff", "concurrent", "sequential"],
        default="handoff",
        help="Workflow type to use (default: handoff)"
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
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        results = asyncio.run(run_workflow(
            query=args.query,
            workflow_type=args.workflow,
            verbose=not args.quiet
        ))
        
        if not args.quiet:
            print("\n" + "=" * 50)
            print("Final Results:")
            for msg in results:
                if hasattr(msg, 'text'):
                    print(f"  {msg.text}")
    else:
        # Interactive mode
        asyncio.run(run_workflow_interactive(
            workflow_type=args.workflow
        ))


if __name__ == "__main__":
    main()
