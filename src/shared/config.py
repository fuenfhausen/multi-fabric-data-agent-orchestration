# Copyright (c) Microsoft. All rights reserved.
"""Configuration management for multi-fabric data agent orchestration."""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: str = Field(
        ...,
        description="Azure OpenAI endpoint URL"
    )
    azure_openai_deployment: str = Field(
        default="gpt-4o",
        description="Azure OpenAI deployment name"
    )
    azure_openai_api_version: str = Field(
        default="2024-08-01-preview",
        description="Azure OpenAI API version"
    )
    
    # Microsoft Fabric Configuration
    fabric_workspace_id: str = Field(
        ...,
        description="Microsoft Fabric workspace GUID"
    )
    fabric_sql_endpoint: str | None = Field(
        default=None,
        description="Fabric SQL endpoint for warehouse queries"
    )
    fabric_lakehouse_name: str | None = Field(
        default=None,
        description="Default lakehouse name"
    )
    fabric_warehouse_name: str | None = Field(
        default=None,
        description="Default warehouse name"
    )
    fabric_eventhouse_name: str | None = Field(
        default=None,
        description="Default eventhouse name"
    )
    
    # Azure Identity Configuration (optional, for service principal)
    azure_tenant_id: str | None = Field(
        default=None,
        description="Microsoft Entra ID tenant ID"
    )
    azure_client_id: str | None = Field(
        default=None,
        description="Service principal client ID"
    )
    azure_client_secret: str | None = Field(
        default=None,
        description="Service principal client secret"
    )
    
    # LangSmith Configuration (optional, for LangGraph tracing)
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing"
    )
    langchain_api_key: str | None = Field(
        default=None,
        description="LangSmith API key"
    )
    langchain_project: str = Field(
        default="fabric-agent-orchestration",
        description="LangSmith project name"
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
