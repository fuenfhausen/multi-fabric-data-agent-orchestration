# Copyright (c) Microsoft. All rights reserved.
"""Tool definitions for LangGraph Fabric agents."""

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..shared.auth import FabricAuthenticator, OneLakeClient, FabricSQLClient
from ..shared.config import get_settings

# Initialize authenticator
_fabric_auth = FabricAuthenticator()


# =============================================================================
# Pydantic Input Schemas
# =============================================================================

class SparkQueryInput(BaseModel):
    """Input schema for Spark SQL query execution."""
    query: str = Field(description="The Spark SQL query to execute")
    max_rows: int = Field(default=1000, description="Maximum rows to return")
    lakehouse_id: str | None = Field(default=None, description="Optional lakehouse ID")


class TSQLQueryInput(BaseModel):
    """Input schema for T-SQL query execution."""
    query: str = Field(description="The T-SQL query to execute")
    max_rows: int = Field(default=1000, description="Maximum rows to return")
    warehouse_id: str | None = Field(default=None, description="Optional warehouse ID")


class KQLQueryInput(BaseModel):
    """Input schema for KQL query execution."""
    query: str = Field(description="The KQL query to execute")
    database_name: str | None = Field(default=None, description="KQL database name")
    max_rows: int = Field(default=1000, description="Maximum rows to return")
    eventhouse_id: str | None = Field(default=None, description="Optional eventhouse ID")


# =============================================================================
# Lakehouse Tools
# =============================================================================

@tool
def execute_spark_sql(
    query: str,
    max_rows: int = 1000,
    lakehouse_id: str | None = None
) -> dict[str, Any]:
    """Execute a Spark SQL query against a Fabric Lakehouse.
    
    Use this tool to query Delta tables, run aggregations, and explore data
    stored in the lakehouse. The query should be valid Spark SQL syntax.
    
    Args:
        query: The Spark SQL query to execute
        max_rows: Maximum rows to return (default 1000)
        lakehouse_id: Optional lakehouse ID
    
    Returns:
        Query results with success status, data, and metadata
    
    Examples:
        - SELECT * FROM customers LIMIT 10
        - SELECT product, SUM(revenue) FROM sales GROUP BY product
        - DESCRIBE TABLE orders
    """
    settings = get_settings()
    workspace_id = settings.fabric_workspace_id
    
    try:
        result = _fabric_auth.execute_query(
            workspace_id=workspace_id,
            query_type="spark",
            query=query,
            max_rows=max_rows,
            item_id=lakehouse_id
        )
        return {
            "success": True,
            "data": result,
            "query": query,
            "rows_returned": len(result.get("data", []))
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


@tool
def list_lakehouse_tables(
    lakehouse_name: str | None = None
) -> dict[str, Any]:
    """List all Delta tables in a Fabric Lakehouse.
    
    Returns the names of all tables available in the specified lakehouse.
    Use this before querying to discover available tables.
    
    Args:
        lakehouse_name: Name of the lakehouse (uses default if not provided)
    
    Returns:
        List of table names with count
    """
    settings = get_settings()
    workspace_id = settings.fabric_workspace_id
    lakehouse = lakehouse_name or settings.fabric_lakehouse_name
    
    if not lakehouse:
        return {"success": False, "error": "No lakehouse name provided and no default configured"}
    
    try:
        client = OneLakeClient(workspace_id)
        tables = client.list_tables(lakehouse)
        return {
            "success": True,
            "lakehouse": lakehouse,
            "tables": tables,
            "count": len(tables)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "lakehouse": lakehouse
        }


@tool
def describe_delta_table(
    table_name: str,
    lakehouse_name: str | None = None
) -> dict[str, Any]:
    """Get schema and metadata for a Delta table in the lakehouse.
    
    Returns column names, data types, and table properties.
    Use this to understand table structure before writing queries.
    
    Args:
        table_name: Name of the Delta table
        lakehouse_name: Name of the lakehouse (uses default if not provided)
    
    Returns:
        Table metadata including schema information
    """
    settings = get_settings()
    workspace_id = settings.fabric_workspace_id
    lakehouse = lakehouse_name or settings.fabric_lakehouse_name
    
    if not lakehouse:
        return {"success": False, "error": "No lakehouse name provided and no default configured"}
    
    try:
        client = OneLakeClient(workspace_id)
        metadata = client.describe_table(lakehouse, table_name)
        return {
            "success": True,
            **metadata
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "table_name": table_name
        }


# =============================================================================
# Warehouse Tools
# =============================================================================

@tool
def execute_tsql(
    query: str,
    max_rows: int = 1000,
    warehouse_id: str | None = None
) -> dict[str, Any]:
    """Execute a T-SQL query against a Fabric Data Warehouse.
    
    Use this tool to query warehouse tables, run analytics, and explore data.
    The query should be valid T-SQL syntax.
    
    Args:
        query: The T-SQL query to execute
        max_rows: Maximum rows to return (default 1000)
        warehouse_id: Optional warehouse ID
    
    Returns:
        Query results with success status, data, and metadata
    
    Examples:
        - SELECT TOP 10 * FROM dbo.customers
        - SELECT product, SUM(revenue) FROM dbo.sales GROUP BY product
        - EXEC sp_help 'dbo.orders'
    """
    settings = get_settings()
    workspace_id = settings.fabric_workspace_id
    
    try:
        result = _fabric_auth.execute_query(
            workspace_id=workspace_id,
            query_type="tsql",
            query=query,
            max_rows=max_rows,
            item_id=warehouse_id
        )
        return {
            "success": True,
            "data": result,
            "query": query,
            "rows_returned": len(result.get("data", []))
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


@tool
def list_warehouse_schemas() -> dict[str, Any]:
    """List all schemas in the Fabric Data Warehouse.
    
    Returns the names of all schemas in the warehouse.
    Use this to discover the database structure.
    
    Returns:
        List of schema names with count
    """
    settings = get_settings()
    
    if not settings.fabric_sql_endpoint:
        return {"success": False, "error": "No SQL endpoint configured"}
    
    try:
        client = FabricSQLClient(
            server=settings.fabric_sql_endpoint,
            database=settings.fabric_warehouse_name or "warehouse"
        )
        schemas = client.list_schemas()
        return {
            "success": True,
            "schemas": schemas,
            "count": len(schemas)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
def list_warehouse_tables(schema: str = "dbo") -> dict[str, Any]:
    """List all tables in a warehouse schema.
    
    Returns the names of all tables in the specified schema.
    
    Args:
        schema: Schema name (default is 'dbo')
    
    Returns:
        List of table names with count
    """
    settings = get_settings()
    
    if not settings.fabric_sql_endpoint:
        return {"success": False, "error": "No SQL endpoint configured"}
    
    try:
        client = FabricSQLClient(
            server=settings.fabric_sql_endpoint,
            database=settings.fabric_warehouse_name or "warehouse"
        )
        tables = client.list_tables(schema)
        return {
            "success": True,
            "schema": schema,
            "tables": tables,
            "count": len(tables)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "schema": schema
        }


@tool
def describe_warehouse_table(
    table_name: str,
    schema: str = "dbo"
) -> dict[str, Any]:
    """Get column information for a warehouse table.
    
    Returns column names, data types, and constraints.
    
    Args:
        table_name: Name of the table
        schema: Schema name (default is 'dbo')
    
    Returns:
        Column information with data types
    """
    settings = get_settings()
    
    if not settings.fabric_sql_endpoint:
        return {"success": False, "error": "No SQL endpoint configured"}
    
    try:
        client = FabricSQLClient(
            server=settings.fabric_sql_endpoint,
            database=settings.fabric_warehouse_name or "warehouse"
        )
        columns = client.describe_table(table_name, schema)
        return {
            "success": True,
            "table": f"{schema}.{table_name}",
            "columns": columns,
            "count": len(columns)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "table": f"{schema}.{table_name}"
        }


# =============================================================================
# Real-Time Intelligence Tools
# =============================================================================

@tool
def execute_kql(
    query: str,
    database_name: str | None = None,
    max_rows: int = 1000,
    eventhouse_id: str | None = None
) -> dict[str, Any]:
    """Execute a KQL query against a Fabric Eventhouse.
    
    Use this tool to query streaming data, analyze time-series, and explore
    real-time data. The query should be valid Kusto Query Language syntax.
    
    Args:
        query: The KQL query to execute
        database_name: KQL database name
        max_rows: Maximum rows to return (default 1000)
        eventhouse_id: Optional eventhouse ID
    
    Returns:
        Query results with success status, data, and metadata
    
    Examples:
        - Events | take 10
        - Events | summarize count() by bin(timestamp, 1h)
        - Events | where timestamp > ago(1d) | top 100 by value
    """
    settings = get_settings()
    workspace_id = settings.fabric_workspace_id
    
    try:
        result = _fabric_auth.execute_query(
            workspace_id=workspace_id,
            query_type="kql",
            query=query,
            max_rows=max_rows,
            item_id=eventhouse_id
        )
        return {
            "success": True,
            "data": result,
            "query": query,
            "database": database_name,
            "rows_returned": len(result.get("data", []))
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


@tool
def list_eventhouse_databases() -> dict[str, Any]:
    """List all KQL databases in the Fabric Eventhouse.
    
    Returns the names of all databases available for querying.
    
    Returns:
        List of database names with count
    """
    settings = get_settings()
    workspace_id = settings.fabric_workspace_id
    
    try:
        items = _fabric_auth.list_workspace_items(
            workspace_id=workspace_id,
            item_types=["KQLDatabase"]
        )
        databases = [item["displayName"] for item in items]
        return {
            "success": True,
            "databases": databases,
            "count": len(databases)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
def get_stream_status(stream_name: str) -> dict[str, Any]:
    """Get status of a real-time data stream.
    
    Returns stream health, throughput, and latency information.
    
    Args:
        stream_name: Name of the event stream
    
    Returns:
        Stream status and details
    """
    settings = get_settings()
    workspace_id = settings.fabric_workspace_id
    
    try:
        items = _fabric_auth.list_workspace_items(
            workspace_id=workspace_id,
            item_types=["Eventstream"]
        )
        
        stream = next(
            (item for item in items if item["displayName"] == stream_name),
            None
        )
        
        if stream:
            details = _fabric_auth.get_item_details(workspace_id, stream["id"])
            return {
                "success": True,
                "stream_name": stream_name,
                "status": "active",
                "details": details
            }
        else:
            return {
                "success": False,
                "error": f"Stream '{stream_name}' not found"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stream_name": stream_name
        }


# =============================================================================
# Tool Collections
# =============================================================================

LAKEHOUSE_TOOLS = [
    execute_spark_sql,
    list_lakehouse_tables,
    describe_delta_table,
]

WAREHOUSE_TOOLS = [
    execute_tsql,
    list_warehouse_schemas,
    list_warehouse_tables,
    describe_warehouse_table,
]

REALTIME_TOOLS = [
    execute_kql,
    list_eventhouse_databases,
    get_stream_status,
]

ALL_TOOLS = LAKEHOUSE_TOOLS + WAREHOUSE_TOOLS + REALTIME_TOOLS
