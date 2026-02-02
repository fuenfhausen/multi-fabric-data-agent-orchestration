# Copyright (c) Microsoft. All rights reserved.
"""Tool definitions for Microsoft Agent Framework Fabric agents."""

from typing import Annotated

from agent_framework import ai_function

from ..shared.auth import FabricAuthenticator, OneLakeClient, FabricSQLClient
from ..shared.config import get_settings

# Initialize authenticator
_fabric_auth = FabricAuthenticator()


# =============================================================================
# Lakehouse Tools
# =============================================================================

@ai_function
def execute_spark_sql(
    query: Annotated[str, "The Spark SQL query to execute"],
    max_rows: Annotated[int, "Maximum rows to return"] = 1000,
    lakehouse_id: Annotated[str | None, "Optional lakehouse ID"] = None
) -> dict:
    """Execute a Spark SQL query against a Fabric Lakehouse.
    
    Use this tool to query Delta tables, run aggregations, and explore data
    stored in the lakehouse. The query should be valid Spark SQL syntax.
    
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


@ai_function
def list_lakehouse_tables(
    lakehouse_name: Annotated[str | None, "Name of the lakehouse"] = None
) -> list[str]:
    """List all Delta tables in a Fabric Lakehouse.
    
    Returns the names of all tables available in the specified lakehouse.
    Use this before querying to discover available tables.
    """
    settings = get_settings()
    workspace_id = settings.fabric_workspace_id
    lakehouse = lakehouse_name or settings.fabric_lakehouse_name
    
    if not lakehouse:
        return {"error": "No lakehouse name provided and no default configured"}
    
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


@ai_function
def describe_delta_table(
    table_name: Annotated[str, "Name of the Delta table"],
    lakehouse_name: Annotated[str | None, "Name of the lakehouse"] = None
) -> dict:
    """Get schema and metadata for a Delta table in the lakehouse.
    
    Returns column names, data types, and table properties.
    Use this to understand table structure before writing queries.
    """
    settings = get_settings()
    workspace_id = settings.fabric_workspace_id
    lakehouse = lakehouse_name or settings.fabric_lakehouse_name
    
    if not lakehouse:
        return {"error": "No lakehouse name provided and no default configured"}
    
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

@ai_function
def execute_tsql(
    query: Annotated[str, "The T-SQL query to execute"],
    max_rows: Annotated[int, "Maximum rows to return"] = 1000,
    warehouse_id: Annotated[str | None, "Optional warehouse ID"] = None
) -> dict:
    """Execute a T-SQL query against a Fabric Data Warehouse.
    
    Use this tool to query warehouse tables, run analytics, and explore data.
    The query should be valid T-SQL syntax.
    
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


@ai_function
def list_warehouse_schemas() -> dict:
    """List all schemas in the Fabric Data Warehouse.
    
    Returns the names of all schemas in the warehouse.
    Use this to discover the database structure.
    """
    settings = get_settings()
    
    if not settings.fabric_sql_endpoint:
        return {"error": "No SQL endpoint configured"}
    
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


@ai_function
def list_warehouse_tables(
    schema: Annotated[str, "Schema name"] = "dbo"
) -> dict:
    """List all tables in a warehouse schema.
    
    Returns the names of all tables in the specified schema.
    Default schema is 'dbo'.
    """
    settings = get_settings()
    
    if not settings.fabric_sql_endpoint:
        return {"error": "No SQL endpoint configured"}
    
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


@ai_function
def describe_warehouse_table(
    table_name: Annotated[str, "Name of the table"],
    schema: Annotated[str, "Schema name"] = "dbo"
) -> dict:
    """Get column information for a warehouse table.
    
    Returns column names, data types, and constraints.
    """
    settings = get_settings()
    
    if not settings.fabric_sql_endpoint:
        return {"error": "No SQL endpoint configured"}
    
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

@ai_function
def execute_kql(
    query: Annotated[str, "The KQL query to execute"],
    database_name: Annotated[str | None, "KQL database name"] = None,
    max_rows: Annotated[int, "Maximum rows to return"] = 1000,
    eventhouse_id: Annotated[str | None, "Optional eventhouse ID"] = None
) -> dict:
    """Execute a KQL query against a Fabric Eventhouse.
    
    Use this tool to query streaming data, analyze time-series, and explore
    real-time data. The query should be valid Kusto Query Language syntax.
    
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


@ai_function
def list_eventhouse_databases() -> dict:
    """List all KQL databases in the Fabric Eventhouse.
    
    Returns the names of all databases available for querying.
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


@ai_function
def get_stream_status(
    stream_name: Annotated[str, "Name of the event stream"]
) -> dict:
    """Get status of a real-time data stream.
    
    Returns stream health, throughput, and latency information.
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
