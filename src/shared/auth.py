# Copyright (c) Microsoft. All rights reserved.
"""Authentication utilities for Microsoft Fabric APIs."""

import time
import struct
from typing import Any

import requests
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential
from azure.core.credentials import AccessToken


class FabricAuthenticator:
    """Handles authentication for Microsoft Fabric REST APIs."""
    
    FABRIC_SCOPE = "https://api.fabric.microsoft.com/.default"
    STORAGE_SCOPE = "https://storage.azure.com/.default"
    COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"
    
    def __init__(self, credential: DefaultAzureCredential | None = None):
        """Initialize the authenticator.
        
        Args:
            credential: Azure credential to use. Defaults to DefaultAzureCredential.
        """
        self.credential = credential or DefaultAzureCredential()
        self._token_cache: dict[str, AccessToken] = {}
    
    def _is_token_expired(self, scope: str) -> bool:
        """Check if cached token is expired or missing."""
        if scope not in self._token_cache:
            return True
        # Refresh 5 minutes before expiry
        return self._token_cache[scope].expires_on < time.time() + 300
    
    def get_token(self, scope: str) -> str:
        """Get access token for the specified scope.
        
        Args:
            scope: The token scope to request.
            
        Returns:
            Access token string.
        """
        if self._is_token_expired(scope):
            self._token_cache[scope] = self.credential.get_token(scope)
        return self._token_cache[scope].token
    
    def get_fabric_token(self) -> str:
        """Get access token for Fabric REST API calls."""
        return self.get_token(self.FABRIC_SCOPE)
    
    def get_storage_token(self) -> str:
        """Get access token for OneLake storage operations."""
        return self.get_token(self.STORAGE_SCOPE)
    
    def get_cognitive_token(self) -> str:
        """Get access token for Azure Cognitive Services."""
        return self.get_token(self.COGNITIVE_SCOPE)
    
    def get_headers(self) -> dict[str, str]:
        """Get HTTP headers for Fabric API requests."""
        return {
            "Authorization": f"Bearer {self.get_fabric_token()}",
            "Content-Type": "application/json"
        }
    
    def execute_query(
        self, 
        workspace_id: str, 
        query_type: str, 
        query: str, 
        max_rows: int = 1000,
        item_id: str | None = None
    ) -> dict[str, Any]:
        """Execute a query against Fabric APIs.
        
        Args:
            workspace_id: The Fabric workspace GUID.
            query_type: Type of query ('spark', 'tsql', or 'kql').
            query: The query string to execute.
            max_rows: Maximum rows to return.
            item_id: Optional item ID (lakehouse, warehouse, or eventhouse).
            
        Returns:
            Query results as dictionary.
        """
        base_url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}"
        
        endpoints = {
            "spark": f"{base_url}/lakehouses/{item_id}/query" if item_id else f"{base_url}/lakehouses/query",
            "tsql": f"{base_url}/warehouses/{item_id}/query" if item_id else f"{base_url}/warehouses/query",
            "kql": f"{base_url}/eventhouses/{item_id}/query" if item_id else f"{base_url}/eventhouses/query"
        }
        
        if query_type not in endpoints:
            raise ValueError(f"Unsupported query type: {query_type}. Must be 'spark', 'tsql', or 'kql'.")
        
        response = requests.post(
            endpoints[query_type],
            headers=self.get_headers(),
            json={"query": query, "maxRows": max_rows},
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    
    def list_workspace_items(
        self, 
        workspace_id: str, 
        item_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """List items in a Fabric workspace.
        
        Args:
            workspace_id: The Fabric workspace GUID.
            item_types: Optional filter for item types.
            
        Returns:
            List of workspace items.
        """
        url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/items"
        
        params = {}
        if item_types:
            params["type"] = ",".join(item_types)
        
        response = requests.get(
            url,
            headers=self.get_headers(),
            params=params,
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("value", [])
    
    def get_item_details(
        self, 
        workspace_id: str, 
        item_id: str
    ) -> dict[str, Any]:
        """Get details of a specific workspace item.
        
        Args:
            workspace_id: The Fabric workspace GUID.
            item_id: The item GUID.
            
        Returns:
            Item details.
        """
        url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/items/{item_id}"
        
        response = requests.get(
            url,
            headers=self.get_headers(),
            timeout=60
        )
        response.raise_for_status()
        return response.json()


class OneLakeClient:
    """Client for OneLake data operations."""
    
    def __init__(
        self, 
        workspace_id: str, 
        credential: DefaultAzureCredential | None = None
    ):
        """Initialize the OneLake client.
        
        Args:
            workspace_id: The Fabric workspace GUID.
            credential: Azure credential to use.
        """
        self.workspace_id = workspace_id
        self.credential = credential or DefaultAzureCredential()
        
        # Lazy import to avoid dependency issues
        from azure.storage.filedatalake import DataLakeServiceClient
        
        account_url = "https://onelake.dfs.fabric.microsoft.com"
        self.service_client = DataLakeServiceClient(
            account_url=account_url,
            credential=self.credential
        )
    
    def list_tables(self, lakehouse_name: str) -> list[str]:
        """List all tables in a lakehouse.
        
        Args:
            lakehouse_name: Name of the lakehouse.
            
        Returns:
            List of table names.
        """
        file_system = self.service_client.get_file_system_client(self.workspace_id)
        tables_dir = file_system.get_directory_client(f"{lakehouse_name}/Tables")
        
        return [
            path.name 
            for path in tables_dir.get_paths() 
            if path.is_directory
        ]
    
    def list_files(self, lakehouse_name: str, path: str = "Files") -> list[dict[str, Any]]:
        """List files in a lakehouse directory.
        
        Args:
            lakehouse_name: Name of the lakehouse.
            path: Directory path within the lakehouse.
            
        Returns:
            List of file information.
        """
        file_system = self.service_client.get_file_system_client(self.workspace_id)
        directory = file_system.get_directory_client(f"{lakehouse_name}/{path}")
        
        return [
            {
                "name": path.name,
                "is_directory": path.is_directory,
                "size": path.content_length,
                "last_modified": path.last_modified
            }
            for path in directory.get_paths()
        ]
    
    def describe_table(
        self, 
        lakehouse_name: str, 
        table_name: str
    ) -> dict[str, Any]:
        """Get metadata for a Delta table.
        
        Args:
            lakehouse_name: Name of the lakehouse.
            table_name: Name of the Delta table.
            
        Returns:
            Table metadata including schema information.
        """
        file_system = self.service_client.get_file_system_client(self.workspace_id)
        table_path = f"{lakehouse_name}/Tables/{table_name}"
        
        # Check if _delta_log exists
        delta_log = file_system.get_directory_client(f"{table_path}/_delta_log")
        
        # Get basic table info
        return {
            "name": table_name,
            "lakehouse": lakehouse_name,
            "path": table_path,
            "format": "delta",
            "has_delta_log": delta_log.exists()
        }


class FabricSQLClient:
    """Client for Fabric SQL endpoint queries."""
    
    def __init__(
        self, 
        server: str, 
        database: str, 
        credential: DefaultAzureCredential | None = None
    ):
        """Initialize the SQL client.
        
        Args:
            server: SQL server endpoint.
            database: Database name.
            credential: Azure credential to use.
        """
        self.server = server
        self.database = database
        self.credential = credential or DefaultAzureCredential()
    
    def _get_connection(self):
        """Create authenticated connection to Fabric SQL endpoint."""
        import pyodbc
        
        # Get access token
        token = self.credential.get_token("https://database.windows.net/.default")
        token_bytes = token.token.encode("UTF-16-LE")
        token_struct = struct.pack(f'<I{len(token_bytes)}s', len(token_bytes), token_bytes)
        
        # Connection string for Fabric SQL endpoint
        conn_str = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
        )
        
        conn = pyodbc.connect(conn_str, attrs_before={
            1256: token_struct  # SQL_COPT_SS_ACCESS_TOKEN
        })
        return conn
    
    def execute_query(self, query: str, max_rows: int = 1000) -> list[dict[str, Any]]:
        """Execute a T-SQL query and return results.
        
        Args:
            query: T-SQL query to execute.
            max_rows: Maximum rows to return.
            
        Returns:
            List of result rows as dictionaries.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Add TOP clause if not present and max_rows specified
            if max_rows and "TOP" not in query.upper():
                query = query.replace("SELECT", f"SELECT TOP {max_rows}", 1)
            
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def list_schemas(self) -> list[str]:
        """List all schemas in the database."""
        results = self.execute_query(
            "SELECT schema_name FROM information_schema.schemata ORDER BY schema_name"
        )
        return [row["schema_name"] for row in results]
    
    def list_tables(self, schema: str = "dbo") -> list[str]:
        """List all tables in a schema."""
        results = self.execute_query(
            f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}' ORDER BY table_name"
        )
        return [row["table_name"] for row in results]
    
    def describe_table(self, table_name: str, schema: str = "dbo") -> list[dict[str, Any]]:
        """Get column information for a table."""
        return self.execute_query(f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns 
            WHERE table_schema = '{schema}' AND table_name = '{table_name}'
            ORDER BY ordinal_position
        """)


def get_production_credential() -> ChainedTokenCredential:
    """Get credential chain optimized for production.
    
    Returns:
        ChainedTokenCredential that tries managed identity first.
    """
    return ChainedTokenCredential(
        ManagedIdentityCredential(),
        AzureCliCredential()
    )
