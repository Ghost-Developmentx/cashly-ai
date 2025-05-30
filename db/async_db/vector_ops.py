"""
Async vector operations for pgvector.
Handles vector similarity searches and indexing.
"""

import logging
from typing import List, Optional, Dict, Any
import asyncpg
import numpy as np

logger = logging.getLogger(__name__)


class AsyncVectorOperations:
    """
    AsyncVectorOperations is a utility class for performing various operations on vector embeddings
    stored in a PostgreSQL database, using the pgvector extension. It supports creating vector extensions,
    building vector similarity indexes, inserting vectors with metadata, searching for similar vectors, and
    updating metadata of existing vector records.

    This class is designed for asynchronous operations and relies on asyncpg as the database driver. It is
    specifically crafted to work with OpenAI embedding vectors of fixed dimensionality.

    Attributes
    ----------
    pool : asyncpg.Pool
        A connection pool for asynchronous interaction with the PostgreSQL database.
    vector_dimensions : int
        The dimensionality of the OpenAI embedding vectors.
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.vector_dimensions = 1536  # OpenAI embedding dimensions

    async def create_vector_extension(self) -> bool:
        """Ensure the pgvector extension is created."""
        try:
            async with self.pool.acquire() as conn:
                # Check if the extension exists
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
                )

                if result > 0:
                    logger.info("✅ pgvector extension already exists")
                    return True

                # Create extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                logger.info("✅ Created pgvector extension")
                return True

        except Exception as e:
            logger.error(f"Failed to create vector extension: {e}")
            return False

    async def create_vector_index(
        self,
        table_name: str,
        column_name: str,
        index_type: str = "ivfflat",
        lists: int = 100,
    ) -> bool:
        """
        Create a vector similarity index.

        Args:
            table_name: Table containing vectors
            column_name: Column containing vectors
            index_type: Index type (ivfflat or hnsw)
            lists: Number of lists for ivfflat

        Returns:
            True if successful
        """
        try:
            index_name = f"idx_{table_name}_{column_name}_vector"

            async with self.pool.acquire() as conn:
                if index_type == "ivfflat":
                    query = f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {table_name}
                        USING ivfflat ({column_name} vector_cosine_ops)
                        WITH (lists = {lists})
                    """
                elif index_type == "hnsw":
                    query = f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {table_name}
                        USING hnsw ({column_name} vector_cosine_ops)
                    """
                else:
                    raise ValueError(f"Unknown index type: {index_type}")

                await conn.execute(query)
                logger.info(
                    f"✅ Created {index_type} index on {table_name}.{column_name}"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            return False

    async def insert_vector(
        self, table_name: str, vector: List[float], metadata: Dict[str, Any]
    ) -> Optional[int]:
        """
        Insert a vector with metadata.

        Args:
            table_name: Target table
            vector: Embedding vector
            metadata: Associated metadata

        Returns:
            Inserted record ID
        """
        try:
            # Prepare vector as an array
            vector_array = np.array(vector, dtype=np.float32)

            # Convert vector to string format for asyncpg
            vector_str = f"[{','.join(map(str, vector_array.tolist()))}]"

            # Build an insert query dynamically
            columns = ["embedding"] + list(metadata.keys())
            placeholders = ["$1::vector"] + [f"${i+2}" for i in range(len(metadata))]

            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                RETURNING id
            """

            values = [vector_str] + list(metadata.values())

            async with self.pool.acquire() as conn:
                record_id = await conn.fetchval(query, *values)
                return record_id

        except Exception as e:
            logger.error(f"Failed to insert vector: {e}")
            return None

    async def search_similar_vectors(
        self,
        table_name: str,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.8,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            table_name: Table to search
            query_vector: Query embedding
            limit: Maximum results
            threshold: Similarity threshold
            filters: Additional WHERE conditions

        Returns:
            List of similar records
        """
        try:
            vector_array = np.array(query_vector, dtype=np.float32)

            # Convert vector to string format for asyncpg
            vector_str = f"[{','.join(map(str, vector_array.tolist()))}]"

            # Build a query with filters
            base_query = f"""
                SELECT *,
                       1 - (embedding <=> $1::vector) as similarity
                FROM {table_name}
                WHERE 1 - (embedding <=> $1::vector) >= $2
            """

            params = [vector_str, threshold]
            param_count = 2

            # Add filters
            if filters:
                for key, value in filters.items():
                    param_count += 1
                    base_query += f" AND {key} = ${param_count}"
                    params.append(value)

            base_query += f" ORDER BY similarity DESC LIMIT ${param_count + 1}"
            params.append(limit)

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(base_query, *params)

                results = []
                for row in rows:
                    result = dict(row)
                    # Convert embedding to list if needed
                    if "embedding" in result:
                        del result["embedding"]  # Remove embedding from result
                    results.append(result)

                return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def update_vector_metadata(
        self, table_name: str, record_id: int, metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for a vector record."""
        try:
            set_clauses = [f"{key} = ${i+2}" for i, key in enumerate(metadata.keys())]
            query = f"""
                UPDATE {table_name}
                SET {', '.join(set_clauses)}
                WHERE id = $1
            """

            values = [record_id] + list(metadata.values())

            async with self.pool.acquire() as conn:
                await conn.execute(query, *values)
                return True

        except Exception as e:
            logger.error(f"Failed to update vector metadata: {e}")
            return False
