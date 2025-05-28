"""
Migration to create conversation embeddings table.
"""

from sqlalchemy import text
from db.connection import DatabaseConnection
from db.vector_extension import VectorExtensionManager
from models.conversation_embedding import Base


def upgrade(db_connection: DatabaseConnection):
    """Create conversation embeddings table."""
    # Ensure pgvector extension is installed
    with db_connection.get_session() as session:
        if not VectorExtensionManager.setup_extension(session):
            raise RuntimeError("Failed to setup pgvector extension")

    # Create all tables
    Base.metadata.create_all(bind=db_connection.engine)

    # Add vector similarity index
    with db_connection.get_session() as session:
        session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_embedding_vector "
                "ON conversation_embeddings "
                "USING ivfflat (embedding vector_cosine_ops) "
                "WITH (lists = 100)"
            )
        )
        session.commit()

    print("Migration completed: conversation_embeddings table created")


def downgrade(db_connection: DatabaseConnection):
    """Drop conversation embeddings table."""
    Base.metadata.drop_all(bind=db_connection.engine)
    print("Migration rolled back: conversation_embeddings table dropped")
