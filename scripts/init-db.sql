-- Create vector extension in template database
\c template1
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector extension in our database
\c cashly_ai_vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';