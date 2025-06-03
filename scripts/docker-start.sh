#!/bin/bash

echo "🚀 Starting Cashly AI Docker Environment..."

# Check if .env.docker exists
if [ ! -f .env.docker ]; then
    echo "❌ .env.docker not found. Creating from template..."
    cp .env.docker.example .env.docker
    echo "⚠️  Please update .env.docker with your AWS credentials"
    exit 1
fi

# Check if AWS credentials are set
source .env.docker
if [ "$AWS_ACCESS_KEY_ID" == "your_access_key" ]; then
    echo "❌ Please update AWS credentials in .env.docker"
    exit 1
fi

# Start services
echo "📦 Starting Docker services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 5

# Check service health
echo "🔍 Checking service status..."
docker-compose ps

# Test connections
echo "🧪 Testing database connections..."

# Test main database
if docker exec cashly-ai-postgres pg_isready -U cashly_ai > /dev/null 2>&1; then
    echo "✅ Main database is ready"
else
    echo "❌ Main database is not responding"
fi

# Test MLflow database
if docker exec cashly-ai-mlflow-postgres pg_isready -U cashly_ml > /dev/null 2>&1; then
    echo "✅ MLflow database is ready"
else
    echo "❌ MLflow database is not responding"
fi

# Test MLflow API
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ MLflow API is ready"
else
    echo "⏳ MLflow API is starting up..."
fi

echo ""
echo "🎉 Services are running!"
echo ""
echo "📊 Access points:"
echo "  - Main DB: postgresql://cashly_ai:qwerT12321@localhost:5433/cashly_ai_vectors"
echo "  - MLflow DB: postgresql://mlflow_user:mlflow_pass_123@localhost:5434/mlflow_tracking"
echo "  - MLflow UI: http://localhost:5000"
echo ""
echo "💡 Run 'docker-compose logs -f' to view logs"