#!/bin/bash

# AI Co-Scientist Development Startup Script

set -e

echo "🚀 Starting AI Co-Scientist Development Environment"

# Check if .env file exists
if [ ! -f "../.env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp ../env.sample ../.env
    echo "📝 Please edit ../.env with your API keys before continuing"
    echo "   Required keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, SERPER_API_KEY, SEMANTIC_SCHOLAR_API_KEY, PERPLEXITY_API_KEY"
    exit 1
fi

# Load environment variables
source ../.env

# Determine which profiles to use
PROFILES=""

if [ "$ENABLE_PERSISTENCE" = "true" ]; then
    echo "📊 Enabling persistence (PostgreSQL + Chroma)"
    PROFILES="$PROFILES --profile persistence"
fi

if [ "$OTEL_ENABLED" = "true" ]; then
    echo "📈 Enabling observability (OpenTelemetry)"
    PROFILES="$PROFILES --profile observability"
fi

# Build and start services
echo "🔨 Building and starting services..."
docker compose -f docker-compose.yml $PROFILES up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🏥 Checking service health..."

# Check backend health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is healthy"
else
    echo "❌ Backend API is not responding"
fi

# Check frontend health
if curl -f http://localhost:5173 > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend is not responding"
fi

# Check Redis
if docker compose -f docker-compose.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
fi

# Check PostgreSQL if enabled
if [ "$ENABLE_PERSISTENCE" = "true" ]; then
    if docker compose -f docker-compose.yml exec -T postgres pg_isready > /dev/null 2>&1; then
        echo "✅ PostgreSQL is healthy"
    else
        echo "❌ PostgreSQL is not responding"
    fi
fi

echo ""
echo "🎉 AI Co-Scientist Development Environment is ready!"
echo ""
echo "📱 Frontend: http://localhost:5173"
echo "🔧 API Docs: http://localhost:8000/docs"
echo "🏥 Health Check: http://localhost:8000/health"
echo ""
echo "📊 Services:"
docker compose -f docker-compose.yml ps
echo ""
echo "📝 Logs: docker compose -f docker-compose.yml logs -f"
echo "🛑 Stop: docker compose -f docker-compose.yml down" 