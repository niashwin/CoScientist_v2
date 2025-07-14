#!/bin/bash

# CoScientist_v2 Production Deployment Script
# This script handles the complete deployment process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"
BACKUP_DIR="./backups"
LOG_DIR="./logs"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking deployment requirements..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f "$ENV_FILE" ]; then
        log_warning ".env file not found. Creating from production template..."
        cp env.production "$ENV_FILE"
        log_error "Please edit $ENV_FILE with your actual configuration values before continuing."
        exit 1
    fi
    
    log_success "Requirements check passed"
}

create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "./ops/prometheus"
    mkdir -p "./ops/grafana/provisioning"
    
    log_success "Directories created"
}

backup_data() {
    if [ "$1" = "--skip-backup" ]; then
        log_warning "Skipping backup as requested"
        return
    fi
    
    log_info "Creating backup of existing data..."
    
    if docker-compose -f "$COMPOSE_FILE" ps -q postgres > /dev/null 2>&1; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        BACKUP_FILE="$BACKUP_DIR/postgres_backup_$TIMESTAMP.sql"
        
        docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U coscientist coscientist > "$BACKUP_FILE"
        log_success "Database backup created: $BACKUP_FILE"
    else
        log_info "No existing database found, skipping backup"
    fi
}

deploy_application() {
    log_info "Deploying CoScientist_v2..."
    
    # Pull latest images
    log_info "Pulling latest images..."
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Build and start services
    log_info "Building and starting services..."
    docker-compose -f "$COMPOSE_FILE" up --build -d
    
    log_success "Application deployed successfully"
}

wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for backend
    for i in {1..30}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Backend is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Backend failed to start within 5 minutes"
            exit 1
        fi
        sleep 10
    done
    
    # Wait for frontend
    for i in {1..30}; do
        if curl -f http://localhost:80 > /dev/null 2>&1; then
            log_success "Frontend is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Frontend failed to start within 5 minutes"
            exit 1
        fi
        sleep 10
    done
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Check all services
    FAILED_SERVICES=()
    
    # Backend health check
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        FAILED_SERVICES+=("backend")
    fi
    
    # Frontend health check
    if ! curl -f http://localhost:80 > /dev/null 2>&1; then
        FAILED_SERVICES+=("frontend")
    fi
    
    # Database health check
    if ! docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U coscientist > /dev/null 2>&1; then
        FAILED_SERVICES+=("postgres")
    fi
    
    # Redis health check
    if ! docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null 2>&1; then
        FAILED_SERVICES+=("redis")
    fi
    
    if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
        log_success "All health checks passed"
    else
        log_error "Health checks failed for: ${FAILED_SERVICES[*]}"
        exit 1
    fi
}

show_deployment_info() {
    log_success "🎉 CoScientist_v2 deployed successfully!"
    echo ""
    echo "📱 Application URLs:"
    echo "   Frontend: http://localhost:80"
    echo "   API: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    echo "   Health Check: http://localhost:8000/health"
    echo ""
    echo "📊 Management Commands:"
    echo "   View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "   Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "   Restart services: docker-compose -f $COMPOSE_FILE restart"
    echo "   View status: docker-compose -f $COMPOSE_FILE ps"
    echo ""
    echo "🔧 Monitoring (if enabled):"
    echo "   Grafana: http://localhost:3000"
    echo "   Prometheus: http://localhost:9090"
    echo ""
}

# Main deployment process
main() {
    log_info "Starting CoScientist_v2 deployment..."
    
    check_requirements
    create_directories
    backup_data "$1"
    deploy_application
    wait_for_services
    run_health_checks
    show_deployment_info
    
    log_success "Deployment completed successfully!"
}

# Handle command line arguments
case "$1" in
    --help|-h)
        echo "CoScientist_v2 Deployment Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --skip-backup    Skip database backup before deployment"
        echo "  --help, -h       Show this help message"
        echo ""
        exit 0
        ;;
    *)
        main "$1"
        ;;
esac 