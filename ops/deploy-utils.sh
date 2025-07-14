#!/bin/bash

# CoScientist_v2 Deployment Utilities
# Collection of utility functions for deployment management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Scale services
scale_services() {
    local service=$1
    local replicas=$2
    
    if [ -z "$service" ] || [ -z "$replicas" ]; then
        log_error "Usage: scale_services <service> <replicas>"
        return 1
    fi
    
    log_info "Scaling $service to $replicas replicas..."
    docker-compose -f "$COMPOSE_FILE" up -d --scale "$service=$replicas"
    log_success "Scaled $service to $replicas replicas"
}

# Update single service
update_service() {
    local service=$1
    
    if [ -z "$service" ]; then
        log_error "Usage: update_service <service>"
        return 1
    fi
    
    log_info "Updating $service..."
    docker-compose -f "$COMPOSE_FILE" up -d --no-deps --build "$service"
    log_success "Updated $service"
}

# Backup database
backup_database() {
    local backup_name=${1:-"manual_$(date +%Y%m%d_%H%M%S)"}
    local backup_file="./backups/postgres_backup_$backup_name.sql"
    
    log_info "Creating database backup: $backup_file"
    mkdir -p ./backups
    
    docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U coscientist coscientist > "$backup_file"
    log_success "Database backup created: $backup_file"
}

# Restore database
restore_database() {
    local backup_file=$1
    
    if [ -z "$backup_file" ] || [ ! -f "$backup_file" ]; then
        log_error "Usage: restore_database <backup_file>"
        log_error "Available backups:"
        ls -la ./backups/*.sql 2>/dev/null || log_error "No backups found"
        return 1
    fi
    
    log_warning "This will replace the current database. Are you sure? (y/N)"
    read -r confirm
    if [[ $confirm != [yY] ]]; then
        log_info "Restore cancelled"
        return 0
    fi
    
    log_info "Restoring database from: $backup_file"
    docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U coscientist -d coscientist < "$backup_file"
    log_success "Database restored successfully"
}

# View logs
view_logs() {
    local service=${1:-""}
    local lines=${2:-"100"}
    
    if [ -z "$service" ]; then
        log_info "Showing logs for all services (last $lines lines)..."
        docker-compose -f "$COMPOSE_FILE" logs --tail="$lines" -f
    else
        log_info "Showing logs for $service (last $lines lines)..."
        docker-compose -f "$COMPOSE_FILE" logs --tail="$lines" -f "$service"
    fi
}

# System status
system_status() {
    log_info "System Status:"
    echo ""
    
    # Service status
    echo "📊 Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    
    # Resource usage
    echo "💻 Resource Usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"
    echo ""
    
    # Health checks
    echo "🏥 Health Checks:"
    
    # Backend
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend: Healthy"
    else
        echo "❌ Backend: Unhealthy"
    fi
    
    # Frontend
    if curl -f http://localhost:80 > /dev/null 2>&1; then
        echo "✅ Frontend: Healthy"
    else
        echo "❌ Frontend: Unhealthy"
    fi
    
    # Database
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U coscientist > /dev/null 2>&1; then
        echo "✅ PostgreSQL: Healthy"
    else
        echo "❌ PostgreSQL: Unhealthy"
    fi
    
    # Redis
    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis: Healthy"
    else
        echo "❌ Redis: Unhealthy"
    fi
    
    echo ""
}

# Clean up old images and containers
cleanup() {
    log_info "Cleaning up old Docker images and containers..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful with this)
    log_warning "Do you want to remove unused volumes? This may delete data! (y/N)"
    read -r confirm
    if [[ $confirm == [yY] ]]; then
        docker volume prune -f
    fi
    
    log_success "Cleanup completed"
}

# Rolling update
rolling_update() {
    log_info "Performing rolling update..."
    
    # Update backend first
    log_info "Updating backend..."
    docker-compose -f "$COMPOSE_FILE" up -d --no-deps --build backend
    
    # Wait for backend to be healthy
    for i in {1..30}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Backend updated successfully"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Backend update failed"
            return 1
        fi
        sleep 10
    done
    
    # Update frontend
    log_info "Updating frontend..."
    docker-compose -f "$COMPOSE_FILE" up -d --no-deps --build frontend
    
    # Wait for frontend to be healthy
    for i in {1..30}; do
        if curl -f http://localhost:80 > /dev/null 2>&1; then
            log_success "Frontend updated successfully"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Frontend update failed"
            return 1
        fi
        sleep 10
    done
    
    log_success "Rolling update completed successfully"
}

# Main command dispatcher
case "$1" in
    scale)
        scale_services "$2" "$3"
        ;;
    update)
        update_service "$2"
        ;;
    backup)
        backup_database "$2"
        ;;
    restore)
        restore_database "$2"
        ;;
    logs)
        view_logs "$2" "$3"
        ;;
    status)
        system_status
        ;;
    cleanup)
        cleanup
        ;;
    rolling-update)
        rolling_update
        ;;
    *)
        echo "CoScientist_v2 Deployment Utilities"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  scale <service> <replicas>    Scale a service to specified replicas"
        echo "  update <service>              Update a single service"
        echo "  backup [name]                 Create database backup"
        echo "  restore <backup_file>         Restore database from backup"
        echo "  logs [service] [lines]        View logs (default: all services, 100 lines)"
        echo "  status                        Show system status"
        echo "  cleanup                       Clean up old Docker resources"
        echo "  rolling-update                Perform rolling update"
        echo ""
        echo "Examples:"
        echo "  $0 scale backend 3"
        echo "  $0 update frontend"
        echo "  $0 backup production_backup"
        echo "  $0 logs backend 50"
        echo "  $0 status"
        echo ""
        exit 1
        ;;
esac 