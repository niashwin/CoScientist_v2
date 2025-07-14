# CoScientist_v2 Deployment Guide

This guide provides comprehensive instructions for deploying CoScientist_v2 in various environments.

## 🚀 Quick Start

### Prerequisites

- Docker (v20.10+)
- Docker Compose (v2.0+)
- 8GB+ RAM recommended
- 20GB+ disk space

### 1. Clone and Configure

```bash
git clone https://github.com/niashwin/CoScientist_v2.git
cd CoScientist_v2
cp env.production .env
```

### 2. Configure Environment

Edit `.env` file with your API keys and configuration:

```bash
# Required API keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Database password
POSTGRES_PASSWORD=your_secure_postgres_password_here
```

### 3. Deploy

```bash
./deploy.sh
```

## 📋 Deployment Options

### Option 1: All-in-One Container (Recommended for Small Deployments)

Single container with both frontend and backend:

```bash
docker-compose -f docker-compose.prod.yml up -d coscientist postgres redis chroma
```

### Option 2: Separate Services (Recommended for Production)

Deploy frontend and backend as separate services:

```bash
docker-compose -f docker-compose.prod.yml --profile separate up -d
```

### Option 3: With Monitoring

Include Prometheus and Grafana:

```bash
docker-compose -f docker-compose.prod.yml --profile separate --profile monitoring up -d
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | - | ✅ |
| `OPENAI_API_KEY` | OpenAI API key | - | ✅ |
| `SERPER_API_KEY` | Serper web search API key | - | ✅ |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API key | - | ✅ |
| `PERPLEXITY_API_KEY` | Perplexity API key | - | ✅ |
| `POSTGRES_PASSWORD` | PostgreSQL password | - | ✅ |
| `SECRET_KEY` | Application secret key | - | ✅ |
| `WORKER_PROCESSES` | Number of worker processes | 4 | ❌ |
| `MEMORY_LIMIT` | Container memory limit | 8G | ❌ |
| `CPU_LIMIT` | Container CPU limit | 4 | ❌ |

### Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Backend | 2-4 cores | 3-6GB | 5GB |
| Frontend | 0.5-1 core | 256MB-512MB | 1GB |
| PostgreSQL | 1-2 cores | 1-2GB | 10GB |
| Redis | 0.5-1 core | 512MB-1GB | 2GB |
| Chroma | 1-2 cores | 2-4GB | 5GB |

## 🛠️ Management Commands

### Using Deploy Utilities

```bash
# View system status
./ops/deploy-utils.sh status

# View logs
./ops/deploy-utils.sh logs [service] [lines]

# Scale services
./ops/deploy-utils.sh scale backend 3

# Update single service
./ops/deploy-utils.sh update frontend

# Backup database
./ops/deploy-utils.sh backup

# Restore database
./ops/deploy-utils.sh restore ./backups/backup_file.sql

# Rolling update
./ops/deploy-utils.sh rolling-update

# Cleanup old resources
./ops/deploy-utils.sh cleanup
```

### Direct Docker Compose Commands

```bash
# View service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f [service]

# Restart services
docker-compose -f docker-compose.prod.yml restart [service]

# Stop all services
docker-compose -f docker-compose.prod.yml down

# Update and restart
docker-compose -f docker-compose.prod.yml up -d --build
```

## 🏥 Health Checks

### Automated Health Checks

The deployment includes built-in health checks:

- **Backend**: `http://localhost:8000/health`
- **Frontend**: `http://localhost:80/`
- **PostgreSQL**: `pg_isready` command
- **Redis**: `redis-cli ping`
- **Chroma**: `http://localhost:8000/api/v1/heartbeat`

### Manual Health Verification

```bash
# Check all services
curl http://localhost:8000/health
curl http://localhost:80/

# Check database
docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U coscientist

# Check Redis
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping
```

## 📊 Monitoring

### Grafana Dashboard

If monitoring is enabled:

1. Access Grafana: `http://localhost:3000`
2. Login: `admin` / `your_grafana_password`
3. Import CoScientist dashboard

### Prometheus Metrics

Access Prometheus: `http://localhost:9090`

Available metrics:
- Application performance
- Resource usage
- Error rates
- Response times

## 🔒 Security

### Production Security Checklist

- [ ] Change default passwords
- [ ] Use strong secret keys
- [ ] Enable SSL/TLS
- [ ] Configure firewall
- [ ] Set up backup encryption
- [ ] Enable audit logging
- [ ] Regular security updates

### SSL Configuration

1. Obtain SSL certificates
2. Place certificates in `/etc/ssl/`
3. Set environment variables:
   ```bash
   ENABLE_SSL=true
   SSL_CERT_PATH=/etc/ssl/certs/cert.pem
   SSL_KEY_PATH=/etc/ssl/private/key.pem
   ```

## 💾 Backup and Recovery

### Automated Backups

Backups are created automatically if `BACKUP_ENABLED=true`:

- **Schedule**: Daily at 2 AM (configurable)
- **Retention**: 30 days (configurable)
- **Location**: `./backups/`

### Manual Backup

```bash
# Create backup
./ops/deploy-utils.sh backup production_backup

# Restore from backup
./ops/deploy-utils.sh restore ./backups/postgres_backup_production_backup.sql
```

### Backup Strategy

1. **Database**: PostgreSQL dumps
2. **Vector Store**: Chroma data volume
3. **Logs**: Application logs
4. **Configuration**: Environment files

## 🚨 Troubleshooting

### Common Issues

#### Services Won't Start

```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs

# Check resource usage
docker stats

# Restart services
docker-compose -f docker-compose.prod.yml restart
```

#### Database Connection Issues

```bash
# Check PostgreSQL status
docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U coscientist

# Check connection string
echo $DATABASE_URL

# Reset database
docker-compose -f docker-compose.prod.yml down
docker volume rm coscientist_v2_postgres_data
docker-compose -f docker-compose.prod.yml up -d
```

#### Memory Issues

```bash
# Check memory usage
docker stats --no-stream

# Increase memory limits in docker-compose.prod.yml
# Or reduce worker processes in .env
```

#### API Key Issues

```bash
# Verify API keys are set
docker-compose -f docker-compose.prod.yml exec backend env | grep API_KEY

# Test API connections
docker-compose -f docker-compose.prod.yml exec backend python -c "
import os
print('Anthropic:', bool(os.getenv('ANTHROPIC_API_KEY')))
print('OpenAI:', bool(os.getenv('OPENAI_API_KEY')))
"
```

### Getting Help

1. Check logs: `./ops/deploy-utils.sh logs`
2. Check system status: `./ops/deploy-utils.sh status`
3. Review configuration: `cat .env`
4. Check Docker resources: `docker system df`

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale backend workers
./ops/deploy-utils.sh scale backend 3

# Scale Celery workers
./ops/deploy-utils.sh scale celery_worker 4
```

### Vertical Scaling

Update resource limits in `docker-compose.prod.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 16G
    reservations:
      cpus: '4'
      memory: 8G
```

### Load Balancing

For multiple instances, consider:

1. **Nginx** for frontend load balancing
2. **HAProxy** for backend load balancing
3. **Redis Cluster** for session storage
4. **PostgreSQL** read replicas

## 🔄 Updates

### Rolling Updates

```bash
# Automated rolling update
./ops/deploy-utils.sh rolling-update

# Manual service updates
./ops/deploy-utils.sh update backend
./ops/deploy-utils.sh update frontend
```

### Version Management

```bash
# Tag current version
git tag v1.0.0

# Deploy specific version
git checkout v1.0.0
./deploy.sh
```

## 📝 Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Check logs and metrics
2. **Monthly**: Update dependencies
3. **Quarterly**: Security audit
4. **Annually**: Architecture review

### Maintenance Commands

```bash
# Clean up old resources
./ops/deploy-utils.sh cleanup

# Update all services
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d --build

# Check for security updates
docker scout cves
```

## 🌐 Cloud Deployment

### AWS Deployment

1. Use ECS or EKS for container orchestration
2. RDS for PostgreSQL
3. ElastiCache for Redis
4. S3 for backups
5. CloudWatch for monitoring

### Google Cloud Deployment

1. Use GKE for container orchestration
2. Cloud SQL for PostgreSQL
3. Memorystore for Redis
4. Cloud Storage for backups
5. Cloud Monitoring for observability

### Azure Deployment

1. Use AKS for container orchestration
2. Azure Database for PostgreSQL
3. Azure Cache for Redis
4. Azure Blob Storage for backups
5. Azure Monitor for monitoring

## 📞 Support

For deployment issues:

1. Check this documentation
2. Review logs and system status
3. Check GitHub issues
4. Contact support team

---

**Next Steps**: After successful deployment, refer to the [User Guide](README.md) for application usage instructions. 