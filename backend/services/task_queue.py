"""
Redis/Celery integration for distributed task processing.
Provides reliable, scalable task execution for the AI Co-Scientist system.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
from dataclasses import dataclass, asdict

try:
    from celery import Celery, Task
    from celery.exceptions import MaxRetriesExceededError, Retry
    from celery.signals import task_prerun, task_postrun, task_failure, task_success
    import redis.asyncio as redis
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

from backend.core.config import settings
from backend.core.exceptions import CoScientistError, ErrorSeverity

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TaskInfo:
    """Task information and metadata"""
    task_id: str
    task_name: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

class CoScientistTask(Task):
    """Custom Celery task with enhanced features"""
    
    def __init__(self):
        super().__init__()
        self.task_info: Optional[TaskInfo] = None
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        logger.info(f"Task {task_id} succeeded with result: {retval}")
        if self.task_info:
            self.task_info.status = TaskStatus.SUCCESS
            self.task_info.completed_at = datetime.utcnow()
            self.task_info.result = retval
            self.task_info.progress = 1.0
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger.error(f"Task {task_id} failed: {exc}")
        if self.task_info:
            self.task_info.status = TaskStatus.FAILURE
            self.task_info.completed_at = datetime.utcnow()
            self.task_info.error = str(exc)
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(f"Task {task_id} retrying: {exc}")
        if self.task_info:
            self.task_info.status = TaskStatus.RETRY
            self.task_info.retry_count += 1
            self.task_info.error = str(exc)
    
    def update_progress(self, progress: float, message: str = None):
        """Update task progress"""
        if self.task_info:
            self.task_info.progress = min(max(progress, 0.0), 1.0)
        
        # Update Celery task state
        self.update_state(
            state='PROGRESS',
            meta={
                'progress': progress,
                'message': message or f"Progress: {progress:.1%}"
            }
        )

def create_celery_app() -> Optional[Celery]:
    """Create and configure Celery application"""
    if not CELERY_AVAILABLE:
        logger.warning("Celery not available, tasks will run synchronously")
        return None
    
    app = Celery('coscientist')
    
    # Configure Celery
    app.conf.update(
        broker_url=settings.CELERY_BROKER_URL,
        result_backend=settings.CELERY_RESULT_BACKEND,
        task_serializer=settings.CELERY_TASK_SERIALIZER,
        accept_content=[settings.CELERY_ACCEPT_CONTENT],
        result_serializer=settings.CELERY_RESULT_SERIALIZER,
        timezone=settings.CELERY_TIMEZONE,
        enable_utc=True,
        
        # Task routing
        task_routes={
            'coscientist.agents.*': {'queue': 'agents'},
            'coscientist.literature.*': {'queue': 'literature'},
            'coscientist.tools.*': {'queue': 'tools'},
            'coscientist.embeddings.*': {'queue': 'embeddings'},
        },
        
        # Task execution
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_default_retry_delay=settings.RETRY_BACKOFF_SECONDS,
        task_max_retries=settings.RETRY_MAX_ATTEMPTS,
        
        # Worker configuration
        worker_prefetch_multiplier=settings.CELERY_WORKER_PREFETCH_MULTIPLIER,
        worker_max_tasks_per_child=1000,
        worker_disable_rate_limits=False,
        
        # Result backend settings
        result_expires=3600,  # 1 hour
        result_persistent=True,
        
        # Monitoring
        worker_send_task_events=True,
        task_send_sent_event=True,
        
        # Security
        task_always_eager=False,  # Set to True for testing
        task_store_eager_result=True,
    )
    
    # Set custom task base class
    app.Task = CoScientistTask
    
    return app

# Global Celery app instance
celery_app = create_celery_app()

class TaskManager:
    """
    High-level task management interface
    """
    
    def __init__(self):
        self.celery_app = celery_app
        self.redis_client: Optional[redis.Redis] = None
        self.task_registry: Dict[str, TaskInfo] = {}
        
        # Initialize Redis client for task tracking
        if CELERY_AVAILABLE:
            try:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True
                )
            except Exception as e:
                logger.warning(f"Redis client initialization failed: {e}")
    
    async def submit_task(
        self,
        task_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        countdown: int = 0,
        eta: Optional[datetime] = None,
        expires: Optional[datetime] = None,
        retry_policy: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a task for execution
        
        Args:
            task_name: Name of the task to execute
            args: Positional arguments for the task
            kwargs: Keyword arguments for the task
            priority: Task priority level
            countdown: Delay before execution (seconds)
            eta: Specific time to execute the task
            expires: Task expiration time
            retry_policy: Custom retry policy
            
        Returns:
            Task ID
        """
        args = args or []
        kwargs = kwargs or {}
        
        task_id = str(uuid.uuid4())
        
        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            task_name=task_name,
            status=TaskStatus.PENDING,
            priority=priority,
            created_at=datetime.utcnow()
        )
        
        # Store task info
        self.task_registry[task_id] = task_info
        await self._store_task_info(task_info)
        
        if self.celery_app:
            # Submit to Celery
            try:
                # Apply retry policy if provided
                if retry_policy:
                    kwargs.update(retry_policy)
                
                # Submit task
                result = self.celery_app.send_task(
                    task_name,
                    args=args,
                    kwargs=kwargs,
                    task_id=task_id,
                    priority=priority.value,
                    countdown=countdown,
                    eta=eta,
                    expires=expires,
                    retry=True,
                    retry_policy=retry_policy or {
                        'max_retries': settings.RETRY_MAX_ATTEMPTS,
                        'interval_start': settings.RETRY_BACKOFF_SECONDS,
                        'interval_step': settings.RETRY_BACKOFF_SECONDS,
                        'interval_max': 60
                    }
                )
                
                logger.info(f"Task {task_id} ({task_name}) submitted to Celery")
                return task_id
                
            except Exception as e:
                logger.error(f"Failed to submit task to Celery: {e}")
                task_info.status = TaskStatus.FAILURE
                task_info.error = str(e)
                await self._store_task_info(task_info)
                raise CoScientistError(
                    f"Failed to submit task: {str(e)}",
                    "TASK_SUBMIT_001",
                    ErrorSeverity.HIGH
                )
        else:
            # Fallback: execute synchronously
            logger.warning(f"Celery not available, executing task {task_name} synchronously")
            try:
                # This would need to be implemented based on available task functions
                result = await self._execute_task_sync(task_name, args, kwargs)
                task_info.status = TaskStatus.SUCCESS
                task_info.completed_at = datetime.utcnow()
                task_info.result = result
                task_info.progress = 1.0
                await self._store_task_info(task_info)
                return task_id
                
            except Exception as e:
                logger.error(f"Synchronous task execution failed: {e}")
                task_info.status = TaskStatus.FAILURE
                task_info.error = str(e)
                task_info.completed_at = datetime.utcnow()
                await self._store_task_info(task_info)
                raise CoScientistError(
                    f"Task execution failed: {str(e)}",
                    "TASK_EXEC_001",
                    ErrorSeverity.HIGH
                )
    
    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get task status and information"""
        # Check local registry first
        if task_id in self.task_registry:
            task_info = self.task_registry[task_id]
            
            # Update from Celery if available
            if self.celery_app:
                try:
                    result = self.celery_app.AsyncResult(task_id)
                    if result.state == 'PENDING':
                        task_info.status = TaskStatus.PENDING
                    elif result.state == 'STARTED':
                        task_info.status = TaskStatus.STARTED
                        task_info.started_at = datetime.utcnow()
                    elif result.state == 'SUCCESS':
                        task_info.status = TaskStatus.SUCCESS
                        task_info.completed_at = datetime.utcnow()
                        task_info.result = result.result
                        task_info.progress = 1.0
                    elif result.state == 'FAILURE':
                        task_info.status = TaskStatus.FAILURE
                        task_info.completed_at = datetime.utcnow()
                        task_info.error = str(result.info)
                    elif result.state == 'RETRY':
                        task_info.status = TaskStatus.RETRY
                        task_info.retry_count += 1
                    elif result.state == 'REVOKED':
                        task_info.status = TaskStatus.REVOKED
                    elif result.state == 'PROGRESS':
                        task_info.progress = result.info.get('progress', 0.0)
                    
                    # Update stored info
                    await self._store_task_info(task_info)
                    
                except Exception as e:
                    logger.error(f"Failed to get task status from Celery: {e}")
            
            return task_info
        
        # Try to load from Redis
        return await self._load_task_info(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if self.celery_app:
            try:
                self.celery_app.control.revoke(task_id, terminate=True)
                
                # Update task info
                if task_id in self.task_registry:
                    task_info = self.task_registry[task_id]
                    task_info.status = TaskStatus.REVOKED
                    task_info.completed_at = datetime.utcnow()
                    await self._store_task_info(task_info)
                
                logger.info(f"Task {task_id} cancelled")
                return True
                
            except Exception as e:
                logger.error(f"Failed to cancel task {task_id}: {e}")
                return False
        
        return False
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        task_name: Optional[str] = None,
        limit: int = 100
    ) -> List[TaskInfo]:
        """List tasks with optional filtering"""
        tasks = []
        
        # Get from local registry
        for task_info in self.task_registry.values():
            if status and task_info.status != status:
                continue
            if task_name and task_info.task_name != task_name:
                continue
            tasks.append(task_info)
        
        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        return tasks[:limit]
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        if not self.celery_app:
            return {"error": "Celery not available"}
        
        try:
            # Get active tasks
            active_tasks = self.celery_app.control.inspect().active()
            
            # Get scheduled tasks
            scheduled_tasks = self.celery_app.control.inspect().scheduled()
            
            # Get reserved tasks
            reserved_tasks = self.celery_app.control.inspect().reserved()
            
            # Get worker stats
            worker_stats = self.celery_app.control.inspect().stats()
            
            return {
                "active_tasks": active_tasks,
                "scheduled_tasks": scheduled_tasks,
                "reserved_tasks": reserved_tasks,
                "worker_stats": worker_stats,
                "total_tasks": len(self.task_registry)
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {"error": str(e)}
    
    async def _store_task_info(self, task_info: TaskInfo):
        """Store task information in Redis"""
        if self.redis_client:
            try:
                key = f"task:{task_info.task_id}"
                data = json.dumps(task_info.to_dict())
                await self.redis_client.setex(key, 3600, data)  # 1 hour expiry
            except Exception as e:
                logger.error(f"Failed to store task info: {e}")
    
    async def _load_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """Load task information from Redis"""
        if self.redis_client:
            try:
                key = f"task:{task_id}"
                data = await self.redis_client.get(key)
                if data:
                    task_dict = json.loads(data)
                    return TaskInfo(
                        task_id=task_dict['task_id'],
                        task_name=task_dict['task_name'],
                        status=TaskStatus(task_dict['status']),
                        priority=TaskPriority(task_dict['priority']),
                        created_at=datetime.fromisoformat(task_dict['created_at']),
                        started_at=datetime.fromisoformat(task_dict['started_at']) if task_dict['started_at'] else None,
                        completed_at=datetime.fromisoformat(task_dict['completed_at']) if task_dict['completed_at'] else None,
                        result=task_dict['result'],
                        error=task_dict['error'],
                        progress=task_dict['progress'],
                        retry_count=task_dict['retry_count'],
                        max_retries=task_dict['max_retries']
                    )
            except Exception as e:
                logger.error(f"Failed to load task info: {e}")
        
        return None
    
    async def _execute_task_sync(self, task_name: str, args: List[Any], kwargs: Dict[str, Any]) -> Any:
        """Execute task synchronously (fallback when Celery not available)"""
        # This would need to be implemented based on available task functions
        # For now, we'll just return a placeholder
        await asyncio.sleep(0.1)  # Simulate work
        return {"status": "completed", "note": "Executed synchronously"}
    
    async def close(self):
        """Close task manager connections"""
        if self.redis_client:
            await self.redis_client.close()

# Global task manager instance
task_manager = TaskManager()

# Task definitions
if celery_app:
    
    @celery_app.task(bind=True, base=CoScientistTask)
    def run_agent_task(self, agent_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent task"""
        self.update_progress(0.1, f"Starting {agent_type} agent")
        
        try:
            # Import here to avoid circular imports
            from backend.agents.swarm_orchestrator import SwarmOrchestrator
            
            self.update_progress(0.3, "Initializing agent")
            orchestrator = SwarmOrchestrator()
            
            self.update_progress(0.5, "Executing agent")
            result = orchestrator.run_agent(agent_type, context)
            
            self.update_progress(0.9, "Finalizing result")
            
            return {
                "status": "success",
                "agent_type": agent_type,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Agent task failed: {e}")
            raise
    
    @celery_app.task(bind=True, base=CoScientistTask)
    def literature_search_task(self, query: str, strategy: str = "broad_survey") -> Dict[str, Any]:
        """Execute literature search task"""
        self.update_progress(0.1, "Starting literature search")
        
        try:
            # Import here to avoid circular imports
            from backend.services.literature_search import SmartLiteratureSearch, SearchStrategy
            
            self.update_progress(0.3, "Initializing search")
            search = SmartLiteratureSearch()
            
            self.update_progress(0.5, "Executing search")
            strategy_enum = SearchStrategy(strategy)
            results = asyncio.run(search.search(query, strategy_enum))
            
            self.update_progress(0.9, "Processing results")
            
            return {
                "status": "success",
                "query": query,
                "strategy": strategy,
                "results": [paper.to_dict() for paper in results],
                "count": len(results),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Literature search task failed: {e}")
            raise
    
    @celery_app.task(bind=True, base=CoScientistTask)
    def scientific_tool_task(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scientific tool task"""
        self.update_progress(0.1, f"Starting {tool_name} tool")
        
        try:
            # Import here to avoid circular imports
            from backend.services.scientific_tools import scientific_tools_registry
            
            self.update_progress(0.3, "Initializing tool")
            
            self.update_progress(0.5, "Executing tool")
            result = asyncio.run(scientific_tools_registry.call_tool(tool_name, **params))
            
            self.update_progress(0.9, "Processing result")
            
            return {
                "status": "success",
                "tool_name": tool_name,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Scientific tool task failed: {e}")
            raise
    
    @celery_app.task(bind=True, base=CoScientistTask)
    def embedding_task(self, texts: List[str], model: str = "text-embedding-3-large") -> Dict[str, Any]:
        """Execute embedding generation task"""
        self.update_progress(0.1, "Starting embedding generation")
        
        try:
            # Import here to avoid circular imports
            from backend.services.embeddings import EmbeddingService
            
            self.update_progress(0.3, "Initializing embedding service")
            embedding_service = EmbeddingService()
            
            self.update_progress(0.5, "Generating embeddings")
            embeddings = asyncio.run(embedding_service.generate_embeddings(texts, model))
            
            self.update_progress(0.9, "Processing embeddings")
            
            return {
                "status": "success",
                "model": model,
                "embeddings": embeddings,
                "count": len(embeddings),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Embedding task failed: {e}")
            raise

# Convenience functions for common task submissions
async def submit_agent_task(agent_type: str, context: Dict[str, Any], priority: TaskPriority = TaskPriority.NORMAL) -> str:
    """Submit an agent execution task"""
    return await task_manager.submit_task(
        "coscientist.agents.run_agent_task",
        args=[agent_type, context],
        priority=priority
    )

async def submit_literature_search(query: str, strategy: str = "broad_survey", priority: TaskPriority = TaskPriority.NORMAL) -> str:
    """Submit a literature search task"""
    return await task_manager.submit_task(
        "coscientist.literature.literature_search_task",
        args=[query, strategy],
        priority=priority
    )

async def submit_scientific_tool_task(tool_name: str, params: Dict[str, Any], priority: TaskPriority = TaskPriority.NORMAL) -> str:
    """Submit a scientific tool task"""
    return await task_manager.submit_task(
        "coscientist.tools.scientific_tool_task",
        args=[tool_name, params],
        priority=priority
    )

async def submit_embedding_task(texts: List[str], model: str = "text-embedding-3-large", priority: TaskPriority = TaskPriority.NORMAL) -> str:
    """Submit an embedding generation task"""
    return await task_manager.submit_task(
        "coscientist.embeddings.embedding_task",
        args=[texts, model],
        priority=priority
    ) 