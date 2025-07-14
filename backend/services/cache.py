"""
Cache service with Redis and in-memory fallback support.
Provides intelligent caching for literature search results, embeddings, and other data.
"""

import asyncio
import json
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import hashlib

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from backend.core.config import settings

logger = logging.getLogger(__name__)

class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass

class InMemoryCache(CacheBackend):
    """In-memory cache implementation with TTL support"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = asyncio.Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache"""
        async with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry.get('expires_at') and time.time() > entry['expires_at']:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                self.misses += 1
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            self.hits += 1
            
            return entry['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in in-memory cache"""
        async with self.lock:
            # Calculate expiration time
            expires_at = None
            if ttl:
                expires_at = time.time() + ttl
            
            # Evict if at max size
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()
            
            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            self.access_times[key] = time.time()
            self.sets += 1
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from in-memory cache"""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                self.deletes += 1
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in in-memory cache"""
        result = await self.get(key)
        return result is not None
    
    async def clear(self) -> bool:
        """Clear all entries from in-memory cache"""
        async with self.lock:
            self.cache.clear()
            self.access_times.clear()
            return True
    
    async def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        # Find least recently used key
        lru_key = min(self.access_times, key=self.access_times.get)
        
        # Remove from cache
        if lru_key in self.cache:
            del self.cache[lru_key]
        del self.access_times[lru_key]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get in-memory cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "backend": "in_memory",
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "hit_rate": hit_rate,
            "utilization": len(self.cache) / self.max_size
        }

class RedisCache(CacheBackend):
    """Redis cache implementation"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.connected = False
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
    
    async def _ensure_connected(self):
        """Ensure Redis connection is established"""
        if not self.connected or not self.redis_client:
            try:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    decode_responses=False,  # We handle encoding ourselves
                    socket_timeout=settings.REDIS_TIMEOUT,
                    socket_connect_timeout=settings.REDIS_TIMEOUT
                )
                
                # Test connection
                await self.redis_client.ping()
                self.connected = True
                logger.info("Redis cache connection established")
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.connected = False
                raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            await self._ensure_connected()
            
            data = await self.redis_client.get(key)
            if data is None:
                self.misses += 1
                return None
            
            # Deserialize
            value = pickle.loads(data)
            self.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        try:
            await self._ensure_connected()
            
            # Serialize
            data = pickle.dumps(value)
            
            # Set with TTL
            if ttl:
                await self.redis_client.setex(key, ttl, data)
            else:
                await self.redis_client.set(key, data)
            
            self.sets += 1
            return True
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        try:
            await self._ensure_connected()
            
            result = await self.redis_client.delete(key)
            if result > 0:
                self.deletes += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache"""
        try:
            await self._ensure_connected()
            
            result = await self.redis_client.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all entries from Redis cache"""
        try:
            await self._ensure_connected()
            
            await self.redis_client.flushdb()
            return True
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        try:
            await self._ensure_connected()
            
            info = await self.redis_client.info()
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "backend": "redis",
                "connected": self.connected,
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "deletes": self.deletes,
                "hit_rate": hit_rate,
                "redis_info": {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {
                "backend": "redis",
                "connected": False,
                "error": str(e)
            }
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.connected = False

class CacheService:
    """
    High-level cache service with fallback strategies
    """
    
    def __init__(self):
        self.primary_backend: Optional[CacheBackend] = None
        self.fallback_backend: CacheBackend = InMemoryCache(max_size=500)
        self.key_prefix = "coscientist:"
        
        # Initialize Redis if available
        if REDIS_AVAILABLE and hasattr(settings, 'REDIS_URL'):
            try:
                self.primary_backend = RedisCache(settings.REDIS_URL)
            except Exception as e:
                logger.warning(f"Redis initialization failed, using in-memory cache: {e}")
                self.primary_backend = None
    
    def _make_key(self, key: str, namespace: str = "default") -> str:
        """Create a namespaced cache key"""
        return f"{self.key_prefix}{namespace}:{key}"
    
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache with fallback"""
        cache_key = self._make_key(key, namespace)
        
        # Try primary backend first
        if self.primary_backend:
            try:
                result = await self.primary_backend.get(cache_key)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Primary cache get failed: {e}")
        
        # Fallback to in-memory cache
        return await self.fallback_backend.get(cache_key)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = "default"
    ) -> bool:
        """Set value in cache with fallback"""
        cache_key = self._make_key(key, namespace)
        
        success = False
        
        # Try primary backend first
        if self.primary_backend:
            try:
                success = await self.primary_backend.set(cache_key, value, ttl)
            except Exception as e:
                logger.warning(f"Primary cache set failed: {e}")
        
        # Always set in fallback cache
        fallback_success = await self.fallback_backend.set(cache_key, value, ttl)
        
        return success or fallback_success
    
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete key from cache"""
        cache_key = self._make_key(key, namespace)
        
        success = False
        
        # Delete from primary backend
        if self.primary_backend:
            try:
                success = await self.primary_backend.delete(cache_key)
            except Exception as e:
                logger.warning(f"Primary cache delete failed: {e}")
        
        # Delete from fallback cache
        fallback_success = await self.fallback_backend.delete(cache_key)
        
        return success or fallback_success
    
    async def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if key exists in cache"""
        cache_key = self._make_key(key, namespace)
        
        # Check primary backend first
        if self.primary_backend:
            try:
                if await self.primary_backend.exists(cache_key):
                    return True
            except Exception as e:
                logger.warning(f"Primary cache exists failed: {e}")
        
        # Check fallback cache
        return await self.fallback_backend.exists(cache_key)
    
    async def clear(self, namespace: Optional[str] = None) -> bool:
        """Clear cache entries"""
        if namespace:
            # Clear specific namespace (would need pattern matching for Redis)
            # For now, we'll just clear everything
            pass
        
        success = False
        
        # Clear primary backend
        if self.primary_backend:
            try:
                success = await self.primary_backend.clear()
            except Exception as e:
                logger.warning(f"Primary cache clear failed: {e}")
        
        # Clear fallback cache
        fallback_success = await self.fallback_backend.clear()
        
        return success or fallback_success
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "primary_backend": None,
            "fallback_backend": None,
            "configuration": {
                "redis_available": REDIS_AVAILABLE,
                "primary_enabled": self.primary_backend is not None,
                "key_prefix": self.key_prefix
            }
        }
        
        # Get primary backend stats
        if self.primary_backend:
            try:
                stats["primary_backend"] = await self.primary_backend.get_stats()
            except Exception as e:
                stats["primary_backend"] = {"error": str(e)}
        
        # Get fallback backend stats
        try:
            stats["fallback_backend"] = await self.fallback_backend.get_stats()
        except Exception as e:
            stats["fallback_backend"] = {"error": str(e)}
        
        return stats
    
    async def close(self):
        """Close all cache connections"""
        if self.primary_backend and hasattr(self.primary_backend, 'close'):
            await self.primary_backend.close()

class CacheManager:
    """
    Manager for different cache namespaces with specific configurations
    """
    
    def __init__(self):
        self.cache_service = CacheService()
        
        # Namespace-specific TTL configurations
        self.namespace_ttls = {
            "literature": settings.LITERATURE_CACHE_TTL,
            "embeddings": settings.EMBEDDING_CACHE_TTL,
            "hypotheses": settings.HYPOTHESIS_CACHE_TTL,
            "api_responses": 300,  # 5 minutes
            "user_sessions": 3600,  # 1 hour
            "circuit_breakers": 60   # 1 minute
        }
    
    async def get_literature(self, key: str) -> Optional[Any]:
        """Get literature search results from cache"""
        return await self.cache_service.get(key, "literature")
    
    async def set_literature(self, key: str, value: Any) -> bool:
        """Cache literature search results"""
        ttl = self.namespace_ttls.get("literature")
        return await self.cache_service.set(key, value, ttl, "literature")
    
    async def get_embeddings(self, key: str) -> Optional[Any]:
        """Get embeddings from cache"""
        return await self.cache_service.get(key, "embeddings")
    
    async def set_embeddings(self, key: str, value: Any) -> bool:
        """Cache embeddings"""
        ttl = self.namespace_ttls.get("embeddings")
        return await self.cache_service.set(key, value, ttl, "embeddings")
    
    async def get_hypothesis(self, key: str) -> Optional[Any]:
        """Get hypothesis from cache"""
        return await self.cache_service.get(key, "hypotheses")
    
    async def set_hypothesis(self, key: str, value: Any) -> bool:
        """Cache hypothesis"""
        ttl = self.namespace_ttls.get("hypotheses")
        return await self.cache_service.set(key, value, ttl, "hypotheses")
    
    async def get_api_response(self, key: str) -> Optional[Any]:
        """Get API response from cache"""
        return await self.cache_service.get(key, "api_responses")
    
    async def set_api_response(self, key: str, value: Any) -> bool:
        """Cache API response"""
        ttl = self.namespace_ttls.get("api_responses")
        return await self.cache_service.set(key, value, ttl, "api_responses")
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments"""
        # Create a deterministic key from arguments
        key_parts = []
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(hash(str(arg))))
        
        # Add keyword arguments (sorted for consistency)
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        
        # Create hash of the key parts
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache manager statistics"""
        base_stats = await self.cache_service.get_stats()
        
        return {
            **base_stats,
            "namespace_ttls": self.namespace_ttls,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def close(self):
        """Close cache manager"""
        await self.cache_service.close()

# Global cache manager instance
cache_manager = CacheManager()

# Convenience functions for common operations
async def get_cached_literature(query: str, strategy: str, limit: int) -> Optional[List[Dict]]:
    """Get cached literature search results"""
    key = cache_manager.generate_key("literature", query, strategy, limit)
    return await cache_manager.get_literature(key)

async def cache_literature_results(query: str, strategy: str, limit: int, results: List[Dict]) -> bool:
    """Cache literature search results"""
    key = cache_manager.generate_key("literature", query, strategy, limit)
    return await cache_manager.set_literature(key, results)

async def get_cached_embeddings(text: str, model: str) -> Optional[List[float]]:
    """Get cached embeddings"""
    key = cache_manager.generate_key("embeddings", text, model)
    return await cache_manager.get_embeddings(key)

async def cache_embeddings(text: str, model: str, embeddings: List[float]) -> bool:
    """Cache embeddings"""
    key = cache_manager.generate_key("embeddings", text, model)
    return await cache_manager.set_embeddings(key, embeddings) 