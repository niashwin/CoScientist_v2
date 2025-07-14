"""
Embeddings Service for AI Co-Scientist system using OpenAI text-embedding-3-large
"""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional
import time
from datetime import datetime, timedelta

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.core.config import settings

class EmbeddingsService:
    """
    Service for generating embeddings using OpenAI text-embedding-3-large
    """
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY
        )
        self.model = settings.EMBEDDING_MODEL
        self.max_chunk_size = 8192  # Max tokens per chunk
        
        # Caching
        self.cache = {}
        self.cache_ttl = settings.HYPOTHESIS_CACHE_TTL
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms between requests
        
        # Usage tracking
        self.request_count = 0
        self.total_tokens = 0
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    async def get_embedding(
        self, 
        text: str, 
        use_cache: bool = True
    ) -> List[float]:
        """
        Get embedding for a single text
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector as list of floats
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(text)
            cached_embedding = self._get_from_cache(cache_key)
            if cached_embedding:
                return cached_embedding
        
        await self._enforce_rate_limit()
        
        try:
            # Truncate text if too long
            text = self._truncate_text(text)
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Update usage tracking
            self.request_count += 1
            self.total_tokens += response.usage.total_tokens
            
            # Cache the result
            if use_cache:
                self._add_to_cache(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    async def get_embeddings_batch(
        self, 
        texts: List[str], 
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached_embedding = self._get_from_cache(cache_key)
                if cached_embedding:
                    embeddings.append(cached_embedding)
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Process uncached texts
        if uncached_texts:
            await self._enforce_rate_limit()
            
            try:
                # Truncate texts if needed
                processed_texts = [self._truncate_text(text) for text in uncached_texts]
                
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=processed_texts,
                    encoding_format="float"
                )
                
                # Update usage tracking
                self.request_count += 1
                self.total_tokens += response.usage.total_tokens
                
                # Extract embeddings and cache them
                for i, data in enumerate(response.data):
                    embedding = data.embedding
                    original_index = uncached_indices[i]
                    embeddings[original_index] = embedding
                    
                    # Cache the result
                    if use_cache:
                        cache_key = self._get_cache_key(uncached_texts[i])
                        self._add_to_cache(cache_key, embedding)
                
            except Exception as e:
                raise Exception(f"Batch embedding generation failed: {str(e)}")
        
        return embeddings
    
    def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            
            # Calculate magnitudes
            magnitude1 = sum(a * a for a in embedding1) ** 0.5
            magnitude2 = sum(b * b for b in embedding2) ** 0.5
            
            # Calculate cosine similarity
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            similarity = dot_product / (magnitude1 * magnitude2)
            return similarity
            
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def find_most_similar(
        self, 
        query_text: str, 
        candidate_texts: List[str], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar texts to a query
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with text, similarity, and index
        """
        try:
            # Get embeddings for query and candidates
            query_embedding = await self.get_embedding(query_text)
            candidate_embeddings = await self.get_embeddings_batch(candidate_texts)
            
            # Calculate similarities
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate_embedding)
                similarities.append({
                    "text": candidate_texts[i],
                    "similarity": similarity,
                    "index": i
                })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Similarity search failed: {e}")
            return []
    
    async def cluster_texts(
        self, 
        texts: List[str], 
        num_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        Cluster texts based on semantic similarity
        
        Args:
            texts: List of texts to cluster
            num_clusters: Number of clusters
            
        Returns:
            Dictionary with cluster assignments and centroids
        """
        try:
            # Get embeddings for all texts
            embeddings = await self.get_embeddings_batch(texts)
            
            # Simple k-means clustering
            import numpy as np
            from sklearn.cluster import KMeans
            
            # Convert to numpy array
            embedding_matrix = np.array(embeddings)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embedding_matrix)
            
            # Organize results
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    "text": texts[i],
                    "index": i,
                    "embedding": embeddings[i]
                })
            
            return {
                "clusters": clusters,
                "centroids": kmeans.cluster_centers_.tolist(),
                "num_clusters": num_clusters
            }
            
        except Exception as e:
            print(f"Clustering failed: {e}")
            return {"clusters": {}, "centroids": [], "num_clusters": 0}
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Get embedding from cache if not expired"""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.utcnow() - cached_data["timestamp"] < timedelta(seconds=self.cache_ttl):
                return cached_data["embedding"]
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        return None
    
    def _add_to_cache(self, cache_key: str, embedding: List[float]):
        """Add embedding to cache"""
        self.cache[cache_key] = {
            "embedding": embedding,
            "timestamp": datetime.utcnow()
        }
        
        # Simple cache cleanup - remove old entries if cache gets too large
        if len(self.cache) > 1000:
            # Remove oldest 20% of entries
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k]["timestamp"]
            )
            for key in sorted_keys[:200]:
                del self.cache[key]
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits"""
        # Simple truncation - in production, you'd want proper tokenization
        if len(text) > self.max_chunk_size:
            return text[:self.max_chunk_size]
        return text
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "cache_size": len(self.cache),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # This is a simplified calculation
        # In production, you'd want proper hit/miss tracking
        return 0.0 if self.request_count == 0 else min(0.8, len(self.cache) / max(1, self.request_count))
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the embeddings service is healthy"""
        try:
            # Simple test embedding
            test_embedding = await self.get_embedding("test", use_cache=False)
            
            return {
                "status": "healthy",
                "model": self.model,
                "embedding_dim": len(test_embedding),
                "response_time": time.time() - self.last_request_time
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model
            }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear()

# Singleton instance
_embeddings_service = None

def get_embeddings_service() -> EmbeddingsService:
    """Get singleton embeddings service instance"""
    global _embeddings_service
    if _embeddings_service is None:
        _embeddings_service = EmbeddingsService()
    return _embeddings_service

# Convenience function for getting single embedding
async def get_embedding(text: str, use_cache: bool = True) -> List[float]:
    """Convenience function to get embedding for a single text"""
    service = get_embeddings_service()
    return await service.get_embedding(text, use_cache) 