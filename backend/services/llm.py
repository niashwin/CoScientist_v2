"""
LLM Service for AI Co-Scientist system using Claude Sonnet 4.0
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable
from datetime import datetime
import time

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for interacting with Claude Sonnet 4.0 with streaming support
    """
    
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY
        )
        self.model = "claude-3-sonnet-20240229"
        self.max_tokens = 4096
        self.temperature = 0.7
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Usage tracking
        self.request_count = 0
        self.total_tokens = 0
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError))
    )
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        stream_callback: Optional[Callable] = None
    ) -> str:
        """
        Generate a response from Claude Sonnet 4.0
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            stream: Whether to stream the response
            stream_callback: Callback for streaming chunks
            
        Returns:
            Complete response text
        """
        await self._enforce_rate_limit()
        
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        if stream:
            return await self._stream_response(kwargs, stream_callback)
        else:
            return await self._generate_response(kwargs)
    
    async def _generate_response(self, kwargs: Dict[str, Any]) -> str:
        """Generate non-streaming response"""
        try:
            response = await self.client.messages.create(**kwargs)
            
            # Update usage tracking
            self.request_count += 1
            if hasattr(response, 'usage'):
                self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
            
            # Extract text from response
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text
                    
            return content
            
        except anthropic.RateLimitError as e:
            # Handle rate limiting
            wait_time = self._extract_wait_time(e)
            if wait_time:
                await asyncio.sleep(wait_time)
            raise
            
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    async def _stream_response(
        self, 
        kwargs: Dict[str, Any], 
        stream_callback: Optional[Callable] = None
    ) -> str:
        """Generate streaming response"""
        try:
            # Remove stream from kwargs since we're using the streaming context manager
            stream_kwargs = kwargs.copy()
            stream_kwargs.pop("stream", None)
            
            full_response = ""
            chunk_buffer = ""
            
            async with self.client.messages.stream(**stream_kwargs) as stream:
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        if chunk.delta.type == "text_delta":
                            chunk_text = chunk.delta.text
                            full_response += chunk_text
                            chunk_buffer += chunk_text
                            
                            # Send chunks to callback if provided
                            if stream_callback:
                                await stream_callback({
                                    "type": "stream_chunk",
                                    "content": chunk_text,
                                    "timestamp": time.time()
                                })
                            
                            # Send buffered chunks periodically
                            if len(chunk_buffer) >= 50:  # Send every 50 characters
                                chunk_buffer = ""
                                
                    elif chunk.type == "message_stop":
                        # Send final chunk if any remaining
                        if chunk_buffer and stream_callback:
                            await stream_callback({
                                "type": "stream_chunk",
                                "content": chunk_buffer,
                                "timestamp": time.time()
                            })
                        
                        # Send completion signal
                        if stream_callback:
                            await stream_callback({
                                "type": "stream_complete",
                                "full_response": full_response,
                                "timestamp": time.time()
                            })
            
            # Update usage tracking
            self.request_count += 1
            
            return full_response
            
        except Exception as e:
            if stream_callback:
                await stream_callback({
                    "type": "stream_error",
                    "error": str(e),
                    "timestamp": time.time()
                })
            raise Exception(f"LLM streaming failed: {str(e)}")
    
    async def generate_with_context(
        self,
        prompt: str,
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        stream: bool = False,
        stream_callback: Optional[Callable] = None
    ) -> str:
        """
        Generate response with conversation context
        
        Args:
            prompt: Current prompt
            context: Previous conversation history
            system_prompt: System prompt
            stream: Whether to stream
            stream_callback: Streaming callback
            
        Returns:
            Generated response
        """
        await self._enforce_rate_limit()
        
        # Build message history
        messages = []
        for ctx in context:
            messages.append({
                "role": ctx["role"],
                "content": ctx["content"]
            })
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        if stream:
            return await self._stream_response(kwargs, stream_callback)
        else:
            return await self._generate_response(kwargs)
    
    async def generate_structured_response(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured response following a schema
        
        Args:
            prompt: The prompt
            schema: Expected response schema
            system_prompt: System prompt
            
        Returns:
            Structured response as dictionary
        """
        # Add schema instructions to prompt
        schema_prompt = f"""
        {prompt}
        
        Please respond with a JSON object following this exact schema:
        {json.dumps(schema, indent=2)}
        
        Ensure your response is valid JSON and follows the schema exactly.
        """
        
        if not system_prompt:
            system_prompt = "You are a helpful assistant that always responds with valid JSON."
        
        response = await self.generate_response(
            schema_prompt,
            system_prompt=system_prompt,
            temperature=0.1  # Lower temperature for structured output
        )
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback: try to parse entire response
                return json.loads(response)
                
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse structured response: {str(e)}\nResponse: {response}")
    
    async def generate_streaming_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response that yields chunks directly
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Yields:
            Text chunks as they are generated
        """
        await self._enforce_rate_limit()
        
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        logger.info(f"Starting streaming response with model: {self.model}")
        
        try:
            chunk_count = 0
            async with self.client.messages.stream(**kwargs) as stream:
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        if chunk.delta.type == "text_delta":
                            chunk_count += 1
                            chunk_text = chunk.delta.text
                            logger.info(f"LLM streaming chunk {chunk_count}: {chunk_text[:30]}...")
                            yield chunk_text
                            
            logger.info(f"LLM streaming completed, total chunks: {chunk_count}")
                            
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            # Fallback to non-streaming if streaming fails
            response = await self._generate_response(kwargs)
            
            logger.info(f"Using fallback non-streaming response, length: {len(response)}")
            
            # Simulate streaming by yielding chunks
            chunk_size = 10
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i + chunk_size]
                logger.info(f"Fallback chunk: {chunk}")
                yield chunk
                await asyncio.sleep(0.05)  # Small delay to simulate streaming
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _extract_wait_time(self, error: anthropic.RateLimitError) -> Optional[float]:
        """Extract wait time from rate limit error"""
        try:
            # Try to extract from error message
            error_str = str(error)
            if "retry after" in error_str.lower():
                # Extract number from error message
                import re
                match = re.search(r'(\d+)', error_str)
                if match:
                    return float(match.group(1))
            
            # Default wait time
            return 60.0
            
        except:
            return 60.0
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "last_request": self.last_request_time
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the LLM service is healthy"""
        try:
            # Simple test request
            response = await self.generate_response(
                "Hello, please respond with 'OK' if you're working correctly.",
                max_tokens=10,
                temperature=0.0
            )
            
            return {
                "status": "healthy",
                "model": self.model,
                "response_time": time.time() - self.last_request_time,
                "test_response": response.strip()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model
            }

# Singleton instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get singleton LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service 