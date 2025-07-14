"""
WebSocket Connection Manager with automatic reconnection support
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional, Set, List
import json
import asyncio
import uuid
from datetime import datetime
from collections import defaultdict
import redis.asyncio as redis

from backend.core.config import settings

class WebSocketConnectionManager:
    """
    Enhanced WebSocket manager with automatic reconnection support,
    message buffering, and connection state persistence.
    """
    
    def __init__(self):
        # Active WebSocket connections
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[str, Dict] = {}
        
        # Message buffers for disconnected clients
        self.message_buffers: Dict[str, List] = defaultdict(list)
        
        # Heartbeat tracking
        self.last_heartbeat: Dict[str, datetime] = {}
        
        # Redis client for distributed state management
        self.redis_client = None
        if settings.REDIS_URL:
            self._init_redis()
        
        # Heartbeat monitor will be started when first connection is made
        self._heartbeat_task = None
    
    def _init_redis(self):
        """Initialize Redis client for distributed state management"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True
            )
        except Exception as e:
            print(f"Redis initialization failed: {e}")
            self.redis_client = None
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Accept WebSocket connection with reconnection support
        
        Returns: client_id for future reference
        """
        await websocket.accept()
        
        # Generate or reuse client ID
        if not client_id:
            client_id = str(uuid.uuid4())
        
        # Store connection
        self.active_connections[client_id] = websocket
        self.last_heartbeat[client_id] = datetime.utcnow()
        
        # Store metadata
        self.connection_metadata[client_id] = {
            "connected_at": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "reconnection_count": 0
        }
        
        # Send any buffered messages
        if client_id in self.message_buffers:
            await self._send_buffered_messages(client_id)
        
        # Send connection confirmation
        await self.send_json(client_id, {
            "type": "connection_established",
            "client_id": client_id,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Start heartbeat monitor if not already running
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
        return client_id
    
    async def disconnect(self, client_id: str, save_state: bool = True):
        """Disconnect client and optionally save state for reconnection"""
        if client_id in self.active_connections:
            if save_state and self.redis_client:
                # Save connection state for potential reconnection
                metadata = self.connection_metadata.get(client_id, {})
                metadata["disconnected_at"] = datetime.utcnow().isoformat()
                
                # Keep metadata for 24 hours
                await self.redis_client.setex(
                    f"ws_metadata:{client_id}",
                    86400,
                    json.dumps(metadata)
                )
            
            # Remove active connection
            del self.active_connections[client_id]
            if client_id in self.last_heartbeat:
                del self.last_heartbeat[client_id]
            if client_id in self.connection_metadata:
                del self.connection_metadata[client_id]
    
    async def reconnect(
        self,
        websocket: WebSocket,
        client_id: str,
        last_message_id: Optional[str] = None
    ) -> bool:
        """
        Handle client reconnection
        
        Returns: True if reconnection successful
        """
        # Check if client can reconnect
        if self.redis_client:
            metadata_key = f"ws_metadata:{client_id}"
            saved_metadata = await self.redis_client.get(metadata_key)
            
            if not saved_metadata:
                return False
            
            metadata = json.loads(saved_metadata)
            metadata["reconnection_count"] = metadata.get("reconnection_count", 0) + 1
        else:
            # Without Redis, allow reconnection but with limited state
            metadata = {"reconnection_count": 1}
        
        # Accept reconnection
        await self.connect(websocket, client_id, metadata.get("session_id"))
        
        # Send reconnection confirmation
        await self.send_json(client_id, {
            "type": "reconnection_successful",
            "client_id": client_id,
            "reconnection_count": metadata["reconnection_count"],
            "last_message_id": last_message_id
        })
        
        # Resend messages since last_message_id
        if last_message_id and client_id in self.message_buffers:
            await self._resend_messages_since(client_id, last_message_id)
        
        return True
    
    async def send_json(
        self,
        client_id: str,
        data: Dict[str, Any],
        save_to_buffer: bool = True
    ) -> bool:
        """
        Send JSON data to client with automatic buffering
        
        Returns: True if sent successfully
        """
        # Add message ID and timestamp
        message = {
            **data,
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_json(message)
                
                # Update heartbeat
                self.last_heartbeat[client_id] = datetime.utcnow()
                
                return True
                
            except Exception as e:
                print(f"Failed to send message to {client_id}: {e}")
                
                # Connection failed - buffer message
                if save_to_buffer:
                    self._buffer_message(client_id, message)
                
                # Disconnect client
                await self.disconnect(client_id, save_state=True)
                
                return False
        else:
            # Client not connected - buffer message
            if save_to_buffer:
                self._buffer_message(client_id, message)
            
            return False
    
    async def broadcast_json(
        self,
        data: Dict[str, Any],
        session_id: Optional[str] = None
    ):
        """Broadcast to all clients or specific session"""
        # Get relevant client IDs
        if session_id:
            client_ids = [
                cid for cid, meta in self.connection_metadata.items()
                if meta.get("session_id") == session_id
            ]
        else:
            client_ids = list(self.active_connections.keys())
        
        # Send to each client
        tasks = []
        for client_id in client_ids:
            task = self.send_json(client_id, data)
            tasks.append(task)
        
        # Send all messages concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _buffer_message(self, client_id: str, message: Dict):
        """Buffer message for disconnected client"""
        buffer = self.message_buffers[client_id]
        buffer.append(message)
        
        # Keep only last 100 messages
        if len(buffer) > 100:
            self.message_buffers[client_id] = buffer[-100:]
    
    async def _send_buffered_messages(self, client_id: str):
        """Send all buffered messages to reconnected client"""
        if client_id in self.message_buffers:
            messages = self.message_buffers[client_id]
            
            if messages:
                # Send buffered messages indicator
                await self.send_json(client_id, {
                    "type": "buffered_messages_start",
                    "count": len(messages)
                }, save_to_buffer=False)
                
                # Send each message
                for message in messages:
                    await self.send_json(client_id, message, save_to_buffer=False)
                
                # Clear buffer
                del self.message_buffers[client_id]
                
                # Send completion indicator
                await self.send_json(client_id, {
                    "type": "buffered_messages_complete"
                }, save_to_buffer=False)
    
    async def _resend_messages_since(self, client_id: str, last_message_id: str):
        """Resend messages since specific message ID"""
        if client_id in self.message_buffers:
            messages = self.message_buffers[client_id]
            
            # Find index of last received message
            start_index = 0
            for i, msg in enumerate(messages):
                if msg.get("message_id") == last_message_id:
                    start_index = i + 1
                    break
            
            # Send messages after that point
            for message in messages[start_index:]:
                await self.send_json(client_id, message, save_to_buffer=False)
    
    async def _heartbeat_monitor(self):
        """Monitor connections and disconnect stale ones"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.utcnow()
                stale_clients = []
                
                for client_id, last_beat in self.last_heartbeat.items():
                    # If no heartbeat for 60 seconds, consider stale
                    if (current_time - last_beat).seconds > 60:
                        stale_clients.append(client_id)
                
                # Disconnect stale clients
                for client_id in stale_clients:
                    print(f"Disconnecting stale client: {client_id}")
                    await self.disconnect(client_id, save_state=True)
                    
            except Exception as e:
                print(f"Heartbeat monitor error: {e}")
    
    async def handle_heartbeat(self, client_id: str):
        """Handle heartbeat from client"""
        if client_id in self.active_connections:
            self.last_heartbeat[client_id] = datetime.utcnow()
            
            # Send pong
            await self.send_json(client_id, {
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            }, save_to_buffer=False)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "buffered_clients": len(self.message_buffers),
            "total_buffered_messages": sum(len(buffer) for buffer in self.message_buffers.values()),
            "clients_by_session": self._get_clients_by_session()
        }
    
    def _get_clients_by_session(self) -> Dict[str, int]:
        """Get count of clients by session"""
        session_counts = defaultdict(int)
        for meta in self.connection_metadata.values():
            session_id = meta.get("session_id", "unknown")
            session_counts[session_id] += 1
        return dict(session_counts)
    
    async def send_to_session(
        self,
        session_id: str,
        data: Dict[str, Any]
    ) -> int:
        """
        Send message to all clients in a session
        
        Returns: Number of clients message was sent to
        """
        client_ids = [
            cid for cid, meta in self.connection_metadata.items()
            if meta.get("session_id") == session_id
        ]
        
        sent_count = 0
        for client_id in client_ids:
            success = await self.send_json(client_id, data)
            if success:
                sent_count += 1
        
        return sent_count
    
    async def cleanup_old_buffers(self, hours_old: int = 24):
        """Clean up old message buffers"""
        try:
            cutoff_time = datetime.utcnow().timestamp() - (hours_old * 3600)
            
            clients_to_remove = []
            for client_id, buffer in self.message_buffers.items():
                if buffer:
                    # Check timestamp of oldest message
                    oldest_msg = buffer[0]
                    if "timestamp" in oldest_msg:
                        msg_time = datetime.fromisoformat(oldest_msg["timestamp"]).timestamp()
                        if msg_time < cutoff_time:
                            clients_to_remove.append(client_id)
            
            # Remove old buffers
            for client_id in clients_to_remove:
                del self.message_buffers[client_id]
                print(f"Cleaned up old buffer for client: {client_id}")
            
            return len(clients_to_remove)
            
        except Exception as e:
            print(f"Buffer cleanup error: {e}")
            return 0
    
    async def force_disconnect_all(self):
        """Force disconnect all clients (for shutdown)"""
        client_ids = list(self.active_connections.keys())
        for client_id in client_ids:
            try:
                await self.send_json(client_id, {
                    "type": "server_shutdown",
                    "message": "Server is shutting down"
                })
                await self.disconnect(client_id, save_state=False)
            except:
                pass
    
    def is_client_connected(self, client_id: str) -> bool:
        """Check if client is currently connected"""
        return client_id in self.active_connections
    
    def get_client_metadata(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific client"""
        return self.connection_metadata.get(client_id)

# Global instance
_connection_manager = None

def get_connection_manager() -> WebSocketConnectionManager:
    """Get singleton connection manager instance"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = WebSocketConnectionManager()
    return _connection_manager 