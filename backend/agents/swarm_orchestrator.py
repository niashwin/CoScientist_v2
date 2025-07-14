"""
Swarm Orchestrator for AI Co-Scientist multi-agent system
"""

import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable
from datetime import datetime
import uuid
import logging

from backend.core.config import settings
from backend.services.llm import LLMService
from backend.services.memory import MemoryService
from backend.services.embeddings import EmbeddingsService
from backend.agents.base import CoScientistSwarm

logger = logging.getLogger(__name__)

class SwarmOrchestrator:
    """
    Orchestrates the multi-agent system using OpenAI Swarm framework
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.memory_service = MemoryService()
        self.embeddings_service = EmbeddingsService()
        
        # Initialize the actual swarm system
        self.swarm = CoScientistSwarm(
            llm_service=self.llm_service,
            memory_service=self.memory_service,
            embeddings_service=self.embeddings_service
        )
        
        # Active sessions
        self.active_sessions: Dict[str, Dict] = {}
    
    async def process_research_goal(
        self,
        research_goal: str,
        session_id: str = None,  # Add session_id parameter
        preferences: Dict[str, Any] = None,
        stream_callback: Optional[Callable] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a research goal through the complete pipeline using CoScientistSwarm
        
        Args:
            research_goal: The research question/goal
            session_id: Session identifier (if None, will generate one)
            preferences: Research preferences and constraints
            stream_callback: Callback for streaming updates
            
        Yields:
            Progress updates and results
        """
        # Use provided session_id or generate a new one
        if session_id is None:
            session_id = str(uuid.uuid4())
        preferences = preferences or {}
        
        try:
            # Initialize session
            self.active_sessions[session_id] = {
                "goal": research_goal,
                "preferences": preferences,
                "status": "active",
                "created_at": datetime.utcnow(),
                "hypotheses": [],
                "reviews": [],
                "tournaments": []
            }
            
            logger.info(f"Starting research session {session_id} with goal: {research_goal}")
            
            # Send initial update
            if stream_callback:
                await stream_callback({
                    "type": "session_started",
                    "session_id": session_id,
                    "message": "Starting AI Co-Scientist research pipeline..."
                })
            
            yield {
                "type": "session_started",
                "session_id": session_id,
                "message": "Research pipeline initialized"
            }
            
            # Use the actual swarm system to process the research goal
            async for update in self.swarm.process_research_goal(
                goal=research_goal,
                session_id=session_id,
                preferences=preferences,
                stream_callback=stream_callback
            ):
                # Update session state based on swarm updates
                await self._update_session_state(session_id, update)
                
                # Forward the update
                yield update
            
        except Exception as e:
            logger.error(f"Research pipeline failed for session {session_id}: {e}")
            error_msg = f"Research pipeline failed: {str(e)}"
            
            if stream_callback:
                await stream_callback({
                    "type": "error",
                    "session_id": session_id,
                    "error": error_msg
                })
            
            yield {
                "type": "error",
                "session_id": session_id,
                "error": error_msg
            }
        
        finally:
            # Clean up session
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = "completed"
                logger.info(f"Completed research session {session_id}")
    
    async def _update_session_state(self, session_id: str, update: Dict[str, Any]) -> None:
        """Update session state based on swarm updates"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Update based on update type
        if update["type"] == "hypothesis_generated":
            session["hypotheses"].append(update.get("hypothesis", {}))
        elif update["type"] == "review_complete":
            session["reviews"].append(update.get("review", {}))
        elif update["type"] == "ranking_complete":
            session["tournaments"].append(update.get("rankings", {}))
        elif update["type"] == "session_complete":
            session["status"] = "completed"
            session["completed_at"] = datetime.utcnow()
        elif update["type"] == "error":
            session["status"] = "failed"
            session["error"] = update.get("error", "Unknown error")
    
    async def run_agent(
        self,
        agent_name: str,
        session_id: str,
        parameters: Dict[str, Any],
        stream_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run a specific agent
        
        Args:
            agent_name: Name of the agent to run
            session_id: Session identifier
            parameters: Agent parameters
            stream_callback: Callback for streaming updates
            
        Returns:
            Agent execution result
        """
        try:
            if stream_callback:
                await stream_callback({
                    "type": "agent_started",
                    "agent": agent_name,
                    "session_id": session_id
                })
            
            # Simulate agent execution
            await asyncio.sleep(1)
            
            result = {
                "agent": agent_name,
                "session_id": session_id,
                "status": "completed",
                "output": f"Agent {agent_name} completed successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if stream_callback:
                await stream_callback({
                    "type": "agent_completed",
                    "agent": agent_name,
                    "result": result
                })
            
            return result
            
        except Exception as e:
            error_result = {
                "agent": agent_name,
                "session_id": session_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if stream_callback:
                await stream_callback({
                    "type": "agent_error",
                    "agent": agent_name,
                    "error": error_result
                })
            
            return error_result
    
    async def update_agent_prompt(self, agent_name: str, new_prompt: str) -> bool:
        """
        Update prompt for a specific agent
        
        Args:
            agent_name: Name of the agent
            new_prompt: New prompt template
            
        Returns:
            True if update successful
        """
        try:
            # This would update the agent's prompt in the actual implementation
            # For now, just return success
            return True
            
        except Exception as e:
            print(f"Failed to update prompt for {agent_name}: {e}")
            return False
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session"""
        return self.active_sessions.get(session_id)
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [
            {
                "session_id": sid,
                "goal": session["goal"],
                "status": session["status"],
                "created_at": session["created_at"].isoformat()
            }
            for sid, session in self.active_sessions.items()
        ]
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause a research session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "paused"
            return True
        return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused research session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "active"
            return True
        return False
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a research session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "cancelled"
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "active_sessions": len([
                s for s in self.active_sessions.values() 
                if s["status"] == "active"
            ]),
            "total_sessions": len(self.active_sessions),
            "available_agents": list(self.agents.keys())
        }

    async def _create_mock_hypothesis(self, content: str, agent_name: str) -> Dict[str, Any]:
        """Create a mock hypothesis for testing"""
        from backend.services.database import database_service
        
        # Create a mock hypothesis object
        class MockHypothesis:
            def __init__(self, content: str, agent_name: str):
                self.id = f"hyp_{hash(content) % 100000}"
                self.content = content
                self.summary = content[:100] + "..." if len(content) > 100 else content
                self.generation_method = "swarm_generation"
                self.created_at = datetime.utcnow()
                self.created_by_agent = agent_name
                
                # Mock scores (0-1 scale internally)
                self.novelty_score = 0.8
                self.feasibility_score = 0.7
                self.impact_score = 0.8
                self.testability_score = 0.8
                self.composite_score = 0.775  # Weighted average
                self.confidence = 0.85
                
                # Tournament data
                self.tournament_wins = 0
                self.tournament_losses = 0
                self.elo_rating = 1200.0
                
                # Evolution data
                self.generation = 0
                self.parent_id = None
                self.evolution_method = None
                
                # Additional data
                self.supporting_literature = []
                self.experimental_protocol = {}
        
        mock_hypothesis = MockHypothesis(content, agent_name)
        
        # Use the database service serialization method to get proper 1-10 scale scores
        return database_service._serialize_hypothesis(mock_hypothesis)

# Singleton instance
_swarm_orchestrator = None

def get_swarm_orchestrator() -> SwarmOrchestrator:
    """Get singleton swarm orchestrator instance"""
    global _swarm_orchestrator
    if _swarm_orchestrator is None:
        _swarm_orchestrator = SwarmOrchestrator()
    return _swarm_orchestrator 