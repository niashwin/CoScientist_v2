"""
REST API endpoints for AI Co-Scientist system
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import asyncio
from datetime import datetime
import uuid

from backend.core.config import settings
from backend.db.models import ResearchGoal, Hypothesis, Review, TournamentMatch, MetaReview
from backend.services.llm import LLMService
from backend.services.memory import MemoryService
from backend.services.task_queue import task_manager

router = APIRouter()

# Request/Response Models
class ResearchRequest(BaseModel):
    goal: str = Field(..., description="Research goal or question")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Research preferences")
    max_hypotheses: int = Field(default=5, ge=1, le=20, description="Maximum number of hypotheses to generate")
    mode: str = Field(default="simple", description="Execution mode: simple or advanced")

class HypothesisResponse(BaseModel):
    id: str
    content: str
    summary: str
    scores: Dict[str, float]
    created_at: str
    created_by_agent: str
    supporting_literature: List[Dict[str, Any]]

class SessionResponse(BaseModel):
    session_id: str
    goal: str
    status: str
    created_at: str
    hypotheses: List[HypothesisResponse]
    progress: float
    current_stage: str

class FeedbackRequest(BaseModel):
    hypothesis_id: str
    feedback_type: str = Field(..., description="Type of feedback: review, correction, suggestion")
    content: str = Field(..., description="Feedback content")
    ratings: Dict[str, int] = Field(default_factory=dict, description="Numerical ratings")

class ErrorResponse(BaseModel):
    error: str
    code: str
    suggestions: List[str] = Field(default_factory=list)

# Dependency injection
async def get_llm_service():
    return LLMService()

async def get_memory_service():
    return MemoryService()

async def get_task_queue():
    return task_manager

@router.post("/run-simple", response_class=StreamingResponse)
async def run_simple_mode(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    llm_service: LLMService = Depends(get_llm_service),
    memory_service: MemoryService = Depends(get_memory_service),
    task_queue = Depends(get_task_queue)
):
    """
    Run simple mode with Server-Sent Events streaming
    
    This endpoint starts the complete AI Co-Scientist pipeline and streams
    results back to the client in real-time using Server-Sent Events.
    """
    session_id = str(uuid.uuid4())
    
    try:
        # Create research session
        research_goal = ResearchGoal(
            id=session_id,
            goal_text=request.goal,
            preferences=request.preferences,
            status="initializing"
        )
        
        if settings.ENABLE_PERSISTENCE:
            await memory_service.save_research_goal(research_goal)
        
        async def generate_stream():
            """Generate Server-Sent Events stream"""
            try:
                # Send initial status
                yield f"data: {json.dumps({'type': 'status', 'message': 'Initializing AI Co-Scientist system...', 'session_id': session_id})}\n\n"
                
                # Import SwarmOrchestrator here to avoid circular imports
                from backend.agents.swarm_orchestrator import SwarmOrchestrator
                
                # Initialize orchestrator
                orchestrator = SwarmOrchestrator()
                
                yield f"data: {json.dumps({'type': 'status', 'message': 'Starting research pipeline...', 'session_id': session_id})}\n\n"
                
                # Create streaming callback
                stream_chunks = []
                async def stream_callback(message: Dict[str, Any]):
                    stream_chunks.append(f"data: {json.dumps(message)}\n\n")
                
                # Process research goal with streaming
                try:
                    async for chunk in orchestrator.process_research_goal(
                        request.goal,
                        preferences=request.preferences,
                        stream_callback=stream_callback
                    ):
                        # Yield any collected chunks
                        while stream_chunks:
                            yield stream_chunks.pop(0)
                        
                        # Also yield the direct chunk
                        yield f"data: {json.dumps(chunk)}\n\n"
                        
                    # Send completion
                    yield f"data: {json.dumps({'type': 'complete', 'session_id': session_id, 'message': 'Research completed successfully'})}\n\n"
                    
                except Exception as e:
                    # Send error
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'session_id': session_id})}\n\n"
                        
            except Exception as e:
                error_data = {
                    'type': 'error',
                    'error': str(e),
                    'code': 'STREAM_ERROR',
                    'suggestions': ['Please try again', 'Check your research goal format']
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=str(e),
                code="SIMPLE_MODE_ERROR",
                suggestions=["Check your input parameters", "Try again with a simpler research goal"]
            ).dict()
        )

@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Get session details and current status"""
    try:
        research_goal = await memory_service.get_research_goal(session_id)
        if not research_goal:
            raise HTTPException(status_code=404, detail="Session not found")
        
        hypotheses = await memory_service.get_hypotheses_by_goal(session_id)
        
        return SessionResponse(
            session_id=session_id,
            goal=research_goal.goal_text,
            status=research_goal.status,
            created_at=research_goal.created_at.isoformat(),
            hypotheses=[
                HypothesisResponse(
                    id=h.id,
                    content=h.content,
                    summary=h.summary,
                    scores={
                        "novelty": h.novelty_score or 0,
                        "feasibility": h.feasibility_score or 0,
                        "impact": h.impact_score or 0,
                        "testability": h.testability_score or 0,
                        "composite": h.composite_score or 0
                    },
                    created_at=h.created_at.isoformat(),
                    created_by_agent=h.created_by_agent,
                    supporting_literature=h.supporting_literature
                ) for h in hypotheses
            ],
            progress=research_goal.preferences.get("progress", 0.0),
            current_stage=research_goal.preferences.get("current_stage", "initializing")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hypothesis/{hypothesis_id}")
async def get_hypothesis_details(
    hypothesis_id: str,
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Get detailed information about a specific hypothesis"""
    try:
        hypothesis = await memory_service.get_hypothesis(hypothesis_id)
        if not hypothesis:
            raise HTTPException(status_code=404, detail="Hypothesis not found")
        
        reviews = await memory_service.get_reviews_by_hypothesis(hypothesis_id)
        
        return {
            "id": hypothesis.id,
            "content": hypothesis.content,
            "summary": hypothesis.summary,
            "scores": {
                "novelty": hypothesis.novelty_score,
                "feasibility": hypothesis.feasibility_score,
                "impact": hypothesis.impact_score,
                "testability": hypothesis.testability_score,
                "composite": hypothesis.composite_score
            },
            "confidence": hypothesis.confidence,
            "reviews": [
                {
                    "id": r.id,
                    "type": r.review_type,
                    "content": r.content,
                    "score": r.quality_score,
                    "created_at": r.created_at.isoformat()
                } for r in reviews
            ],
            "lineage": {
                "parent_id": hypothesis.parent_id,
                "evolution_method": hypothesis.evolution_method,
                "generation": hypothesis.generation
            },
            "supporting_literature": hypothesis.supporting_literature,
            "experimental_protocol": hypothesis.experimental_protocol,
            "tournament_stats": {
                "wins": hypothesis.tournament_wins,
                "losses": hypothesis.tournament_losses,
                "elo_rating": hypothesis.elo_rating
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Submit expert feedback on a hypothesis"""
    try:
        # Verify hypothesis exists
        hypothesis = await memory_service.get_hypothesis(feedback.hypothesis_id)
        if not hypothesis:
            raise HTTPException(status_code=404, detail="Hypothesis not found")
        
        # Create review record
        review = Review(
            id=str(uuid.uuid4()),
            hypothesis_id=feedback.hypothesis_id,
            review_type="expert_feedback",
            content=feedback.content,
            critiques=[{"type": feedback.feedback_type, "content": feedback.content}],
            suggestions=[],
            created_by_agent="human_expert"
        )
        
        # Add ratings if provided
        if feedback.ratings:
            review.quality_score = sum(feedback.ratings.values()) / len(feedback.ratings)
        
        if settings.ENABLE_PERSISTENCE:
            await memory_service.save_review(review)
        
        return {
            "message": "Feedback submitted successfully",
            "review_id": review.id,
            "timestamp": review.created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def list_sessions(
    limit: int = 10,
    offset: int = 0,
    memory_service: MemoryService = Depends(get_memory_service)
):
    """List recent research sessions"""
    try:
        sessions = await memory_service.list_research_goals(limit=limit, offset=offset)
        
        return {
            "sessions": [
                {
                    "id": s.id,
                    "goal": s.goal_text,
                    "status": s.status,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat()
                } for s in sessions
            ],
            "total": len(sessions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/pause")
async def pause_session(
    session_id: str,
    task_queue = Depends(get_task_queue)
):
    """Pause an active research session"""
    try:
        # For now, just return success - actual pause logic would need session tracking
        return {"message": f"Session {session_id} pause requested", "note": "Pause functionality requires session-specific task tracking"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/resume")
async def resume_session(
    session_id: str,
    task_queue = Depends(get_task_queue)
):
    """Resume a paused research session"""
    try:
        # For now, just return success - actual resume logic would need session tracking
        return {"message": f"Session {session_id} resume requested", "note": "Resume functionality requires session-specific task tracking"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "llm": "available" if settings.ANTHROPIC_API_KEY else "unavailable",
            "embeddings": "available" if settings.OPENAI_API_KEY else "unavailable",
            "persistence": "enabled" if settings.ENABLE_PERSISTENCE else "disabled",
            "auth": "enabled" if settings.AUTH_ENABLED else "disabled"
        }
    }

@router.get("/stats")
async def get_system_stats(
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Get system statistics"""
    try:
        stats = await memory_service.get_system_stats()
        return {
            "total_sessions": stats.get("total_sessions", 0),
            "total_hypotheses": stats.get("total_hypotheses", 0),
            "total_reviews": stats.get("total_reviews", 0),
            "avg_composite_score": stats.get("avg_composite_score", 0),
            "top_performing_agents": stats.get("top_agents", []),
            "recent_activity": stats.get("recent_activity", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@router.get("/literature-logs/{session_id}")
async def get_literature_logs(session_id: str):
    """Get literature search logs for a specific session"""
    from backend.services.literature_logger import get_session_logs, get_log_summary
    
    logs = get_session_logs(session_id)
    summary = get_log_summary(session_id)
    
    return {
        "session_id": session_id,
        "summary": summary,
        "logs": [
            {
                "timestamp": log.timestamp,
                "event_type": log.event_type,
                "query": log.query,
                "stage": log.stage,
                "paper_count": log.paper_count,
                "message": log.message,
                "error": log.error,
                "source": log.source
            }
            for log in logs
        ]
    }

@router.get("/literature-logs")
async def get_recent_literature_logs(limit: int = 50):
    """Get recent literature search logs across all sessions"""
    from backend.services.literature_logger import literature_logger
    
    logs = literature_logger.get_recent_logs(limit)
    
    return {
        "logs": [
            {
                "timestamp": log.timestamp,
                "session_id": log.session_id,
                "event_type": log.event_type,
                "query": log.query,
                "stage": log.stage,
                "paper_count": log.paper_count,
                "message": log.message,
                "error": log.error,
                "source": log.source
            }
            for log in logs
        ]
    }

@router.delete("/literature-logs/{session_id}")
async def clear_literature_logs(session_id: str):
    """Clear literature search logs for a specific session"""
    from backend.services.literature_logger import clear_session_logs
    
    clear_session_logs(session_id)
    
    return {"message": f"Literature logs cleared for session {session_id}"} 