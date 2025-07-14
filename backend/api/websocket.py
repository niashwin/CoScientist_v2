"""
WebSocket endpoints for real-time streaming communication
"""

from fastapi import WebSocket, WebSocketDisconnect, Depends
from fastapi.routing import APIRouter
from typing import Dict, Any, Optional, Set
import json
import asyncio
import uuid
from datetime import datetime
from collections import defaultdict
import logging

from backend.core.config import settings
from backend.services.websocket_manager import WebSocketConnectionManager
from backend.services.task_queue import task_manager
from backend.services.memory import MemoryService
from backend.agents.swarm_orchestrator import SwarmOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter()

# Global connection manager (will be initialized when needed)
connection_manager = None

# Dependency injection
async def get_task_queue():
    return task_manager

async def get_memory_service():
    return MemoryService()

async def get_swarm_orchestrator():
    return SwarmOrchestrator()

def get_connection_manager():
    global connection_manager
    if connection_manager is None:
        connection_manager = WebSocketConnectionManager()
    return connection_manager

@router.websocket("/ws/auto-run")
async def websocket_auto_run_endpoint(
    websocket: WebSocket,
    task_queue = Depends(get_task_queue),
    memory_service: MemoryService = Depends(get_memory_service),
    swarm: SwarmOrchestrator = Depends(get_swarm_orchestrator)
):
    """
    WebSocket endpoint for simple mode with automatic reconnection support
    
    This endpoint handles the complete AI Co-Scientist pipeline with real-time
    streaming of agent outputs, hypothesis generation, and system updates.
    """
    client_id = None
    session_id = None
    
    try:
        # Get connection manager
        conn_mgr = get_connection_manager()
        
        # Check for reconnection attempt
        headers = dict(websocket.headers)
        existing_client_id = headers.get("x-client-id")
        last_message_id = headers.get("x-last-message-id")
        
        if existing_client_id:
            # Try to reconnect
            success = await conn_mgr.reconnect(
                websocket,
                existing_client_id,
                last_message_id
            )
            
            if success:
                client_id = existing_client_id
            else:
                # Failed reconnection - treat as new connection
                client_id = await conn_mgr.connect(websocket)
        else:
            # New connection
            client_id = await conn_mgr.connect(websocket)
        
        # Main message loop
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Handle different message types
            if data["type"] == "ping":
                await conn_mgr.handle_heartbeat(client_id)
                
            elif data["type"] == "start_research":
                session_id = await handle_start_research(
                    data, client_id, conn_mgr, swarm, memory_service
                )
                
            elif data["type"] == "pause":
                await handle_pause_research(
                    session_id, client_id, conn_mgr, task_queue
                )
                
            elif data["type"] == "resume":
                await handle_resume_research(
                    session_id, client_id, conn_mgr, task_queue
                )
                
            elif data["type"] == "stop":
                await handle_stop_research(
                    session_id, client_id, conn_mgr, swarm
                )
                
            elif data["type"] == "inject_hypothesis":
                await handle_inject_hypothesis(
                    data, session_id, client_id, conn_mgr, memory_service
                )
                
            elif data["type"] == "provide_feedback":
                await handle_provide_feedback(
                    data, session_id, client_id, conn_mgr, memory_service
                )
    
    except WebSocketDisconnect:
        # Client disconnected - save state for reconnection
        if client_id:
            await conn_mgr.disconnect(client_id, save_state=True)
    
    except Exception as e:
        # Unexpected error - send to client and disconnect
        if client_id:
            await conn_mgr.send_json(client_id, {
                "type": "error",
                "error": str(e),
                "code": "WEBSOCKET_ERROR",
                "suggestions": ["Please refresh and try again"]
            })
            await conn_mgr.disconnect(client_id, save_state=True)

@router.websocket("/ws/advanced")
async def websocket_advanced_endpoint(
    websocket: WebSocket,
    task_queue = Depends(get_task_queue),
    memory_service: MemoryService = Depends(get_memory_service),
    swarm: SwarmOrchestrator = Depends(get_swarm_orchestrator)
):
    """
    WebSocket endpoint for advanced mode with step-by-step control
    
    This endpoint allows users to control the AI Co-Scientist pipeline
    step by step, with the ability to pause, modify, and guide the process.
    """
    client_id = None
    session_id = None
    
    try:
        conn_mgr = get_connection_manager()
        client_id = await conn_mgr.connect(websocket)
        
        # Send initial capabilities
        await conn_mgr.send_json(client_id, {
            "type": "capabilities",
            "available_agents": [
                "Generation", "Reflection", "Ranking", 
                "Evolution", "Proximity", "MetaReview"
            ],
            "available_modes": [
                "literature_review", "scientific_debate", 
                "tournament", "evolution", "meta_analysis"
            ],
            "controls": [
                "pause", "resume", "skip_step", "modify_prompt",
                "inject_hypothesis", "provide_feedback"
            ]
        })
        
        # Main message loop for advanced mode
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "ping":
                await conn_mgr.handle_heartbeat(client_id)
                
            elif data["type"] == "start_session":
                session_id = await handle_start_advanced_session(
                    data, client_id, conn_mgr, memory_service
                )
                
            elif data["type"] == "run_agent":
                await handle_run_specific_agent(
                    data, session_id, client_id, conn_mgr, swarm
                )
                
            elif data["type"] == "modify_prompt":
                await handle_modify_prompt(
                    data, session_id, client_id, conn_mgr, swarm
                )
                
            elif data["type"] == "skip_step":
                await handle_skip_step(
                    data, session_id, client_id, conn_mgr
                )
                
            elif data["type"] == "get_step_results":
                await handle_get_step_results(
                    data, session_id, client_id, conn_mgr, memory_service
                )
    
    except WebSocketDisconnect:
        if client_id:
            await conn_mgr.disconnect(client_id, save_state=True)
    
    except Exception as e:
        if client_id:
            await conn_mgr.send_json(client_id, {
                "type": "error",
                "error": str(e),
                "code": "ADVANCED_WEBSOCKET_ERROR"
            })
            await conn_mgr.disconnect(client_id, save_state=True)

# Handler functions

async def handle_start_research(
    data: Dict[str, Any],
    client_id: str,
    connection_manager: WebSocketConnectionManager,
    swarm: SwarmOrchestrator,
    memory_service: MemoryService
) -> str:
    """Handle start research request"""
    session_id = str(uuid.uuid4())
    research_goal = data["goal"]
    preferences = data.get("preferences", {})
    
    # Create research session
    from backend.db.models import ResearchGoal
    research_session = ResearchGoal(
        id=session_id,
        goal_text=research_goal,
        preferences=preferences,
        status="active"
    )
    
    if settings.ENABLE_PERSISTENCE:
        await memory_service.save_research_goal(research_session)
    
    # Send confirmation
    await connection_manager.send_json(client_id, {
        "type": "session_started",
        "session_id": session_id,
        "goal": research_goal,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Start the research process asynchronously
    asyncio.create_task(
        run_research_pipeline(
            session_id, research_goal, preferences, 
            client_id, connection_manager, swarm, memory_service
        )
    )
    
    return session_id

async def run_research_pipeline(
    session_id: str,
    research_goal: str,
    preferences: Dict[str, Any],
    client_id: str,
    connection_manager: WebSocketConnectionManager,
    swarm: SwarmOrchestrator,
    memory_service: MemoryService
):
    """Run the complete research pipeline with streaming"""
    try:
        # Create streaming callback
        async def stream_callback(message: Dict[str, Any]):
            logger.info(f"Sending WebSocket message: {message.get('type', 'unknown')} - {message}")
            await connection_manager.send_json(client_id, message)
        
        # Process research goal with streaming
        async for update in swarm.process_research_goal(
            research_goal, 
            session_id=session_id,  # Add missing session_id parameter
            preferences=preferences,
            stream_callback=stream_callback
        ):
            # Also send the direct updates from the generator
            logger.info(f"Sending direct update: {update.get('type', 'unknown')} - {update}")
            await connection_manager.send_json(client_id, update)
        
        # Send completion
        await connection_manager.send_json(client_id, {
            "type": "research_complete",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Research pipeline completed successfully"
        })
        
    except Exception as e:
        await connection_manager.send_json(client_id, {
            "type": "error",
            "error": str(e),
            "session_id": session_id,
            "code": "PIPELINE_ERROR"
        })

async def handle_pause_research(
    session_id: str,
    client_id: str,
    connection_manager: WebSocketConnectionManager,
    task_queue
):
    """Handle pause research request"""
    if session_id:
        success = await task_queue.pause_session(session_id)
        await connection_manager.send_json(client_id, {
            "type": "session_paused" if success else "pause_failed",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })

async def handle_resume_research(
    session_id: str,
    client_id: str,
    connection_manager: WebSocketConnectionManager,
    task_queue
):
    """Handle resume research request"""
    if session_id:
        success = await task_queue.resume_session(session_id)
        await connection_manager.send_json(client_id, {
            "type": "session_resumed" if success else "resume_failed",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })

async def handle_stop_research(
    session_id: str,
    client_id: str,
    connection_manager: WebSocketConnectionManager,
    swarm: SwarmOrchestrator
):
    """Handle stop research request"""
    if session_id:
        # Request termination of the session
        swarm.swarm.request_session_termination(session_id)
        
        # Send confirmation
        await connection_manager.send_json(client_id, {
            "type": "session_stopped",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Research session termination requested"
        })
        
        logger.info(f"Stop requested for session {session_id}")

async def handle_inject_hypothesis(
    data: Dict[str, Any],
    session_id: str,
    client_id: str,
    connection_manager: WebSocketConnectionManager,
    memory_service: MemoryService
):
    """Handle user-injected hypothesis"""
    if not session_id:
        return
    
    hypothesis_content = data.get("hypothesis", "")
    if not hypothesis_content:
        return
    
    # Create hypothesis record
    from backend.db.models import Hypothesis
    hypothesis = Hypothesis(
        id=str(uuid.uuid4()),
        research_goal_id=session_id,
        content=hypothesis_content,
        summary=hypothesis_content[:200] + "..." if len(hypothesis_content) > 200 else hypothesis_content,
        generation_method="user_injection",
        created_by_agent="human_expert",
        generation=0
    )
    
    if settings.ENABLE_PERSISTENCE:
        await memory_service.save_hypothesis(hypothesis)
    
    await connection_manager.send_json(client_id, {
        "type": "hypothesis_injected",
        "hypothesis_id": hypothesis.id,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat()
    })

async def handle_provide_feedback(
    data: Dict[str, Any],
    session_id: str,
    client_id: str,
    connection_manager: WebSocketConnectionManager,
    memory_service: MemoryService
):
    """Handle user feedback on hypothesis"""
    hypothesis_id = data.get("hypothesis_id")
    feedback_content = data.get("feedback", "")
    
    if not hypothesis_id or not feedback_content:
        return
    
    # Create review record
    from backend.db.models import Review
    review = Review(
        id=str(uuid.uuid4()),
        hypothesis_id=hypothesis_id,
        review_type="user_feedback",
        content=feedback_content,
        critiques=[{"type": "user_input", "content": feedback_content}],
        suggestions=[],
        created_by_agent="human_expert"
    )
    
    if settings.ENABLE_PERSISTENCE:
        await memory_service.save_review(review)
    
    await connection_manager.send_json(client_id, {
        "type": "feedback_received",
        "review_id": review.id,
        "hypothesis_id": hypothesis_id,
        "timestamp": datetime.utcnow().isoformat()
    })

async def handle_start_advanced_session(
    data: Dict[str, Any],
    client_id: str,
    connection_manager: WebSocketConnectionManager,
    memory_service: MemoryService
) -> str:
    """Handle advanced session start"""
    session_id = str(uuid.uuid4())
    research_goal = data["goal"]
    
    # Create session
    from backend.db.models import ResearchGoal
    research_session = ResearchGoal(
        id=session_id,
        goal_text=research_goal,
        preferences={"mode": "advanced"},
        status="active"
    )
    
    if settings.ENABLE_PERSISTENCE:
        await memory_service.save_research_goal(research_session)
    
    await connection_manager.send_json(client_id, {
        "type": "advanced_session_started",
        "session_id": session_id,
        "goal": research_goal,
        "available_steps": [
            "literature_search",
            "hypothesis_generation",
            "hypothesis_review",
            "tournament_ranking",
            "hypothesis_evolution",
            "meta_review"
        ]
    })
    
    return session_id

async def handle_run_specific_agent(
    data: Dict[str, Any],
    session_id: str,
    client_id: str,
    connection_manager: WebSocketConnectionManager,
    swarm: SwarmOrchestrator
):
    """Handle running a specific agent"""
    agent_name = data.get("agent")
    parameters = data.get("parameters", {})
    
    if not agent_name or not session_id:
        return
    
    # Create streaming callback
    async def stream_callback(message: Dict[str, Any]):
        await connection_manager.send_json(client_id, message)
    
    # Run specific agent
    try:
        result = await swarm.run_agent(
            agent_name, 
            session_id, 
            parameters,
            stream_callback=stream_callback
        )
        
        await connection_manager.send_json(client_id, {
            "type": "agent_completed",
            "agent": agent_name,
            "session_id": session_id,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        await connection_manager.send_json(client_id, {
            "type": "agent_error",
            "agent": agent_name,
            "error": str(e),
            "session_id": session_id
        })

async def handle_modify_prompt(
    data: Dict[str, Any],
    session_id: str,
    client_id: str,
    connection_manager: WebSocketConnectionManager,
    swarm: SwarmOrchestrator
):
    """Handle prompt modification"""
    agent_name = data.get("agent")
    new_prompt = data.get("prompt")
    
    if not agent_name or not new_prompt:
        return
    
    # Update prompt in swarm
    success = await swarm.update_agent_prompt(agent_name, new_prompt)
    
    await connection_manager.send_json(client_id, {
        "type": "prompt_updated" if success else "prompt_update_failed",
        "agent": agent_name,
        "session_id": session_id
    })

async def handle_skip_step(
    data: Dict[str, Any],
    session_id: str,
    client_id: str,
    connection_manager: WebSocketConnectionManager
):
    """Handle skipping a step"""
    step_name = data.get("step")
    
    await connection_manager.send_json(client_id, {
        "type": "step_skipped",
        "step": step_name,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat()
    })

async def handle_get_step_results(
    data: Dict[str, Any],
    session_id: str,
    client_id: str,
    connection_manager: WebSocketConnectionManager,
    memory_service: MemoryService
):
    """Handle getting step results"""
    step_name = data.get("step")
    
    # Get results from memory
    results = await memory_service.get_step_results(session_id, step_name)
    
    await connection_manager.send_json(client_id, {
        "type": "step_results",
        "step": step_name,
        "session_id": session_id,
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }) 