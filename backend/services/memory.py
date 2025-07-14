"""
Memory Service for AI Co-Scientist system
Handles database operations with PostgreSQL and Chroma vector database
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import hashlib
import uuid

from sqlmodel import SQLModel, create_engine, Session, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.core.config import settings
from backend.db.models import (
    ResearchGoal, Hypothesis, Review, TournamentMatch, 
    MetaReview, AgentActivity
)

# Optional Chroma integration
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

class MemoryService:
    """
    Service for managing persistent memory and context
    """
    
    def __init__(self):
        self.engine = None
        self.async_session = None
        self.chroma_client = None
        self.chroma_collection = None
        
        # In-memory fallback
        self.memory_store = {
            "research_goals": {},
            "hypotheses": {},
            "reviews": {},
            "tournaments": {},
            "meta_reviews": {},
            "agent_activities": {}
        }
        
        if settings.ENABLE_PERSISTENCE:
            self._init_database()
            if CHROMA_AVAILABLE:
                self._init_chroma()
    
    def _init_database(self):
        """Initialize PostgreSQL database connection"""
        try:
            # Create async engine
            self.engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.DEBUG,
                future=True
            )
            
            # Create async session factory
            self.async_session = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
        except Exception as e:
            print(f"Database initialization failed: {e}")
            print("Falling back to in-memory storage")
            self.engine = None
    
    def _init_chroma(self):
        """Initialize Chroma vector database"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PATH,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="coscientist_hypotheses",
                metadata={"description": "Hypothesis embeddings for similarity search"}
            )
            
        except Exception as e:
            print(f"Chroma initialization failed: {e}")
            self.chroma_client = None
    
    async def save_research_goal(self, research_goal: ResearchGoal) -> bool:
        """Save research goal to database"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    session.add(research_goal)
                    await session.commit()
                    return True
            else:
                # In-memory fallback
                self.memory_store["research_goals"][research_goal.id] = research_goal
                return True
                
        except Exception as e:
            print(f"Failed to save research goal: {e}")
            return False
    
    async def get_research_goal(self, goal_id: str) -> Optional[ResearchGoal]:
        """Get research goal by ID"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    result = await session.execute(
                        select(ResearchGoal).where(ResearchGoal.id == goal_id)
                    )
                    return result.scalar_one_or_none()
            else:
                return self.memory_store["research_goals"].get(goal_id)
                
        except Exception as e:
            print(f"Failed to get research goal: {e}")
            return None
    
    async def list_research_goals(self, limit: int = 10, offset: int = 0) -> List[ResearchGoal]:
        """List research goals with pagination"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    result = await session.execute(
                        select(ResearchGoal)
                        .order_by(ResearchGoal.created_at.desc())
                        .limit(limit)
                        .offset(offset)
                    )
                    return result.scalars().all()
            else:
                goals = list(self.memory_store["research_goals"].values())
                goals.sort(key=lambda x: x.created_at, reverse=True)
                return goals[offset:offset + limit]
                
        except Exception as e:
            print(f"Failed to list research goals: {e}")
            return []
    
    async def save_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """Save hypothesis to database and vector store"""
        try:
            # Save to database
            if self.async_session:
                async with self.async_session() as session:
                    session.add(hypothesis)
                    await session.commit()
            else:
                self.memory_store["hypotheses"][hypothesis.id] = hypothesis
            
            # Save to vector store for similarity search
            if self.chroma_collection:
                await self._add_to_vector_store(hypothesis)
            
            return True
            
        except Exception as e:
            print(f"Failed to save hypothesis: {e}")
            return False
    
    async def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get hypothesis by ID"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    result = await session.execute(
                        select(Hypothesis).where(Hypothesis.id == hypothesis_id)
                    )
                    return result.scalar_one_or_none()
            else:
                return self.memory_store["hypotheses"].get(hypothesis_id)
                
        except Exception as e:
            print(f"Failed to get hypothesis: {e}")
            return None
    
    async def get_hypotheses_by_goal(self, goal_id: str) -> List[Hypothesis]:
        """Get all hypotheses for a research goal"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    result = await session.execute(
                        select(Hypothesis)
                        .where(Hypothesis.research_goal_id == goal_id)
                        .order_by(Hypothesis.created_at.desc())
                    )
                    return result.scalars().all()
            else:
                return [
                    h for h in self.memory_store["hypotheses"].values()
                    if h.research_goal_id == goal_id
                ]
                
        except Exception as e:
            print(f"Failed to get hypotheses by goal: {e}")
            return []
    
    async def save_review(self, review: Review) -> bool:
        """Save review to database"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    session.add(review)
                    await session.commit()
                    return True
            else:
                self.memory_store["reviews"][review.id] = review
                return True
                
        except Exception as e:
            print(f"Failed to save review: {e}")
            return False
    
    async def get_reviews_by_hypothesis(self, hypothesis_id: str) -> List[Review]:
        """Get all reviews for a hypothesis"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    result = await session.execute(
                        select(Review)
                        .where(Review.hypothesis_id == hypothesis_id)
                        .order_by(Review.created_at.desc())
                    )
                    return result.scalars().all()
            else:
                return [
                    r for r in self.memory_store["reviews"].values()
                    if r.hypothesis_id == hypothesis_id
                ]
                
        except Exception as e:
            print(f"Failed to get reviews: {e}")
            return []
    
    async def save_tournament_match(self, match: TournamentMatch) -> bool:
        """Save tournament match result"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    session.add(match)
                    await session.commit()
                    return True
            else:
                self.memory_store["tournaments"][match.id] = match
                return True
                
        except Exception as e:
            print(f"Failed to save tournament match: {e}")
            return False
    
    async def get_tournament_history(self, goal_id: str) -> List[TournamentMatch]:
        """Get tournament history for a research goal"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    result = await session.execute(
                        select(TournamentMatch)
                        .where(TournamentMatch.research_goal_id == goal_id)
                        .order_by(TournamentMatch.created_at.desc())
                    )
                    return result.scalars().all()
            else:
                return [
                    t for t in self.memory_store["tournaments"].values()
                    if t.research_goal_id == goal_id
                ]
                
        except Exception as e:
            print(f"Failed to get tournament history: {e}")
            return []
    
    async def save_meta_review(self, meta_review: MetaReview) -> bool:
        """Save meta review"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    session.add(meta_review)
                    await session.commit()
                    return True
            else:
                self.memory_store["meta_reviews"][meta_review.id] = meta_review
                return True
                
        except Exception as e:
            print(f"Failed to save meta review: {e}")
            return False
    
    async def get_meta_reviews_by_goal(self, goal_id: str) -> List[MetaReview]:
        """Get meta reviews for a research goal"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    result = await session.execute(
                        select(MetaReview)
                        .where(MetaReview.research_goal_id == goal_id)
                        .order_by(MetaReview.created_at.desc())
                    )
                    return result.scalars().all()
            else:
                return [
                    m for m in self.memory_store["meta_reviews"].values()
                    if m.research_goal_id == goal_id
                ]
                
        except Exception as e:
            print(f"Failed to get meta reviews: {e}")
            return []
    
    async def save_agent_activity(self, activity: AgentActivity) -> bool:
        """Save agent activity for performance tracking"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    session.add(activity)
                    await session.commit()
                    return True
            else:
                self.memory_store["agent_activities"][activity.id] = activity
                return True
                
        except Exception as e:
            print(f"Failed to save agent activity: {e}")
            return False
    
    async def find_similar_hypotheses(
        self, 
        hypothesis_content: str, 
        limit: int = 5
    ) -> List[Tuple[Hypothesis, float]]:
        """Find similar hypotheses using vector similarity"""
        try:
            if not self.chroma_collection:
                return []
            
            # Query vector store
            results = self.chroma_collection.query(
                query_texts=[hypothesis_content],
                n_results=limit,
                include=['metadatas', 'distances']
            )
            
            similar_hypotheses = []
            for i, metadata in enumerate(results['metadatas'][0]):
                hypothesis_id = metadata['hypothesis_id']
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                hypothesis = await self.get_hypothesis(hypothesis_id)
                if hypothesis:
                    similar_hypotheses.append((hypothesis, similarity))
            
            return similar_hypotheses
            
        except Exception as e:
            print(f"Failed to find similar hypotheses: {e}")
            return []
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            stats = {
                "total_sessions": 0,
                "total_hypotheses": 0,
                "total_reviews": 0,
                "avg_composite_score": 0,
                "top_agents": [],
                "recent_activity": []
            }
            
            if self.async_session:
                async with self.async_session() as session:
                    # Count sessions
                    result = await session.execute(select(ResearchGoal))
                    stats["total_sessions"] = len(result.scalars().all())
                    
                    # Count hypotheses
                    result = await session.execute(select(Hypothesis))
                    hypotheses = result.scalars().all()
                    stats["total_hypotheses"] = len(hypotheses)
                    
                    # Average composite score
                    scores = [h.composite_score for h in hypotheses if h.composite_score]
                    if scores:
                        stats["avg_composite_score"] = sum(scores) / len(scores)
                    
                    # Count reviews
                    result = await session.execute(select(Review))
                    stats["total_reviews"] = len(result.scalars().all())
                    
                    # Top performing agents
                    agent_performance = {}
                    for h in hypotheses:
                        agent = h.created_by_agent
                        if agent not in agent_performance:
                            agent_performance[agent] = {"count": 0, "avg_score": 0}
                        agent_performance[agent]["count"] += 1
                        if h.composite_score:
                            agent_performance[agent]["avg_score"] += h.composite_score
                    
                    for agent, perf in agent_performance.items():
                        if perf["count"] > 0:
                            perf["avg_score"] /= perf["count"]
                    
                    stats["top_agents"] = sorted(
                        agent_performance.items(),
                        key=lambda x: x[1]["avg_score"],
                        reverse=True
                    )[:5]
            
            else:
                # In-memory stats
                stats["total_sessions"] = len(self.memory_store["research_goals"])
                stats["total_hypotheses"] = len(self.memory_store["hypotheses"])
                stats["total_reviews"] = len(self.memory_store["reviews"])
            
            return stats
            
        except Exception as e:
            print(f"Failed to get system stats: {e}")
            return {}
    
    async def get_step_results(self, session_id: str, step_name: str) -> Dict[str, Any]:
        """Get results for a specific step in advanced mode"""
        try:
            results = {"step": step_name, "results": []}
            
            if step_name == "hypothesis_generation":
                hypotheses = await self.get_hypotheses_by_goal(session_id)
                results["results"] = [
                    {
                        "id": h.id,
                        "content": h.content,
                        "summary": h.summary,
                        "created_by": h.created_by_agent
                    } for h in hypotheses
                ]
            
            elif step_name == "hypothesis_review":
                hypotheses = await self.get_hypotheses_by_goal(session_id)
                for h in hypotheses:
                    reviews = await self.get_reviews_by_hypothesis(h.id)
                    results["results"].append({
                        "hypothesis_id": h.id,
                        "reviews": [
                            {
                                "id": r.id,
                                "type": r.review_type,
                                "content": r.content,
                                "score": r.quality_score
                            } for r in reviews
                        ]
                    })
            
            elif step_name == "tournament_ranking":
                tournaments = await self.get_tournament_history(session_id)
                results["results"] = [
                    {
                        "id": t.id,
                        "winner": t.winner_id,
                        "participants": [t.hypothesis_1_id, t.hypothesis_2_id],
                        "rationale": t.comparison_rationale
                    } for t in tournaments
                ]
            
            return results
            
        except Exception as e:
            print(f"Failed to get step results: {e}")
            return {"step": step_name, "results": [], "error": str(e)}
    
    async def _add_to_vector_store(self, hypothesis: Hypothesis):
        """Add hypothesis to vector store for similarity search"""
        try:
            if not self.chroma_collection:
                return
            
            # Use OpenAI embeddings if available
            from backend.services.embeddings import get_embedding
            
            embedding = await get_embedding(hypothesis.content)
            
            self.chroma_collection.add(
                embeddings=[embedding],
                documents=[hypothesis.content],
                metadatas=[{
                    "hypothesis_id": hypothesis.id,
                    "research_goal_id": hypothesis.research_goal_id,
                    "created_by_agent": hypothesis.created_by_agent,
                    "generation_method": hypothesis.generation_method,
                    "composite_score": hypothesis.composite_score or 0
                }],
                ids=[hypothesis.id]
            )
            
        except Exception as e:
            print(f"Failed to add to vector store: {e}")
    
    async def update_hypothesis_scores(
        self, 
        hypothesis_id: str, 
        scores: Dict[str, float]
    ) -> bool:
        """Update hypothesis scores"""
        try:
            if self.async_session:
                async with self.async_session() as session:
                    result = await session.execute(
                        select(Hypothesis).where(Hypothesis.id == hypothesis_id)
                    )
                    hypothesis = result.scalar_one_or_none()
                    
                    if hypothesis:
                        hypothesis.novelty_score = scores.get("novelty")
                        hypothesis.feasibility_score = scores.get("feasibility")
                        hypothesis.impact_score = scores.get("impact")
                        hypothesis.testability_score = scores.get("testability")
                        hypothesis.composite_score = scores.get("composite")
                        hypothesis.confidence = scores.get("confidence")
                        
                        await session.commit()
                        return True
            else:
                hypothesis = self.memory_store["hypotheses"].get(hypothesis_id)
                if hypothesis:
                    hypothesis.novelty_score = scores.get("novelty")
                    hypothesis.feasibility_score = scores.get("feasibility")
                    hypothesis.impact_score = scores.get("impact")
                    hypothesis.testability_score = scores.get("testability")
                    hypothesis.composite_score = scores.get("composite")
                    hypothesis.confidence = scores.get("confidence")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Failed to update hypothesis scores: {e}")
            return False
    
    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old sessions and related data"""
        try:
            if not self.async_session:
                return 0
            
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            async with self.async_session() as session:
                # Get old sessions
                result = await session.execute(
                    select(ResearchGoal).where(ResearchGoal.created_at < cutoff_date)
                )
                old_goals = result.scalars().all()
                
                # Delete related data
                for goal in old_goals:
                    await session.delete(goal)
                
                await session.commit()
                return len(old_goals)
                
        except Exception as e:
            print(f"Failed to cleanup old sessions: {e}")
            return 0

# Singleton instance
_memory_service = None

def get_memory_service() -> MemoryService:
    """Get singleton memory service instance"""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service 