"""
Database integration service for PostgreSQL and Chroma vector database.
Provides comprehensive data persistence and vector search capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import uuid

try:
    import asyncpg
    from asyncpg import Pool
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from backend.core.config import settings
from backend.core.exceptions import CoScientistError, ErrorSeverity
from backend.services.embeddings import EmbeddingsService

logger = logging.getLogger(__name__)

@dataclass
class ResearchSession:
    """Research session data model"""
    id: str
    goal: str
    preferences: Dict[str, Any]
    status: str = "active"
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

@dataclass
class Hypothesis:
    """Hypothesis data model"""
    id: str
    research_session_id: str
    content: str
    summary: str
    generation_method: str
    novelty_score: Optional[float] = None
    feasibility_score: Optional[float] = None
    impact_score: Optional[float] = None
    testability_score: Optional[float] = None
    composite_score: Optional[float] = None
    confidence: Optional[float] = None
    parent_id: Optional[str] = None
    evolution_method: Optional[str] = None
    generation: int = 0
    created_at: datetime = None
    created_by_agent: str = "unknown"
    supporting_literature: List[Dict] = None
    experimental_protocol: Optional[Dict] = None
    tournament_wins: int = 0
    tournament_losses: int = 0
    elo_rating: float = 1200.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.supporting_literature is None:
            self.supporting_literature = []

@dataclass
class Review:
    """Review data model"""
    id: str
    hypothesis_id: str
    review_type: str
    content: str
    critiques: List[Dict] = None
    suggestions: List[str] = None
    correctness_score: Optional[float] = None
    novelty_assessment: Optional[str] = None
    quality_score: Optional[float] = None
    safety_concerns: List[str] = None
    created_at: datetime = None
    created_by_agent: str = "unknown"
    literature_cited: List[Dict] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.critiques is None:
            self.critiques = []
        if self.suggestions is None:
            self.suggestions = []
        if self.safety_concerns is None:
            self.safety_concerns = []
        if self.literature_cited is None:
            self.literature_cited = []

class PostgreSQLService:
    """PostgreSQL database service"""
    
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.connected = False
    
    async def connect(self):
        """Connect to PostgreSQL database"""
        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available, PostgreSQL disabled")
            return
        
        try:
            self.pool = await asyncpg.create_pool(
                settings.POSTGRES_URL,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            self.connected = True
            logger.info("PostgreSQL connection established")
            
            # Create tables if they don't exist
            await self._create_tables()
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from PostgreSQL"""
        if self.pool:
            await self.pool.close()
            self.connected = False
            logger.info("PostgreSQL connection closed")
    
    async def _create_tables(self):
        """Create database tables"""
        async with self.pool.acquire() as conn:
            # Research sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_sessions (
                    id VARCHAR(255) PRIMARY KEY,
                    goal TEXT NOT NULL,
                    preferences JSONB,
                    status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Hypotheses table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id VARCHAR(255) PRIMARY KEY,
                    research_session_id VARCHAR(255) REFERENCES research_sessions(id),
                    content TEXT NOT NULL,
                    summary TEXT,
                    generation_method VARCHAR(100),
                    novelty_score FLOAT,
                    feasibility_score FLOAT,
                    impact_score FLOAT,
                    testability_score FLOAT,
                    composite_score FLOAT,
                    confidence FLOAT,
                    parent_id VARCHAR(255) REFERENCES hypotheses(id),
                    evolution_method VARCHAR(100),
                    generation INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    created_by_agent VARCHAR(100),
                    supporting_literature JSONB,
                    experimental_protocol JSONB,
                    tournament_wins INTEGER DEFAULT 0,
                    tournament_losses INTEGER DEFAULT 0,
                    elo_rating FLOAT DEFAULT 1200.0
                )
            """)
            
            # Reviews table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id VARCHAR(255) PRIMARY KEY,
                    hypothesis_id VARCHAR(255) REFERENCES hypotheses(id),
                    review_type VARCHAR(50),
                    content TEXT NOT NULL,
                    critiques JSONB,
                    suggestions JSONB,
                    correctness_score FLOAT,
                    novelty_assessment VARCHAR(50),
                    quality_score FLOAT,
                    safety_concerns JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    created_by_agent VARCHAR(100),
                    literature_cited JSONB
                )
            """)
            
            # Tournament matches table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tournament_matches (
                    id VARCHAR(255) PRIMARY KEY,
                    research_session_id VARCHAR(255) REFERENCES research_sessions(id),
                    hypothesis_1_id VARCHAR(255) REFERENCES hypotheses(id),
                    hypothesis_2_id VARCHAR(255) REFERENCES hypotheses(id),
                    winner_id VARCHAR(255) REFERENCES hypotheses(id),
                    comparison_type VARCHAR(50),
                    debate_transcript TEXT,
                    comparison_rationale TEXT,
                    criteria_scores JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    round_number INTEGER
                )
            """)
            
            # Meta reviews table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS meta_reviews (
                    id VARCHAR(255) PRIMARY KEY,
                    research_session_id VARCHAR(255) REFERENCES research_sessions(id),
                    common_issues JSONB,
                    improvement_patterns JSONB,
                    success_patterns JSONB,
                    research_overview TEXT,
                    key_insights JSONB,
                    future_directions JSONB,
                    suggested_contacts JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    hypotheses_analyzed INTEGER,
                    reviews_synthesized INTEGER
                )
            """)
            
            # Agent activity table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_activity (
                    id VARCHAR(255) PRIMARY KEY,
                    agent_name VARCHAR(100),
                    action_type VARCHAR(100),
                    research_session_id VARCHAR(255) REFERENCES research_sessions(id),
                    execution_time_ms INTEGER,
                    success BOOLEAN,
                    error_message TEXT,
                    input_context JSONB,
                    output_quality_score FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hypotheses_session 
                ON hypotheses(research_session_id)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hypotheses_score 
                ON hypotheses(composite_score DESC)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviews_hypothesis 
                ON reviews(hypothesis_id)
            """)
            
            logger.info("Database tables created successfully")
    
    async def create_research_session(self, session: ResearchSession) -> str:
        """Create a new research session"""
        if not self.connected:
            raise CoScientistError("Database not connected", "DB_001", ErrorSeverity.HIGH)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO research_sessions (id, goal, preferences, status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, session.id, session.goal, json.dumps(session.preferences), 
                session.status, session.created_at, session.updated_at)
        
        return session.id
    
    async def get_research_session(self, session_id: str) -> Optional[ResearchSession]:
        """Get a research session by ID"""
        if not self.connected:
            return None
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, goal, preferences, status, created_at, updated_at
                FROM research_sessions WHERE id = $1
            """, session_id)
            
            if row:
                return ResearchSession(
                    id=row['id'],
                    goal=row['goal'],
                    preferences=row['preferences'],
                    status=row['status'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
        
        return None
    
    async def create_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Create a new hypothesis"""
        if not self.connected:
            raise CoScientistError("Database not connected", "DB_001", ErrorSeverity.HIGH)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO hypotheses (
                    id, research_session_id, content, summary, generation_method,
                    novelty_score, feasibility_score, impact_score, testability_score,
                    composite_score, confidence, parent_id, evolution_method, generation,
                    created_at, created_by_agent, supporting_literature, experimental_protocol,
                    tournament_wins, tournament_losses, elo_rating
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
            """, hypothesis.id, hypothesis.research_session_id, hypothesis.content,
                hypothesis.summary, hypothesis.generation_method, hypothesis.novelty_score,
                hypothesis.feasibility_score, hypothesis.impact_score, hypothesis.testability_score,
                hypothesis.composite_score, hypothesis.confidence, hypothesis.parent_id,
                hypothesis.evolution_method, hypothesis.generation, hypothesis.created_at,
                hypothesis.created_by_agent, json.dumps(hypothesis.supporting_literature),
                json.dumps(hypothesis.experimental_protocol), hypothesis.tournament_wins,
                hypothesis.tournament_losses, hypothesis.elo_rating)
        
        return hypothesis.id
    
    async def get_hypotheses(self, session_id: str, limit: int = 100) -> List[Hypothesis]:
        """Get hypotheses for a research session"""
        if not self.connected:
            return []
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM hypotheses 
                WHERE research_session_id = $1 
                ORDER BY composite_score DESC NULLS LAST, created_at DESC
                LIMIT $2
            """, session_id, limit)
            
            hypotheses = []
            for row in rows:
                hypothesis = Hypothesis(
                    id=row['id'],
                    research_session_id=row['research_session_id'],
                    content=row['content'],
                    summary=row['summary'],
                    generation_method=row['generation_method'],
                    novelty_score=row['novelty_score'],
                    feasibility_score=row['feasibility_score'],
                    impact_score=row['impact_score'],
                    testability_score=row['testability_score'],
                    composite_score=row['composite_score'],
                    confidence=row['confidence'],
                    parent_id=row['parent_id'],
                    evolution_method=row['evolution_method'],
                    generation=row['generation'],
                    created_at=row['created_at'],
                    created_by_agent=row['created_by_agent'],
                    supporting_literature=row['supporting_literature'] or [],
                    experimental_protocol=row['experimental_protocol'],
                    tournament_wins=row['tournament_wins'],
                    tournament_losses=row['tournament_losses'],
                    elo_rating=row['elo_rating']
                )
                hypotheses.append(hypothesis)
            
            return hypotheses
    
    async def update_hypothesis_scores(self, hypothesis_id: str, scores: Dict[str, float]):
        """Update hypothesis scores"""
        if not self.connected:
            return
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE hypotheses SET
                    novelty_score = $2,
                    feasibility_score = $3,
                    impact_score = $4,
                    testability_score = $5,
                    composite_score = $6,
                    confidence = $7,
                    updated_at = NOW()
                WHERE id = $1
            """, hypothesis_id, scores.get('novelty'), scores.get('feasibility'),
                scores.get('impact'), scores.get('testability'), scores.get('composite'),
                scores.get('confidence'))
    
    async def create_review(self, review: Review) -> str:
        """Create a new review"""
        if not self.connected:
            raise CoScientistError("Database not connected", "DB_001", ErrorSeverity.HIGH)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO reviews (
                    id, hypothesis_id, review_type, content, critiques, suggestions,
                    correctness_score, novelty_assessment, quality_score, safety_concerns,
                    created_at, created_by_agent, literature_cited
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """, review.id, review.hypothesis_id, review.review_type, review.content,
                json.dumps(review.critiques), json.dumps(review.suggestions),
                review.correctness_score, review.novelty_assessment, review.quality_score,
                json.dumps(review.safety_concerns), review.created_at, review.created_by_agent,
                json.dumps(review.literature_cited))
        
        return review.id
    
    async def get_reviews(self, hypothesis_id: str) -> List[Review]:
        """Get reviews for a hypothesis"""
        if not self.connected:
            return []
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM reviews 
                WHERE hypothesis_id = $1 
                ORDER BY created_at DESC
            """, hypothesis_id)
            
            reviews = []
            for row in rows:
                review = Review(
                    id=row['id'],
                    hypothesis_id=row['hypothesis_id'],
                    review_type=row['review_type'],
                    content=row['content'],
                    critiques=row['critiques'] or [],
                    suggestions=row['suggestions'] or [],
                    correctness_score=row['correctness_score'],
                    novelty_assessment=row['novelty_assessment'],
                    quality_score=row['quality_score'],
                    safety_concerns=row['safety_concerns'] or [],
                    created_at=row['created_at'],
                    created_by_agent=row['created_by_agent'],
                    literature_cited=row['literature_cited'] or []
                )
                reviews.append(review)
            
            return reviews
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.connected:
            return {"error": "Database not connected"}
        
        async with self.pool.acquire() as conn:
            # Get table counts
            sessions_count = await conn.fetchval("SELECT COUNT(*) FROM research_sessions")
            hypotheses_count = await conn.fetchval("SELECT COUNT(*) FROM hypotheses")
            reviews_count = await conn.fetchval("SELECT COUNT(*) FROM reviews")
            
            # Get recent activity
            recent_sessions = await conn.fetchval("""
                SELECT COUNT(*) FROM research_sessions 
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            
            return {
                "connected": True,
                "tables": {
                    "research_sessions": sessions_count,
                    "hypotheses": hypotheses_count,
                    "reviews": reviews_count
                },
                "recent_activity": {
                    "sessions_24h": recent_sessions
                }
            }

class ChromaService:
    """Chroma vector database service"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.connected = False
        self.embedding_service = EmbeddingsService()
    
    async def connect(self):
        """Connect to Chroma database"""
        if not CHROMA_AVAILABLE:
            logger.warning("chromadb not available, vector search disabled")
            return
        
        try:
            # Initialize Chroma client
            if hasattr(settings, 'CHROMA_PATH'):
                self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
            else:
                self.client = chromadb.Client()
            
            # Get or create collection
            collection_name = getattr(settings, 'CHROMA_COLLECTION_NAME', 'coscientist_embeddings')
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "AI Co-Scientist embeddings"}
            )
            
            self.connected = True
            logger.info("Chroma vector database connected")
            
        except Exception as e:
            logger.error(f"Failed to connect to Chroma: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from Chroma"""
        if self.client:
            # Chroma client doesn't need explicit disconnection
            self.connected = False
            logger.info("Chroma connection closed")
    
    async def add_hypothesis_embedding(self, hypothesis: Hypothesis):
        """Add hypothesis embedding to vector database"""
        if not self.connected:
            return
        
        try:
            # Generate embedding for hypothesis content
            embedding = await self.embedding_service.get_embedding(
                hypothesis.content
            )
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[hypothesis.content],
                metadatas=[{
                    "id": hypothesis.id,
                    "research_session_id": hypothesis.research_session_id,
                    "summary": hypothesis.summary,
                    "generation_method": hypothesis.generation_method,
                    "created_by_agent": hypothesis.created_by_agent,
                    "composite_score": hypothesis.composite_score or 0.0,
                    "type": "hypothesis"
                }],
                ids=[hypothesis.id]
            )
            
            logger.debug(f"Added hypothesis {hypothesis.id} to vector database")
            
        except Exception as e:
            logger.error(f"Failed to add hypothesis embedding: {e}")
    
    async def add_literature_embedding(self, paper_id: str, title: str, abstract: str, metadata: Dict[str, Any]):
        """Add literature paper embedding to vector database"""
        if not self.connected:
            return
        
        try:
            # Generate embedding for paper content
            content = f"{title}\n\n{abstract}"
            embedding = await self.embedding_service.get_embedding(
                content
            )
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[{
                    "id": paper_id,
                    "title": title,
                    "type": "literature",
                    **metadata
                }],
                ids=[f"paper_{paper_id}"]
            )
            
            logger.debug(f"Added paper {paper_id} to vector database")
            
        except Exception as e:
            logger.error(f"Failed to add literature embedding: {e}")
    
    async def search_similar_hypotheses(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar hypotheses"""
        if not self.connected:
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.get_embedding(
                query
            )
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"type": "hypothesis"}
            )
            
            # Format results
            similar_hypotheses = []
            for i, doc_id in enumerate(results['ids'][0]):
                similar_hypotheses.append({
                    "id": doc_id,
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity": 1.0 - results['distances'][0][i]  # Convert distance to similarity
                })
            
            return similar_hypotheses
            
        except Exception as e:
            logger.error(f"Failed to search similar hypotheses: {e}")
            return []
    
    async def search_relevant_literature(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant literature"""
        if not self.connected:
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.get_embedding(
                query
            )
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"type": "literature"}
            )
            
            # Format results
            relevant_papers = []
            for i, doc_id in enumerate(results['ids'][0]):
                relevant_papers.append({
                    "id": doc_id,
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity": 1.0 - results['distances'][0][i]
                })
            
            return relevant_papers
            
        except Exception as e:
            logger.error(f"Failed to search relevant literature: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.connected:
            return {"error": "Chroma not connected"}
        
        try:
            count = self.collection.count()
            
            return {
                "connected": True,
                "collection_name": self.collection.name,
                "document_count": count,
                "embedding_dimension": 1536  # OpenAI text-embedding-3-large
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

class DatabaseService:
    """
    High-level database service combining PostgreSQL and Chroma
    """
    
    def __init__(self):
        self.postgres = PostgreSQLService()
        self.chroma = ChromaService()
    
    async def connect(self):
        """Connect to both databases"""
        await self.postgres.connect()
        await self.chroma.connect()
    
    async def disconnect(self):
        """Disconnect from both databases"""
        await self.postgres.disconnect()
        await self.chroma.disconnect()
    
    async def create_research_session(self, goal: str, preferences: Dict[str, Any]) -> str:
        """Create a new research session"""
        session_id = str(uuid.uuid4())
        session = ResearchSession(
            id=session_id,
            goal=goal,
            preferences=preferences
        )
        
        if self.postgres.connected:
            await self.postgres.create_research_session(session)
        
        return session_id
    
    async def create_hypothesis(
        self,
        research_session_id: str,
        content: str,
        summary: str,
        generation_method: str,
        created_by_agent: str,
        **kwargs
    ) -> str:
        """Create a new hypothesis with vector embedding"""
        hypothesis_id = str(uuid.uuid4())
        hypothesis = Hypothesis(
            id=hypothesis_id,
            research_session_id=research_session_id,
            content=content,
            summary=summary,
            generation_method=generation_method,
            created_by_agent=created_by_agent,
            **kwargs
        )
        
        # Store in PostgreSQL
        if self.postgres.connected:
            await self.postgres.create_hypothesis(hypothesis)
        
        # Store embedding in Chroma
        if self.chroma.connected:
            await self.chroma.add_hypothesis_embedding(hypothesis)
        
        return hypothesis_id
    
    async def create_review(
        self,
        hypothesis_id: str,
        review_type: str,
        content: str,
        created_by_agent: str,
        **kwargs
    ) -> str:
        """Create a new review"""
        review_id = str(uuid.uuid4())
        review = Review(
            id=review_id,
            hypothesis_id=hypothesis_id,
            review_type=review_type,
            content=content,
            created_by_agent=created_by_agent,
            **kwargs
        )
        
        if self.postgres.connected:
            await self.postgres.create_review(review)
        
        return review_id
    
    async def get_research_session(self, session_id: str) -> Optional[ResearchSession]:
        """Get research session by ID"""
        if self.postgres.connected:
            return await self.postgres.get_research_session(session_id)
        return None
    
    async def get_hypotheses(self, session_id: str, limit: int = 100) -> List[Hypothesis]:
        """Get hypotheses for a research session"""
        if self.postgres.connected:
            return await self.postgres.get_hypotheses(session_id, limit)
        return []
    
    async def get_reviews(self, hypothesis_id: str) -> List[Review]:
        """Get reviews for a hypothesis"""
        if self.postgres.connected:
            return await self.postgres.get_reviews(hypothesis_id)
        return []
    
    async def search_similar_hypotheses(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar hypotheses using vector similarity"""
        if self.chroma.connected:
            return await self.chroma.search_similar_hypotheses(query, limit)
        return []
    
    async def search_relevant_literature(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant literature using vector similarity"""
        if self.chroma.connected:
            return await self.chroma.search_relevant_literature(query, limit)
        return []
    
    async def update_hypothesis_scores(self, hypothesis_id: str, scores: Dict[str, float]):
        """Update hypothesis scores"""
        if self.postgres.connected:
            await self.postgres.update_hypothesis_scores(hypothesis_id, scores)
    
    async def add_literature_to_vector_db(self, papers: List[Dict[str, Any]]):
        """Add literature papers to vector database"""
        if not self.chroma.connected:
            return
        
        for paper in papers:
            await self.chroma.add_literature_embedding(
                paper_id=paper.get('id', str(uuid.uuid4())),
                title=paper.get('title', ''),
                abstract=paper.get('abstract', ''),
                metadata={
                    'authors': paper.get('authors', []),
                    'year': paper.get('year', 0),
                    'source': paper.get('source', 'unknown'),
                    'doi': paper.get('doi', ''),
                    'citation_count': paper.get('citation_count', 0)
                }
            )
    
    async def get_database_health(self) -> Dict[str, Any]:
        """Get health status of both databases"""
        postgres_stats = await self.postgres.get_database_stats()
        chroma_stats = await self.chroma.get_collection_stats()
        
        return {
            "postgres": postgres_stats,
            "chroma": chroma_stats,
            "overall_health": (
                "healthy" if postgres_stats.get("connected") and chroma_stats.get("connected")
                else "degraded" if postgres_stats.get("connected") or chroma_stats.get("connected")
                else "unhealthy"
            )
        }

    def _serialize_hypothesis(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """Convert hypothesis to dictionary for API responses"""
        return {
            "id": hypothesis.id,
            "content": hypothesis.content,
            "summary": hypothesis.summary,
            "generation_method": hypothesis.generation_method,
            "created_at": hypothesis.created_at.isoformat() if hypothesis.created_at else None,
            "created_by_agent": hypothesis.created_by_agent,
            
            # Convert scores to 1-10 scale for display
            "scores": {
                "novelty": round((hypothesis.novelty_score or 0.0) * 10, 1),
                "feasibility": round((hypothesis.feasibility_score or 0.0) * 10, 1),
                "impact": round((hypothesis.impact_score or 0.0) * 10, 1),
                "testability": round((hypothesis.testability_score or 0.0) * 10, 1),
                "composite": round((hypothesis.composite_score or 0.0) * 10, 1)
            },
            
            # Internal scores (0-1 scale) for system use
            "internal_scores": {
                "novelty": hypothesis.novelty_score or 0.0,
                "feasibility": hypothesis.feasibility_score or 0.0,
                "impact": hypothesis.impact_score or 0.0,
                "testability": hypothesis.testability_score or 0.0,
                "composite": hypothesis.composite_score or 0.0
            },
            
            "confidence": hypothesis.confidence,
            "supporting_literature": hypothesis.supporting_literature or [],
            "experimental_protocol": hypothesis.experimental_protocol or {},
            "tournament_wins": hypothesis.tournament_wins,
            "tournament_losses": hypothesis.tournament_losses,
            "elo_rating": hypothesis.elo_rating,
            "generation": hypothesis.generation,
            "parent_id": hypothesis.parent_id,
            "evolution_method": hypothesis.evolution_method
        }

# Global database service instance
database_service = DatabaseService()

# Convenience functions
async def get_research_session(session_id: str) -> Optional[ResearchSession]:
    """Get research session by ID"""
    return await database_service.get_research_session(session_id)

async def create_hypothesis(
    research_session_id: str,
    content: str,
    summary: str,
    generation_method: str,
    created_by_agent: str,
    **kwargs
) -> str:
    """Create a new hypothesis"""
    return await database_service.create_hypothesis(
        research_session_id, content, summary, generation_method, created_by_agent, **kwargs
    )

async def search_similar_hypotheses(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for similar hypotheses"""
    return await database_service.search_similar_hypotheses(query, limit)

async def search_relevant_literature(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for relevant literature"""
    return await database_service.search_relevant_literature(query, limit) 