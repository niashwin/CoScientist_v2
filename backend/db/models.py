from sqlmodel import SQLModel, Field, Column, JSON, Relationship
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum

class ResearchGoal(SQLModel, table=True):
    """Main research goal/session tracking"""
    id: str = Field(primary_key=True)
    goal_text: str
    preferences: Dict[str, Any] = Field(sa_column=Column(JSON))
    status: str = Field(default="active")  # active, completed, failed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    hypotheses: List["Hypothesis"] = Relationship(back_populates="research_goal")
    meta_reviews: List["MetaReview"] = Relationship(back_populates="research_goal")

class Hypothesis(SQLModel, table=True):
    """Core hypothesis storage"""
    id: str = Field(primary_key=True)
    research_goal_id: str = Field(foreign_key="researchgoal.id")
    content: str  # Full hypothesis text
    summary: str  # Short summary
    generation_method: str  # "literature_review", "scientific_debate", "evolution", etc.
    
    # Scores from multi-dimensional evaluation
    novelty_score: Optional[float] = None
    feasibility_score: Optional[float] = None
    impact_score: Optional[float] = None
    testability_score: Optional[float] = None
    composite_score: Optional[float] = None
    confidence: Optional[float] = None
    
    # Lineage tracking
    parent_id: Optional[str] = Field(foreign_key="hypothesis.id", default=None)
    evolution_method: Optional[str] = None  # How it was evolved from parent
    generation: int = Field(default=0)  # Generation number in evolution
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by_agent: str  # Which agent created this
    supporting_literature: List[Dict] = Field(sa_column=Column(JSON), default=[])
    experimental_protocol: Optional[Dict] = Field(sa_column=Column(JSON), default=None)
    
    # Tournament tracking
    tournament_wins: int = Field(default=0)
    tournament_losses: int = Field(default=0)
    elo_rating: float = Field(default=1200.0)
    
    # Relationships
    research_goal: ResearchGoal = Relationship(back_populates="hypotheses")
    reviews: List["Review"] = Relationship(back_populates="hypothesis")
    children: List["Hypothesis"] = Relationship(back_populates="parent")
    parent: Optional["Hypothesis"] = Relationship(
        back_populates="children",
        sa_relationship_kwargs={"remote_side": "Hypothesis.id"}
    )

class Review(SQLModel, table=True):
    """Reviews of hypotheses"""
    id: str = Field(primary_key=True)
    hypothesis_id: str = Field(foreign_key="hypothesis.id")
    review_type: str  # "initial", "full", "deep_verification", "observation", "simulation"
    
    # Review content
    content: str  # Full review text
    critiques: List[Dict] = Field(sa_column=Column(JSON))  # Structured critiques
    suggestions: List[str] = Field(sa_column=Column(JSON))  # Improvement suggestions
    
    # Scores
    correctness_score: Optional[float] = None
    novelty_assessment: Optional[str] = None  # "known", "incremental", "novel", "breakthrough"
    quality_score: Optional[float] = None
    safety_concerns: List[str] = Field(sa_column=Column(JSON), default=[])
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by_agent: str
    literature_cited: List[Dict] = Field(sa_column=Column(JSON), default=[])
    
    # Relationships
    hypothesis: Hypothesis = Relationship(back_populates="reviews")

class TournamentMatch(SQLModel, table=True):
    """Tournament comparison records"""
    id: str = Field(primary_key=True)
    research_goal_id: str = Field(foreign_key="researchgoal.id")
    
    # Participants
    hypothesis_1_id: str = Field(foreign_key="hypothesis.id")
    hypothesis_2_id: str = Field(foreign_key="hypothesis.id")
    winner_id: str = Field(foreign_key="hypothesis.id")
    
    # Comparison details
    comparison_type: str  # "single_turn", "debate"
    debate_transcript: Optional[str] = None
    comparison_rationale: str
    
    # Detailed scoring
    criteria_scores: Dict[str, Dict[str, float]] = Field(sa_column=Column(JSON))
    # Example: {
    #   "hypothesis_1": {"novelty": 0.8, "feasibility": 0.6, ...},
    #   "hypothesis_2": {"novelty": 0.7, "feasibility": 0.9, ...}
    # }
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    round_number: int  # Which tournament round
    
class MetaReview(SQLModel, table=True):
    """System-wide meta reviews and feedback"""
    id: str = Field(primary_key=True)
    research_goal_id: str = Field(foreign_key="researchgoal.id")
    
    # Synthesized feedback
    common_issues: List[Dict] = Field(sa_column=Column(JSON))
    improvement_patterns: Dict[str, Any] = Field(sa_column=Column(JSON))
    success_patterns: Dict[str, Any] = Field(sa_column=Column(JSON))
    
    # Research overview
    research_overview: Optional[str] = None
    key_insights: List[str] = Field(sa_column=Column(JSON))
    future_directions: List[Dict] = Field(sa_column=Column(JSON))
    suggested_contacts: List[Dict] = Field(sa_column=Column(JSON))
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    hypotheses_analyzed: int
    reviews_synthesized: int
    
    # Relationships
    research_goal: ResearchGoal = Relationship(back_populates="meta_reviews")

class AgentActivity(SQLModel, table=True):
    """Track agent activities for optimization"""
    id: str = Field(primary_key=True)
    agent_name: str
    action_type: str
    research_goal_id: str = Field(foreign_key="researchgoal.id")
    
    # Performance metrics
    execution_time_ms: int
    success: bool
    error_message: Optional[str] = None
    
    # Context for prompt optimization
    input_context: Dict[str, Any] = Field(sa_column=Column(JSON))
    output_quality_score: Optional[float] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Database initialization
async def init_db():
    """Initialize database with tables"""
    from sqlmodel import create_engine
    from backend.core.config import settings
    
    engine = create_engine(settings.DATABASE_URL)
    SQLModel.metadata.create_all(engine) 