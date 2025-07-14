from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys (mandatory in production, optional for testing)
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    SERPER_API_KEY: str = ""
    SEMANTIC_SCHOLAR_API_KEY: str = ""
    PERPLEXITY_API_KEY: str = ""
    
    # Optional toggles
    ENABLE_PERSISTENCE: bool = False
    AUTH_ENABLED: bool = False
    OTEL_ENABLED: bool = False
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # Database URLs
    POSTGRES_URL: str = "postgresql://cs:cs@postgres:5432/coscientist"
    CHROMA_PATH: str = "/chroma"
    
    # Enhanced configuration
    REDIS_URL: str = "redis://redis:6379/0"
    MAX_WORKERS_PER_AGENT: int = 4
    HYPOTHESIS_CACHE_TTL: int = 3600
    TOURNAMENT_MIN_COMPARISONS: int = 10
    EVOLUTION_MUTATION_RATE: float = 0.2
    LITERATURE_SEARCH_DEPTH: int = 3
    RESULT_CONFIDENCE_THRESHOLD: float = 0.7
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_BACKOFF_SECONDS: int = 2
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT: int = 60
    
    # Celery configuration
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_TIMEZONE: str = "UTC"
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = 1
    
    # Cache configuration
    LITERATURE_CACHE_TTL: int = 7200
    EMBEDDING_CACHE_TTL: int = 86400
    
    # External API configuration
    EXTERNAL_API_TIMEOUT: int = 30
    REDIS_TIMEOUT: int = 5
    
    # External API URLs
    SERPER_API_URL: str = "https://google.serper.dev/scholar"
    SEMANTIC_SCHOLAR_API_URL: str = "https://api.semanticscholar.org/graph/v1"
    PERPLEXITY_API_URL: str = "https://api.perplexity.ai"
    CHEMBL_API_URL: str = "https://www.ebi.ac.uk/chembl/api/data"
    PUBCHEM_API_URL: str = "https://pubchem.ncbi.nlm.nih.gov/rest"
    ALPHAFOLD_API_URL: str = "https://alphafold.ebi.ac.uk/api"
    UNIPROT_API_URL: str = "https://rest.uniprot.org"
    
    # Development settings
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    WORKER_PROCESSES: int = 8
    
    # External tools (optional)
    EXTERNAL_TOOLS: bool = False
    ALPHAFOLD_API_KEY: Optional[str] = None
    CHEMBL_API_KEY: Optional[str] = None
    PUBCHEM_API_KEY: Optional[str] = None
    UNIPROT_API_KEY: Optional[str] = None
    
    # Computed properties
    @property
    def DATABASE_URL(self) -> str:
        """Get database URL based on persistence setting"""
        if self.ENABLE_PERSISTENCE:
            return self.POSTGRES_URL
        else:
            # Use in-memory SQLite for development
            return "sqlite:///./coscientist.db"
    
    @property
    def CHROMA_ENABLED(self) -> bool:
        """Check if Chroma vector database is enabled"""
        return self.ENABLE_PERSISTENCE
    
    class Config:
        # Look for .env file in project root (parent of backend directory)
        env_file = Path(__file__).parent.parent.parent / ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env file

# Global settings instance
settings = Settings() 