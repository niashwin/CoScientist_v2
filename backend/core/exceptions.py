"""
Custom exceptions for the AI Co-Scientist system.
Provides structured error handling with recovery strategies.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
import traceback

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"        # Can continue with degraded functionality
    MEDIUM = "medium"  # Should retry or use fallback
    HIGH = "high"      # Cannot continue without intervention
    CRITICAL = "critical"  # System failure

class ErrorCategory(Enum):
    """Error categories for better handling"""
    EXTERNAL_API = "external_api"
    LLM = "llm"
    DATABASE = "database"
    TOOL = "tool"
    VALIDATION = "validation"
    SYSTEM = "system"
    AGENT = "agent"
    LITERATURE = "literature"
    EMBEDDINGS = "embeddings"

class CoScientistError(Exception):
    """Base exception with recovery strategies"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        user_message: Optional[str] = None,
        recovery_suggestions: Optional[List[str]] = None,
        retry_after: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.user_message = user_message or message
        self.recovery_suggestions = recovery_suggestions or []
        self.retry_after = retry_after
        self.context = context or {}
        self.traceback = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "error": {
                "code": self.error_code,
                "message": self.user_message,
                "severity": self.severity.value,
                "category": self.category.value,
                "suggestions": self.recovery_suggestions,
                "retry_after": self.retry_after,
                "context": self.context
            }
        }

# Specific Error Types

class LiteratureSearchError(CoScientistError):
    """Literature search failures"""
    
    def __init__(self, message: str, source: str, query: str):
        super().__init__(
            message=message,
            error_code="LIT_SEARCH_001",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.LITERATURE,
            user_message=f"Literature search failed for query: {query}",
            recovery_suggestions=[
                "Try rephrasing your research goal",
                "The system will attempt alternative sources",
                "Some literature may be temporarily unavailable"
            ],
            context={"source": source, "query": query}
        )

class LLMError(CoScientistError):
    """LLM API failures"""
    
    def __init__(self, message: str, model: str, token_count: Optional[int] = None):
        retry_after = 60 if "rate_limit" in message.lower() else None
        
        super().__init__(
            message=message,
            error_code="LLM_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.LLM,
            user_message="AI model temporarily unavailable",
            recovery_suggestions=[
                "The system will retry automatically",
                "Your session has been saved",
                f"Retry after {retry_after} seconds" if retry_after else None
            ],
            retry_after=retry_after,
            context={"model": model, "token_count": token_count}
        )

class HypothesisGenerationError(CoScientistError):
    """Hypothesis generation failures"""
    
    def __init__(self, message: str, research_goal: str, attempt_number: int):
        super().__init__(
            message=message,
            error_code="HYP_GEN_001",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AGENT,
            user_message="Failed to generate hypothesis",
            recovery_suggestions=[
                "Trying alternative generation methods",
                "Consider refining your research goal",
                "Add more specific constraints or preferences"
            ],
            context={
                "research_goal": research_goal,
                "attempt": attempt_number
            }
        )

class ToolExecutionError(CoScientistError):
    """Scientific tool failures"""
    
    def __init__(self, message: str, tool_name: str, params: Dict):
        super().__init__(
            message=message,
            error_code="TOOL_001",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.TOOL,
            user_message=f"Tool '{tool_name}' temporarily unavailable",
            recovery_suggestions=[
                "Continuing with alternative approaches",
                "Tool results will be approximated",
                "You can retry this specific tool later"
            ],
            context={"tool": tool_name, "parameters": params}
        )

class ValidationError(CoScientistError):
    """Input validation failures"""
    
    def __init__(self, message: str, field: str, value: Any):
        super().__init__(
            message=message,
            error_code="VAL_001",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            user_message=f"Invalid input for {field}",
            recovery_suggestions=[
                f"Check the format of {field}",
                "Refer to the API documentation",
                "Use the example format provided"
            ],
            context={"field": field, "value": str(value)}
        )

class DatabaseError(CoScientistError):
    """Database operation failures"""
    
    def __init__(self, message: str, operation: str, table: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="DB_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATABASE,
            user_message="Database operation failed",
            recovery_suggestions=[
                "The system will retry the operation",
                "Data may be temporarily stored in memory",
                "Check your database connection"
            ],
            context={"operation": operation, "table": table}
        )

class EmbeddingError(CoScientistError):
    """Embedding generation failures"""
    
    def __init__(self, message: str, model: str, text_length: int):
        super().__init__(
            message=message,
            error_code="EMB_001",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.EMBEDDINGS,
            user_message="Failed to generate embeddings",
            recovery_suggestions=[
                "The system will try alternative embedding models",
                "Vector search may be temporarily unavailable",
                "Continuing with text-based search"
            ],
            context={"model": model, "text_length": text_length}
        )

class AgentError(CoScientistError):
    """Agent execution failures"""
    
    def __init__(self, message: str, agent_name: str, context: Dict[str, Any]):
        super().__init__(
            message=message,
            error_code="AGENT_001",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AGENT,
            user_message=f"Agent '{agent_name}' encountered an error",
            recovery_suggestions=[
                "The system will try alternative approaches",
                "Agent will be restarted automatically",
                "Consider adjusting your research parameters"
            ],
            context={"agent": agent_name, "execution_context": context}
        ) 