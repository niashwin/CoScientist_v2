"""
Literature Search Logger Service

This service tracks all literature search activity during research sessions
to help debug frontend display issues and monitor backend behavior.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)

@dataclass
class LiteratureLogEntry:
    """Single literature search log entry"""
    timestamp: str
    session_id: str
    event_type: str  # 'search_start', 'search_complete', 'websocket_send', 'error'
    query: str
    stage: Optional[str] = None  # 'searching', 'analyzing', 'complete'
    paper_count: int = 0
    message: str = ""
    papers: List[Dict[str, Any]] = None
    error: Optional[str] = None
    source: str = "unknown"  # 'external_api', 'fallback', 'cache'
    
    def __post_init__(self):
        if self.papers is None:
            self.papers = []

class LiteratureLogger:
    """Logger for all literature search activity"""
    
    def __init__(self, log_file: str = "literature_search_log.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory log for current session (for debugging)
        self.current_session_logs: Dict[str, List[LiteratureLogEntry]] = {}
        
        # Setup file logger
        self.file_logger = logging.getLogger("literature_search_file")
        self.file_logger.setLevel(logging.INFO)
        
        # Create file handler if it doesn't exist
        if not self.file_logger.handlers:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            self.file_logger.addHandler(file_handler)
    
    def log_search_start(self, session_id: str, query: str, source: str = "unknown"):
        """Log the start of a literature search"""
        entry = LiteratureLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            event_type="search_start",
            query=query,
            source=source,
            message=f"Starting literature search for query: {query}"
        )
        
        self._write_log_entry(entry)
        logger.info(f"[LITERATURE] Search started - Session: {session_id}, Query: {query}, Source: {source}")
    
    def log_search_complete(self, session_id: str, query: str, papers: List[Dict[str, Any]], source: str = "unknown"):
        """Log the completion of a literature search"""
        entry = LiteratureLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            event_type="search_complete",
            query=query,
            paper_count=len(papers),
            papers=papers,
            source=source,
            message=f"Literature search completed - Found {len(papers)} papers"
        )
        
        self._write_log_entry(entry)
        logger.info(f"[LITERATURE] Search completed - Session: {session_id}, Query: {query}, Papers: {len(papers)}, Source: {source}")
    
    def log_websocket_send(self, session_id: str, query: str, stage: str, papers: List[Dict[str, Any]], message: str):
        """Log when literature search update is sent via WebSocket"""
        entry = LiteratureLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            event_type="websocket_send",
            query=query,
            stage=stage,
            paper_count=len(papers),
            papers=papers,
            message=message
        )
        
        self._write_log_entry(entry)
        logger.info(f"[LITERATURE] WebSocket sent - Session: {session_id}, Stage: {stage}, Papers: {len(papers)}, Message: {message}")
    
    def log_error(self, session_id: str, query: str, error: str, source: str = "unknown"):
        """Log a literature search error"""
        entry = LiteratureLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            event_type="error",
            query=query,
            error=error,
            source=source,
            message=f"Literature search error: {error}"
        )
        
        self._write_log_entry(entry)
        logger.error(f"[LITERATURE] Error - Session: {session_id}, Query: {query}, Error: {error}, Source: {source}")
    
    def log_cache_hit(self, session_id: str, query: str, paper_count: int):
        """Log when literature search results come from cache"""
        entry = LiteratureLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            event_type="cache_hit",
            query=query,
            paper_count=paper_count,
            source="cache",
            message=f"Cache hit - Found {paper_count} cached papers"
        )
        
        self._write_log_entry(entry)
        logger.info(f"[LITERATURE] Cache hit - Session: {session_id}, Query: {query}, Papers: {paper_count}")
    
    def log_fallback_used(self, session_id: str, query: str, paper_count: int):
        """Log when fallback literature search is used"""
        entry = LiteratureLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            event_type="fallback_used",
            query=query,
            paper_count=paper_count,
            source="fallback",
            message=f"Using fallback literature search - Generated {paper_count} papers"
        )
        
        self._write_log_entry(entry)
        logger.info(f"[LITERATURE] Fallback used - Session: {session_id}, Query: {query}, Papers: {paper_count}")
    
    def _write_log_entry(self, entry: LiteratureLogEntry):
        """Write log entry to file and memory"""
        # Add to in-memory log
        if entry.session_id not in self.current_session_logs:
            self.current_session_logs[entry.session_id] = []
        self.current_session_logs[entry.session_id].append(entry)
        
        # Write to file as JSON line
        try:
            log_line = json.dumps(asdict(entry))
            self.file_logger.info(log_line)
        except Exception as e:
            logger.error(f"Failed to write literature log entry: {e}")
    
    def get_session_logs(self, session_id: str) -> List[LiteratureLogEntry]:
        """Get all logs for a specific session"""
        return self.current_session_logs.get(session_id, [])
    
    def get_recent_logs(self, limit: int = 50) -> List[LiteratureLogEntry]:
        """Get recent logs across all sessions"""
        all_logs = []
        for session_logs in self.current_session_logs.values():
            all_logs.extend(session_logs)
        
        # Sort by timestamp and return most recent
        all_logs.sort(key=lambda x: x.timestamp, reverse=True)
        return all_logs[:limit]
    
    def clear_session_logs(self, session_id: str):
        """Clear logs for a specific session"""
        if session_id in self.current_session_logs:
            del self.current_session_logs[session_id]
            logger.info(f"[LITERATURE] Cleared logs for session: {session_id}")
    
    def get_log_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of literature search activity for a session"""
        logs = self.get_session_logs(session_id)
        
        if not logs:
            return {
                "session_id": session_id,
                "total_events": 0,
                "searches": 0,
                "websocket_sends": 0,
                "errors": 0,
                "cache_hits": 0,
                "fallback_uses": 0
            }
        
        summary = {
            "session_id": session_id,
            "total_events": len(logs),
            "searches": len([l for l in logs if l.event_type == "search_start"]),
            "websocket_sends": len([l for l in logs if l.event_type == "websocket_send"]),
            "errors": len([l for l in logs if l.event_type == "error"]),
            "cache_hits": len([l for l in logs if l.event_type == "cache_hit"]),
            "fallback_uses": len([l for l in logs if l.event_type == "fallback_used"]),
            "first_event": logs[0].timestamp if logs else None,
            "last_event": logs[-1].timestamp if logs else None,
            "queries": list(set([l.query for l in logs if l.query])),
            "stages": list(set([l.stage for l in logs if l.stage])),
            "sources": list(set([l.source for l in logs if l.source]))
        }
        
        return summary

# Global literature logger instance
literature_logger = LiteratureLogger()

# Convenience functions
def log_search_start(session_id: str, query: str, source: str = "unknown"):
    """Log the start of a literature search"""
    literature_logger.log_search_start(session_id, query, source)

def log_search_complete(session_id: str, query: str, papers: List[Dict[str, Any]], source: str = "unknown"):
    """Log the completion of a literature search"""
    literature_logger.log_search_complete(session_id, query, papers, source)

def log_websocket_send(session_id: str, query: str, stage: str, papers: List[Dict[str, Any]], message: str):
    """Log when literature search update is sent via WebSocket"""
    literature_logger.log_websocket_send(session_id, query, stage, papers, message)

def log_error(session_id: str, query: str, error: str, source: str = "unknown"):
    """Log a literature search error"""
    literature_logger.log_error(session_id, query, error, source)

def log_cache_hit(session_id: str, query: str, paper_count: int):
    """Log when literature search results come from cache"""
    literature_logger.log_cache_hit(session_id, query, paper_count)

def log_fallback_used(session_id: str, query: str, paper_count: int):
    """Log when fallback literature search is used"""
    literature_logger.log_fallback_used(session_id, query, paper_count)

def get_session_logs(session_id: str) -> List[LiteratureLogEntry]:
    """Get all logs for a specific session"""
    return literature_logger.get_session_logs(session_id)

def get_log_summary(session_id: str) -> Dict[str, Any]:
    """Get a summary of literature search activity for a session"""
    return literature_logger.get_log_summary(session_id)

def clear_session_logs(session_id: str):
    """Clear logs for a specific session"""
    literature_logger.clear_session_logs(session_id) 