"""
Literature search service with multiple external API integrations.
Implements circuit breakers, fallback strategies, and intelligent search routing.
"""

import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from urllib.parse import quote_plus

from backend.core.config import settings
from backend.core.exceptions import LiteratureSearchError, CoScientistError, ErrorSeverity
from backend.core.circuit_breaker import CircuitBreaker
from backend.services.cache import CacheService

logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    BROAD_SURVEY = "broad_survey"
    DEEP_DIVE = "deep_dive"
    CITATION_CHASE = "citation_chase"
    GAP_ANALYSIS = "gap_analysis"
    RECENT_ADVANCES = "recent_advances"

@dataclass
class Paper:
    """Standardized paper representation across all sources"""
    title: str
    authors: List[str]
    abstract: str
    year: int
    doi: Optional[str] = None
    url: Optional[str] = None
    citation_count: int = 0
    venue: Optional[str] = None
    keywords: List[str] = None
    relevance_score: float = 0.0
    source: str = "unknown"
    paper_id: Optional[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BaseSearchProvider:
    """Base class for literature search providers"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.CIRCUIT_BREAKER_THRESHOLD,
            timeout=settings.CIRCUIT_BREAKER_TIMEOUT
        )
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.EXTERNAL_API_TIMEOUT)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, limit: int = 10) -> List[Paper]:
        """Search for papers - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get detailed information about a specific paper"""
        raise NotImplementedError

class SerperSearchProvider(BaseSearchProvider):
    """Google Scholar search via Serper API"""
    
    def __init__(self):
        super().__init__(
            api_key=settings.SERPER_API_KEY,
            base_url=settings.SERPER_API_URL
        )
    
    async def search(self, query: str, limit: int = 10) -> List[Paper]:
        """Search Google Scholar via Serper"""
        try:
            async with self.circuit_breaker:
                headers = {
                    'X-API-KEY': self.api_key,
                    'Content-Type': 'application/json'
                }
                
                # Add filetype:pdf to focus on PDF academic papers
                enhanced_query = f"{query} filetype:pdf"
                
                payload = {
                    'q': enhanced_query,
                    'num': min(limit, 100)  # Serper allows up to 100 results
                }
                
                # Use data with JSON string format (as required by Serper Scholar API)
                async with self.session.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(payload)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return self._parse_serper_response(data)
        
        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            raise LiteratureSearchError(
                f"Serper search failed: {str(e)}",
                source="serper",
                query=query
            )
    
    def _parse_serper_response(self, data: Dict) -> List[Paper]:
        """Parse Serper API response into Paper objects"""
        papers = []
        
        organic_results = data.get('organic', [])
        total_results = len(organic_results)
        
        for index, item in enumerate(organic_results):
            # Calculate relevance score based on position (higher position = lower relevance)
            # First result gets 1.0, second gets 0.95, etc.
            position_score = max(0.1, 1.0 - (index * 0.05))
            
            # Also consider if the item has additional relevance indicators
            relevance_score = position_score
            
            # Boost relevance if it has citation count or other quality indicators
            if item.get('citation_count', 0) > 0:
                relevance_score = min(1.0, relevance_score + 0.1)
            
            # Boost relevance if the title/snippet contains key academic terms
            title_snippet = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
            academic_terms = ['research', 'study', 'analysis', 'method', 'approach', 'technique', 'novel', 'experimental']
            academic_score = sum(1 for term in academic_terms if term in title_snippet) * 0.02
            relevance_score = min(1.0, relevance_score + academic_score)
            
            paper = Paper(
                title=item.get('title', ''),
                authors=self._extract_authors(item.get('snippet', '')),
                abstract=item.get('snippet', ''),
                year=self._extract_year(item.get('snippet', '')),
                doi=self._extract_doi(item.get('link', '')),
                url=item.get('link', ''),
                citation_count=item.get('citation_count', 0),
                venue=item.get('source', ''),
                relevance_score=relevance_score,
                source="serper"
            )
            papers.append(paper)
        
        return papers
    
    def _extract_authors(self, snippet: str) -> List[str]:
        """Extract author names from snippet"""
        # Simple heuristic - look for patterns like "by Author Name"
        import re
        author_pattern = r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        matches = re.findall(author_pattern, snippet)
        return matches[:3]  # Limit to first 3 authors
    
    def _extract_year(self, snippet: str) -> int:
        """Extract publication year from snippet"""
        import re
        year_pattern = r'(\d{4})'
        matches = re.findall(year_pattern, snippet)
        for match in matches:
            year = int(match)
            if 1900 <= year <= datetime.now().year:
                return year
        return datetime.now().year
    
    def _extract_doi(self, url: str) -> Optional[str]:
        """Extract DOI from URL if present"""
        import re
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        match = re.search(doi_pattern, url)
        return match.group(0) if match else None

class SemanticScholarProvider(BaseSearchProvider):
    """Semantic Scholar API integration"""
    
    def __init__(self):
        super().__init__(
            api_key=settings.SEMANTIC_SCHOLAR_API_KEY,
            base_url=settings.SEMANTIC_SCHOLAR_API_URL
        )
    
    async def search(self, query: str, limit: int = 10) -> List[Paper]:
        """Search Semantic Scholar"""
        try:
            async with self.circuit_breaker:
                headers = {}
                if self.api_key:
                    headers['x-api-key'] = self.api_key
                
                params = {
                    'query': query,
                    'limit': limit,
                    'fields': 'title,authors,abstract,year,citationCount,venue,url,externalIds'
                }
                
                async with self.session.get(
                    f"{self.base_url}/paper/search",
                    headers=headers,
                    params=params
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return self._parse_semantic_scholar_response(data)
        
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            raise LiteratureSearchError(
                f"Semantic Scholar search failed: {str(e)}",
                source="semantic_scholar",
                query=query
            )
    
    def _parse_semantic_scholar_response(self, data: Dict) -> List[Paper]:
        """Parse Semantic Scholar response"""
        papers = []
        
        for item in data.get('data', []):
            authors = [author.get('name', '') for author in item.get('authors', [])]
            
            paper = Paper(
                title=item.get('title', ''),
                authors=authors,
                abstract=item.get('abstract', ''),
                year=item.get('year', datetime.now().year),
                doi=item.get('externalIds', {}).get('DOI'),
                url=item.get('url', ''),
                citation_count=item.get('citationCount', 0),
                venue=item.get('venue', ''),
                paper_id=item.get('paperId'),
                source="semantic_scholar"
            )
            papers.append(paper)
        
        return papers
    
    async def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get detailed paper information"""
        try:
            headers = {}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            params = {
                'fields': 'title,authors,abstract,year,citationCount,venue,url,externalIds,references,citations'
            }
            
            async with self.session.get(
                f"{self.base_url}/paper/{paper_id}",
                headers=headers,
                params=params
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                return self._parse_paper_details(data)
        
        except Exception as e:
            logger.error(f"Failed to get paper details for {paper_id}: {e}")
            return None
    
    def _parse_paper_details(self, data: Dict) -> Paper:
        """Parse detailed paper information"""
        authors = [author.get('name', '') for author in data.get('authors', [])]
        
        return Paper(
            title=data.get('title', ''),
            authors=authors,
            abstract=data.get('abstract', ''),
            year=data.get('year', datetime.now().year),
            doi=data.get('externalIds', {}).get('DOI'),
            url=data.get('url', ''),
            citation_count=data.get('citationCount', 0),
            venue=data.get('venue', ''),
            paper_id=data.get('paperId'),
            source="semantic_scholar"
        )

class PerplexitySearchProvider(BaseSearchProvider):
    """Perplexity API for research queries"""
    
    def __init__(self):
        super().__init__(
            api_key=settings.PERPLEXITY_API_KEY,
            base_url=settings.PERPLEXITY_API_URL
        )
    
    async def search(self, query: str, limit: int = 10) -> List[Paper]:
        """Search using Perplexity API"""
        try:
            async with self.circuit_breaker:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'model': 'llama-3.1-sonar-small-128k-online',
                    'messages': [
                        {
                            'role': 'user',
                            'content': f'Find recent scientific papers about: {query}. Provide title, authors, year, and abstract for each paper.'
                        }
                    ],
                    'max_tokens': 2000,
                    'temperature': 0.1
                }
                
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return self._parse_perplexity_response(data, query)
        
        except Exception as e:
            logger.error(f"Perplexity search failed: {e}")
            raise LiteratureSearchError(
                f"Perplexity search failed: {str(e)}",
                source="perplexity",
                query=query
            )
    
    def _parse_perplexity_response(self, data: Dict, query: str) -> List[Paper]:
        """Parse Perplexity response to extract paper information"""
        papers = []
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Simple parsing - in production, this would be more sophisticated
        import re
        
        # Look for paper-like structures in the response
        paper_pattern = r'Title:\s*(.+?)\nAuthors:\s*(.+?)\nYear:\s*(\d{4})\nAbstract:\s*(.+?)(?=\n\n|\nTitle:|\Z)'
        matches = re.findall(paper_pattern, content, re.DOTALL)
        
        for i, match in enumerate(matches):
            title, authors_str, year, abstract = match
            authors = [name.strip() for name in authors_str.split(',')]
            
            paper = Paper(
                title=title.strip(),
                authors=authors,
                abstract=abstract.strip(),
                year=int(year),
                relevance_score=1.0 - (i * 0.1),  # Decreasing relevance
                source="perplexity"
            )
            papers.append(paper)
        
        return papers

class SmartLiteratureSearch:
    """
    Intelligent literature search that orchestrates multiple providers
    with fallback strategies and result synthesis.
    """
    
    def __init__(self):
        self.providers = {
            'serper': SerperSearchProvider(),
            'semantic_scholar': SemanticScholarProvider(),
            'perplexity': PerplexitySearchProvider()
        }
        self.cache = CacheService()
        
        # Strategy configurations
        self.strategy_configs = {
            SearchStrategy.BROAD_SURVEY: {
                'providers': ['serper', 'semantic_scholar'],
                'limit_per_provider': 25,  # Increased from 15
                'merge_strategy': 'diversified'
            },
            SearchStrategy.DEEP_DIVE: {
                'providers': ['semantic_scholar', 'perplexity'],
                'limit_per_provider': 20,  # Increased from 10
                'merge_strategy': 'quality_focused'
            },
            SearchStrategy.CITATION_CHASE: {
                'providers': ['semantic_scholar'],
                'limit_per_provider': 30,  # Increased from 20
                'merge_strategy': 'citation_network'
            },
            SearchStrategy.GAP_ANALYSIS: {
                'providers': ['perplexity', 'semantic_scholar'],
                'limit_per_provider': 20,  # Increased from 12
                'merge_strategy': 'gap_focused'
            },
            SearchStrategy.RECENT_ADVANCES: {
                'providers': ['serper', 'perplexity'],
                'limit_per_provider': 20,  # Increased from 10
                'merge_strategy': 'recency_focused'
            }
        }
    
    async def search(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.BROAD_SURVEY,
        limit: int = 50  # Increased from 20 to 50
    ) -> List[Paper]:
        """
        Execute intelligent literature search with specified strategy
        """
        from backend.services.literature_logger import log_cache_hit, log_search_start, log_search_complete
        
        # Log search start
        log_search_start("unknown", query, f"smart_literature_search_{strategy.value}")
        
        # Check cache first
        cache_key = self._generate_cache_key(query, strategy, limit)
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: {query}")
            papers = [Paper(**paper_data) for paper_data in cached_result]
            
            # Log cache hit
            log_cache_hit("unknown", query, len(papers))
            
            return papers
        
        logger.info(f"Starting literature search: {query} with strategy: {strategy.value}")
        
        config = self.strategy_configs[strategy]
        results = []
        
        # Execute searches in parallel across providers
        search_tasks = []
        for provider_name in config['providers']:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                task = self._safe_search(
                    provider,
                    query,
                    config['limit_per_provider']
                )
                search_tasks.append((provider_name, task))
        
        # Wait for all searches to complete
        provider_results = {}
        for provider_name, task in search_tasks:
            try:
                papers = await task
                provider_results[provider_name] = papers
                logger.info(f"{provider_name} returned {len(papers)} papers")
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                provider_results[provider_name] = []
        
        # Merge and deduplicate results
        merged_papers = await self._merge_results(
            provider_results,
            config['merge_strategy'],
            limit
        )
        
        # Cache results
        await self.cache.set(
            cache_key,
            [paper.to_dict() for paper in merged_papers],
            ttl=settings.LITERATURE_CACHE_TTL
        )
        
        logger.info(f"Literature search completed: {len(merged_papers)} papers")
        
        # Log search completion
        log_search_complete("unknown", query, [paper.to_dict() for paper in merged_papers], f"smart_literature_search_{strategy.value}")
        
        return merged_papers
    
    async def _safe_search(self, provider: BaseSearchProvider, query: str, limit: int) -> List[Paper]:
        """Execute search with provider in a safe context"""
        try:
            async with provider:
                return await provider.search(query, limit)
        except Exception as e:
            logger.error(f"Provider search failed: {e}")
            return []
    
    async def _merge_results(
        self,
        provider_results: Dict[str, List[Paper]],
        merge_strategy: str,
        limit: int
    ) -> List[Paper]:
        """Merge results from multiple providers using specified strategy"""
        
        all_papers = []
        for provider_name, papers in provider_results.items():
            all_papers.extend(papers)
        
        # Deduplicate based on title similarity
        deduplicated = self._deduplicate_papers(all_papers)
        
        # Apply merge strategy
        if merge_strategy == 'diversified':
            # Ensure diversity across sources
            sorted_papers = self._diversified_sort(deduplicated)
        elif merge_strategy == 'quality_focused':
            # Prioritize citation count and venue quality
            sorted_papers = self._quality_sort(deduplicated)
        elif merge_strategy == 'citation_network':
            # Focus on highly cited papers
            sorted_papers = sorted(deduplicated, key=lambda p: p.citation_count, reverse=True)
        elif merge_strategy == 'recency_focused':
            # Prioritize recent papers
            sorted_papers = sorted(deduplicated, key=lambda p: p.year, reverse=True)
        elif merge_strategy == 'gap_focused':
            # Focus on papers that might reveal research gaps
            sorted_papers = self._gap_focused_sort(deduplicated)
        else:
            # Default: sort by relevance score
            sorted_papers = sorted(deduplicated, key=lambda p: p.relevance_score, reverse=True)
        
        return sorted_papers[:limit]
    
    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers based on title similarity"""
        from difflib import SequenceMatcher
        
        unique_papers = []
        seen_titles = []
        
        for paper in papers:
            is_duplicate = False
            for seen_title in seen_titles:
                similarity = SequenceMatcher(None, paper.title.lower(), seen_title.lower()).ratio()
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.append(paper.title)
        
        return unique_papers
    
    def _diversified_sort(self, papers: List[Paper]) -> List[Paper]:
        """Sort papers to ensure diversity across sources"""
        source_counts = {}
        sorted_papers = []
        
        # Sort by relevance first
        papers_by_relevance = sorted(papers, key=lambda p: p.relevance_score, reverse=True)
        
        for paper in papers_by_relevance:
            source_count = source_counts.get(paper.source, 0)
            # Prefer papers from sources we haven't seen much
            paper.relevance_score *= (1.0 - source_count * 0.1)
            source_counts[paper.source] = source_count + 1
            sorted_papers.append(paper)
        
        return sorted(sorted_papers, key=lambda p: p.relevance_score, reverse=True)
    
    def _quality_sort(self, papers: List[Paper]) -> List[Paper]:
        """Sort papers by quality indicators"""
        def quality_score(paper: Paper) -> float:
            score = 0.0
            
            # Citation count (normalized)
            if paper.citation_count > 0:
                score += min(paper.citation_count / 100.0, 1.0) * 0.4
            
            # Venue quality (simple heuristic)
            high_quality_venues = ['Nature', 'Science', 'Cell', 'PNAS', 'Journal of']
            if any(venue in paper.venue for venue in high_quality_venues):
                score += 0.3
            
            # Recency
            current_year = datetime.now().year
            if paper.year >= current_year - 2:
                score += 0.2
            elif paper.year >= current_year - 5:
                score += 0.1
            
            # Original relevance
            score += paper.relevance_score * 0.1
            
            return score
        
        return sorted(papers, key=quality_score, reverse=True)
    
    def _gap_focused_sort(self, papers: List[Paper]) -> List[Paper]:
        """Sort papers to identify potential research gaps"""
        # Simple heuristic: papers with moderate citation counts might indicate gaps
        def gap_score(paper: Paper) -> float:
            score = 0.0
            
            # Sweet spot for citation counts (not too high, not too low)
            if 5 <= paper.citation_count <= 50:
                score += 0.4
            
            # Recent papers with few citations might indicate emerging areas
            current_year = datetime.now().year
            if paper.year >= current_year - 3 and paper.citation_count < 20:
                score += 0.3
            
            # Papers with specific gap-indicating keywords
            gap_keywords = ['limitation', 'future work', 'further research', 'gap', 'unexplored']
            if any(keyword in paper.abstract.lower() for keyword in gap_keywords):
                score += 0.3
            
            return score
        
        return sorted(papers, key=gap_score, reverse=True)
    
    def _generate_cache_key(self, query: str, strategy: SearchStrategy, limit: int) -> str:
        """Generate cache key for search parameters"""
        key_string = f"{query}:{strategy.value}:{limit}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_paper_citations(self, paper_id: str, source: str = "semantic_scholar") -> List[Paper]:
        """Get papers that cite the given paper"""
        if source == "semantic_scholar" and "semantic_scholar" in self.providers:
            provider = self.providers["semantic_scholar"]
            async with provider:
                # This would require additional API calls to get citations
                # Implementation depends on the specific API capabilities
                pass
        
        return []
    
    async def search_similar_papers(self, paper: Paper, limit: int = 10) -> List[Paper]:
        """Find papers similar to the given paper"""
        # Use the paper's keywords and abstract for similarity search
        search_query = f"{' '.join(paper.keywords)} {paper.abstract[:200]}"
        return await self.search(search_query, SearchStrategy.DEEP_DIVE, limit)

# Factory function for easy instantiation
def create_literature_search() -> SmartLiteratureSearch:
    """Create a configured literature search instance"""
    return SmartLiteratureSearch() 