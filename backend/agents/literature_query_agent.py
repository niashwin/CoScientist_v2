"""
Literature Query Evolution Agent for AI Co-Scientist system
Generates optimized academic search queries using LLM reasoning
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from backend.services.llm import LLMService

logger = logging.getLogger(__name__)

class LiteratureQueryEvolutionAgent:
    """
    Specialized agent for evolving literature search queries based on research context
    """
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        
        # Define subtasks for different iterations
        self.iteration_subtasks = {
            0: "Find recent advances and fundamental mechanisms in the field",
            1: "Identify alternative approaches and methodological variations", 
            2: "Discover gaps, limitations, and unresolved challenges",
            3: "Explore interdisciplinary applications and emerging trends",
            4: "Find experimental protocols and validation methods",
            5: "Identify key researchers and collaborative opportunities"
        }
        
        self.base_prompt = """You are a research assistant specializing in academic literature discovery. Your task is to generate effective Google Scholar search queries based on a research goal and specific subtask.

RESEARCH GOAL: {goal}
CONCEPTS: {concepts}
CURRENT SUBTASK: {subtask}

Generate 1 optimized Google Scholar search query that will find the most relevant academic papers for this subtask. For the query:

1. Use specific academic terminology and keywords from the field
2. Include relevant time constraints if the goal specifies them
3. Consider synonyms and alternative phrasings of key concepts
4. Balance specificity with breadth to avoid missing important papers
5. Structure queries to target different aspects of the subtask

Format your output as one search query per line, nothing else. Do not include explanations or additional text."""

    async def generate_evolved_query(
        self,
        research_goal: str,
        iteration: int,
        existing_hypotheses: List[Dict] = None,
        additional_concepts: List[str] = None
    ) -> str:
        """
        Generate an evolved literature search query based on research context
        
        Args:
            research_goal: The original research question/goal
            iteration: Current research iteration (0, 1, 2, ...)
            existing_hypotheses: List of existing hypotheses to extract concepts from
            additional_concepts: Additional concepts to include
            
        Returns:
            Optimized search query string
        """
        try:
            # Extract concepts from existing hypotheses
            concepts = self._extract_concepts_from_hypotheses(existing_hypotheses or [])
            
            # Add any additional concepts
            if additional_concepts:
                concepts.extend(additional_concepts)
            
            # Remove duplicates and limit to most relevant concepts
            unique_concepts = list(set(concepts))[:5]  # Limit to top 5 concepts
            
            # Get subtask for current iteration
            subtask = self.iteration_subtasks.get(
                iteration, 
                "Find comprehensive literature review and synthesis opportunities"
            )
            
            # Format concepts for the prompt
            concepts_text = ", ".join(unique_concepts) if unique_concepts else "No specific concepts identified yet"
            
            # Generate the prompt
            prompt = self.base_prompt.format(
                goal=research_goal,
                concepts=concepts_text,
                subtask=subtask
            )
            
            logger.info(f"Generating literature query for iteration {iteration}")
            logger.info(f"Subtask: {subtask}")
            logger.info(f"Concepts: {concepts_text}")
            
            # Get LLM response
            response = await self.llm_service.generate_response(prompt)
            
            # Clean up the response (remove any extra text, get first line)
            query = self._clean_query_response(response)
            
            # Add filetype:pdf filter for academic papers
            if "filetype:pdf" not in query.lower():
                query += " filetype:pdf"
            
            logger.info(f"Generated query: {query}")
            
            return query
            
        except Exception as e:
            logger.error(f"Error generating evolved query: {e}")
            # Fallback to simple query
            fallback_query = f"{research_goal} recent advances mechanisms filetype:pdf"
            logger.info(f"Using fallback query: {fallback_query}")
            return fallback_query
    
    def _extract_concepts_from_hypotheses(self, hypotheses: List[Dict]) -> List[str]:
        """
        Extract key scientific concepts from existing hypotheses
        
        Args:
            hypotheses: List of hypothesis dictionaries
            
        Returns:
            List of extracted concepts
        """
        concepts = []
        
        # Common scientific terms that indicate important concepts
        concept_indicators = [
            "mechanism", "pathway", "protein", "gene", "enzyme", "receptor",
            "signaling", "regulation", "expression", "activation", "inhibition",
            "binding", "interaction", "complex", "domain", "motif", "structure",
            "function", "activity", "process", "response", "treatment", "therapy",
            "drug", "compound", "molecule", "cell", "tissue", "organ", "system",
            "disease", "disorder", "syndrome", "condition", "phenotype", "genotype",
            "mutation", "variant", "polymorphism", "biomarker", "target", "approach"
        ]
        
        for hypothesis in hypotheses:
            content = hypothesis.get('content', '').lower()
            
            # Extract words that appear near concept indicators
            words = content.split()
            for i, word in enumerate(words):
                # Clean word of punctuation
                clean_word = ''.join(c for c in word if c.isalnum())
                
                # If word is a concept indicator, look for nearby important terms
                if clean_word in concept_indicators:
                    # Look at surrounding words
                    start_idx = max(0, i - 2)
                    end_idx = min(len(words), i + 3)
                    context_words = words[start_idx:end_idx]
                    
                    # Extract capitalized words or technical terms
                    for context_word in context_words:
                        clean_context = ''.join(c for c in context_word if c.isalnum())
                        if len(clean_context) > 3 and (
                            context_word[0].isupper() or  # Capitalized
                            clean_context in concept_indicators or  # Technical term
                            len(clean_context) > 8  # Long scientific terms
                        ):
                            concepts.append(clean_context.lower())
        
        # Also extract direct concept indicators that appear in hypotheses
        for hypothesis in hypotheses:
            content = hypothesis.get('content', '').lower()
            for indicator in concept_indicators:
                if indicator in content:
                    concepts.append(indicator)
        
        # Remove duplicates and return most frequent concepts
        concept_counts = {}
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Sort by frequency and return top concepts
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, count in sorted_concepts[:10]]
    
    def _clean_query_response(self, response: str) -> str:
        """
        Clean the LLM response to extract just the search query
        
        Args:
            response: Raw LLM response
            
        Returns:
            Clean search query string
        """
        # Split by lines and get the first non-empty line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        if not lines:
            return "recent advances mechanisms"
        
        # Take the first line as the query
        query = lines[0]
        
        # Remove any numbering or formatting
        query = query.lstrip('1234567890.- ')
        
        # Remove quotes if present
        query = query.strip('"\'')
        
        # Ensure reasonable length (not too long for search engines)
        if len(query) > 200:
            query = query[:200].rsplit(' ', 1)[0]  # Cut at word boundary
        
        return query
    
    def get_subtask_for_iteration(self, iteration: int) -> str:
        """
        Get the subtask description for a given iteration
        
        Args:
            iteration: Research iteration number
            
        Returns:
            Subtask description string
        """
        return self.iteration_subtasks.get(
            iteration,
            "Find comprehensive literature review and synthesis opportunities"
        )
    
    async def generate_multiple_queries(
        self,
        research_goal: str,
        iteration: int,
        existing_hypotheses: List[Dict] = None,
        num_queries: int = 3
    ) -> List[str]:
        """
        Generate multiple diverse queries for the same research context
        
        Args:
            research_goal: The original research question/goal
            iteration: Current research iteration
            existing_hypotheses: List of existing hypotheses
            num_queries: Number of queries to generate
            
        Returns:
            List of optimized search queries
        """
        queries = []
        
        # Generate different subtasks for diversity
        diverse_subtasks = [
            self.get_subtask_for_iteration(iteration),
            "Find experimental methodologies and protocols",
            "Identify theoretical frameworks and models",
            "Discover clinical applications and translations"
        ]
        
        for i in range(min(num_queries, len(diverse_subtasks))):
            try:
                # Use different subtask for each query
                custom_subtask = diverse_subtasks[i]
                
                # Extract concepts
                concepts = self._extract_concepts_from_hypotheses(existing_hypotheses or [])
                unique_concepts = list(set(concepts))[:5]
                concepts_text = ", ".join(unique_concepts) if unique_concepts else "No specific concepts identified yet"
                
                # Generate prompt with custom subtask
                prompt = self.base_prompt.format(
                    goal=research_goal,
                    concepts=concepts_text,
                    subtask=custom_subtask
                )
                
                # Get LLM response
                response = await self.llm_service.generate_response(prompt)
                query = self._clean_query_response(response)
                
                # Add filetype filter
                if "filetype:pdf" not in query.lower():
                    query += " filetype:pdf"
                
                queries.append(query)
                
            except Exception as e:
                logger.error(f"Error generating query {i+1}: {e}")
                # Add fallback query
                queries.append(f"{research_goal} research filetype:pdf")
        
        return queries 