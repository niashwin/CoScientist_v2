"""
Adaptive Prompt Manager for dynamic prompt optimization
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json
import logging
from pathlib import Path

from backend.core.config import settings
from backend.services.memory import MemoryService

logger = logging.getLogger(__name__)

class AdaptivePromptManager:
    """
    Manages dynamic prompt optimization based on agent performance and context.
    Adapts prompts based on success patterns, domain-specific requirements,
    and accumulated feedback from the meta-review system.
    """
    
    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service
        
        # Load base prompts
        self.base_prompts = self._load_base_prompts()
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.success_patterns = defaultdict(dict)
        self.failure_patterns = defaultdict(dict)
        
        # Meta-feedback accumulation
        self.meta_feedback = defaultdict(list)
        
        # Context-specific instructions
        self.domain_instructions = {
            "drug_discovery": {
                "generation": "Focus on molecular targets, drug-target interactions, and pharmacological mechanisms",
                "reflection": "Evaluate drug safety, efficacy, and regulatory considerations",
                "ranking": "Prioritize clinical relevance and translational potential"
            },
            "basic_research": {
                "generation": "Emphasize fundamental mechanisms and theoretical frameworks",
                "reflection": "Assess scientific rigor and reproducibility",
                "ranking": "Value novelty and potential for paradigm shifts"
            },
            "biotechnology": {
                "generation": "Consider scalability, manufacturing feasibility, and commercial viability",
                "reflection": "Evaluate technical feasibility and market potential",
                "ranking": "Balance innovation with practical implementation"
            }
        }
        
        # Optimization parameters
        self.optimization_window = timedelta(hours=24)  # Consider recent performance
        self.min_samples_for_optimization = 5
        self.performance_threshold = 0.7
        
    def _load_base_prompts(self) -> Dict[str, str]:
        """Load base prompt templates from files"""
        prompts = {}
        prompt_dir = Path(__file__).parent.parent / "prompts"
        
        for prompt_file in prompt_dir.glob("*.prompt"):
            agent_name = prompt_file.stem
            with open(prompt_file, 'r') as f:
                prompts[agent_name] = f.read()
        
        return prompts
    
    def get_optimized_prompt(self, agent_type: str, context: Dict[str, Any]) -> str:
        """
        Get dynamically optimized prompt based on performance and context
        
        Args:
            agent_type: Type of agent (generation, reflection, ranking, etc.)
            context: Current session context including goal, preferences, etc.
            
        Returns:
            Optimized prompt string
        """
        # Start with base prompt
        base_prompt = self.base_prompts.get(agent_type, "")
        
        if not base_prompt:
            logger.warning(f"No base prompt found for agent type: {agent_type}")
            return ""
        
        # Analyze performance patterns
        success_patterns = self._analyze_success_patterns(agent_type)
        failure_patterns = self._analyze_failure_patterns(agent_type)
        
        # Get domain-specific instructions
        domain_instructions = self._get_domain_specific_instructions(agent_type, context)
        
        # Get accumulated meta-feedback
        feedback_instructions = self._get_accumulated_feedback(agent_type)
        
        # Combine all elements
        optimized_prompt = self._combine_prompt_elements(
            base_prompt=base_prompt,
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
            domain_instructions=domain_instructions,
            feedback_instructions=feedback_instructions,
            context=context
        )
        
        logger.info(f"Generated optimized prompt for {agent_type} (length: {len(optimized_prompt)})")
        return optimized_prompt
    
    def _analyze_success_patterns(self, agent_type: str) -> Dict[str, Any]:
        """Analyze successful agent performances to identify patterns"""
        recent_performances = self._get_recent_performances(agent_type)
        
        if not recent_performances:
            return {}
        
        # Filter successful performances
        successful_performances = [
            p for p in recent_performances 
            if p.get('success', False) and p.get('quality_score', 0) > self.performance_threshold
        ]
        
        if len(successful_performances) < self.min_samples_for_optimization:
            return {}
        
        # Extract patterns from successful performances
        patterns = {
            "successful_contexts": [],
            "effective_instructions": [],
            "optimal_parameters": {}
        }
        
        for performance in successful_performances:
            context = performance.get('context', {})
            
            # Analyze context patterns
            if 'research_domain' in context:
                patterns["successful_contexts"].append(context['research_domain'])
            
            # Analyze instruction patterns
            if 'custom_instructions' in context:
                patterns["effective_instructions"].append(context['custom_instructions'])
            
            # Analyze parameter patterns
            if 'parameters' in context:
                for param, value in context['parameters'].items():
                    if param not in patterns["optimal_parameters"]:
                        patterns["optimal_parameters"][param] = []
                    patterns["optimal_parameters"][param].append(value)
        
        return patterns
    
    def _analyze_failure_patterns(self, agent_type: str) -> Dict[str, Any]:
        """Analyze failed agent performances to identify anti-patterns"""
        recent_performances = self._get_recent_performances(agent_type)
        
        if not recent_performances:
            return {}
        
        # Filter failed performances
        failed_performances = [
            p for p in recent_performances 
            if not p.get('success', True) or p.get('quality_score', 1) < self.performance_threshold
        ]
        
        if not failed_performances:
            return {}
        
        # Extract anti-patterns from failed performances
        patterns = {
            "problematic_contexts": [],
            "ineffective_instructions": [],
            "problematic_parameters": {}
        }
        
        for performance in failed_performances:
            context = performance.get('context', {})
            
            # Analyze problematic contexts
            if 'research_domain' in context:
                patterns["problematic_contexts"].append(context['research_domain'])
            
            # Analyze ineffective instructions
            if 'custom_instructions' in context:
                patterns["ineffective_instructions"].append(context['custom_instructions'])
            
            # Analyze problematic parameters
            if 'parameters' in context:
                for param, value in context['parameters'].items():
                    if param not in patterns["problematic_parameters"]:
                        patterns["problematic_parameters"][param] = []
                    patterns["problematic_parameters"][param].append(value)
        
        return patterns
    
    def _get_recent_performances(self, agent_type: str) -> List[Dict[str, Any]]:
        """Get recent performance data for an agent type"""
        cutoff_time = datetime.utcnow() - self.optimization_window
        
        recent_performances = [
            p for p in self.performance_history[agent_type]
            if p.get('timestamp', datetime.min) > cutoff_time
        ]
        
        return recent_performances
    
    def _get_domain_specific_instructions(self, agent_type: str, context: Dict[str, Any]) -> str:
        """Get domain-specific instructions based on context"""
        research_goal = context.get('goal', '').lower()
        preferences = context.get('preferences', {})
        
        # Detect domain from research goal and preferences
        domain = self._detect_research_domain(research_goal, preferences)
        
        if domain and domain in self.domain_instructions:
            return self.domain_instructions[domain].get(agent_type, "")
        
        return ""
    
    def _detect_research_domain(self, research_goal: str, preferences: Dict[str, Any]) -> Optional[str]:
        """Detect research domain from goal and preferences"""
        # Simple keyword-based detection - can be enhanced with ML
        drug_keywords = ['drug', 'therapeutic', 'treatment', 'medicine', 'pharmaceutical', 'clinical']
        biotech_keywords = ['biotech', 'biotechnology', 'commercial', 'industrial', 'scale', 'manufacturing']
        basic_keywords = ['mechanism', 'fundamental', 'basic', 'theoretical', 'molecular', 'cellular']
        
        goal_lower = research_goal.lower()
        
        if any(keyword in goal_lower for keyword in drug_keywords):
            return "drug_discovery"
        elif any(keyword in goal_lower for keyword in biotech_keywords):
            return "biotechnology"
        elif any(keyword in goal_lower for keyword in basic_keywords):
            return "basic_research"
        
        # Check preferences
        focus_area = preferences.get('focus_area', '').lower()
        if 'drug' in focus_area or 'therapeutic' in focus_area:
            return "drug_discovery"
        elif 'biotech' in focus_area:
            return "biotechnology"
        elif 'basic' in focus_area or 'fundamental' in focus_area:
            return "basic_research"
        
        return None
    
    def _get_accumulated_feedback(self, agent_type: str) -> str:
        """Get accumulated meta-feedback for an agent type"""
        if agent_type not in self.meta_feedback:
            return ""
        
        # Get recent feedback
        recent_feedback = self.meta_feedback[agent_type][-5:]  # Last 5 feedback items
        
        if not recent_feedback:
            return ""
        
        # Synthesize feedback into instructions
        feedback_instructions = []
        
        for feedback in recent_feedback:
            if 'improvement_suggestion' in feedback:
                feedback_instructions.append(feedback['improvement_suggestion'])
            
            if 'common_issues' in feedback:
                for issue in feedback['common_issues']:
                    feedback_instructions.append(f"Avoid: {issue}")
        
        if feedback_instructions:
            return "Based on recent feedback:\n" + "\n".join(f"- {instruction}" for instruction in feedback_instructions)
        
        return ""
    
    def _combine_prompt_elements(
        self,
        base_prompt: str,
        success_patterns: Dict[str, Any],
        failure_patterns: Dict[str, Any],
        domain_instructions: str,
        feedback_instructions: str,
        context: Dict[str, Any]
    ) -> str:
        """Combine all prompt elements into optimized prompt"""
        
        # Start with base prompt
        optimized_prompt = base_prompt
        
        # Add domain-specific instructions
        if domain_instructions:
            optimized_prompt += f"\n\nDomain-specific guidance:\n{domain_instructions}"
        
        # Add success pattern instructions
        if success_patterns and success_patterns.get('effective_instructions'):
            effective_instructions = success_patterns['effective_instructions']
            if effective_instructions:
                # Take most common effective instruction
                most_common = max(set(effective_instructions), key=effective_instructions.count)
                optimized_prompt += f"\n\nBased on successful patterns:\n{most_common}"
        
        # Add failure avoidance instructions
        if failure_patterns and failure_patterns.get('ineffective_instructions'):
            ineffective_instructions = failure_patterns['ineffective_instructions']
            if ineffective_instructions:
                # Take most common ineffective instruction to avoid
                most_common = max(set(ineffective_instructions), key=ineffective_instructions.count)
                optimized_prompt += f"\n\nAvoid the following approach:\n{most_common}"
        
        # Add accumulated feedback
        if feedback_instructions:
            optimized_prompt += f"\n\n{feedback_instructions}"
        
        # Add context-specific adaptations
        if context.get('iteration', 0) > 0:
            optimized_prompt += f"\n\nThis is iteration {context['iteration']} of the research process. Build upon previous findings while exploring new directions."
        
        return optimized_prompt
    
    def update_performance(self, agent_type: str, result: Dict[str, Any]) -> None:
        """
        Update agent performance tracking
        
        Args:
            agent_type: Type of agent
            result: Performance result including success, quality_score, context, etc.
        """
        performance_record = {
            'timestamp': datetime.utcnow(),
            'success': result.get('success', False),
            'quality_score': result.get('quality_score', 0.0),
            'context': result.get('context', {}),
            'execution_time': result.get('execution_time', 0),
            'error_message': result.get('error_message')
        }
        
        self.performance_history[agent_type].append(performance_record)
        
        # Keep only recent history (last 100 records per agent)
        if len(self.performance_history[agent_type]) > 100:
            self.performance_history[agent_type] = self.performance_history[agent_type][-100:]
        
        logger.info(f"Updated performance for {agent_type}: success={result.get('success')}, quality={result.get('quality_score')}")
    
    def add_meta_feedback(self, agent_type: str, feedback: Dict[str, Any]) -> None:
        """
        Add meta-feedback from the meta-review agent
        
        Args:
            agent_type: Type of agent the feedback is for
            feedback: Meta-feedback including improvement suggestions, common issues, etc.
        """
        feedback_record = {
            'timestamp': datetime.utcnow(),
            'improvement_suggestion': feedback.get('improvement_suggestion'),
            'common_issues': feedback.get('common_issues', []),
            'success_patterns': feedback.get('success_patterns', {}),
            'source': feedback.get('source', 'meta_review')
        }
        
        self.meta_feedback[agent_type].append(feedback_record)
        
        # Keep only recent feedback (last 20 records per agent)
        if len(self.meta_feedback[agent_type]) > 20:
            self.meta_feedback[agent_type] = self.meta_feedback[agent_type][-20:]
        
        logger.info(f"Added meta-feedback for {agent_type}: {feedback.get('improvement_suggestion', 'No suggestion')}")
    
    def get_performance_stats(self, agent_type: str) -> Dict[str, Any]:
        """Get performance statistics for an agent type"""
        performances = self.performance_history[agent_type]
        
        if not performances:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_quality": 0.0,
                "average_execution_time": 0.0
            }
        
        total = len(performances)
        successes = sum(1 for p in performances if p.get('success', False))
        quality_scores = [p.get('quality_score', 0) for p in performances if p.get('quality_score') is not None]
        execution_times = [p.get('execution_time', 0) for p in performances if p.get('execution_time') is not None]
        
        return {
            "total_executions": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            "average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0.0,
            "recent_trend": self._calculate_recent_trend(performances)
        }
    
    def _calculate_recent_trend(self, performances: List[Dict[str, Any]]) -> str:
        """Calculate recent performance trend"""
        if len(performances) < 4:
            return "insufficient_data"
        
        # Compare recent half with earlier half
        mid_point = len(performances) // 2
        recent_half = performances[mid_point:]
        earlier_half = performances[:mid_point]
        
        recent_success_rate = sum(1 for p in recent_half if p.get('success', False)) / len(recent_half)
        earlier_success_rate = sum(1 for p in earlier_half if p.get('success', False)) / len(earlier_half)
        
        if recent_success_rate > earlier_success_rate + 0.1:
            return "improving"
        elif recent_success_rate < earlier_success_rate - 0.1:
            return "declining"
        else:
            return "stable"
    
    def export_optimization_data(self) -> Dict[str, Any]:
        """Export optimization data for analysis"""
        return {
            "performance_history": dict(self.performance_history),
            "meta_feedback": dict(self.meta_feedback),
            "success_patterns": dict(self.success_patterns),
            "failure_patterns": dict(self.failure_patterns),
            "export_timestamp": datetime.utcnow().isoformat()
        }
    
    def import_optimization_data(self, data: Dict[str, Any]) -> None:
        """Import optimization data from previous sessions"""
        if "performance_history" in data:
            for agent_type, history in data["performance_history"].items():
                self.performance_history[agent_type].extend(history)
        
        if "meta_feedback" in data:
            for agent_type, feedback in data["meta_feedback"].items():
                self.meta_feedback[agent_type].extend(feedback)
        
        logger.info("Imported optimization data from previous sessions") 