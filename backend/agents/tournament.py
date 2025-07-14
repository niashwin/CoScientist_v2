"""
Enhanced Tournament System for multi-dimensional hypothesis evaluation
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math
import logging
from dataclasses import dataclass
from enum import Enum

from backend.services.llm import LLMService
from backend.services.memory import MemoryService
from backend.db.models import Hypothesis, TournamentMatch

logger = logging.getLogger(__name__)

class ComparisonType(Enum):
    SINGLE_TURN = "single_turn"
    DEBATE = "debate"
    MULTI_ROUND = "multi_round"

@dataclass
class EvaluationCriteria:
    """Evaluation criteria for hypothesis comparison"""
    novelty_weight: float = 0.3
    feasibility_weight: float = 0.2
    impact_weight: float = 0.3
    testability_weight: float = 0.2
    
    def validate(self) -> bool:
        """Validate that weights sum to 1.0"""
        total = self.novelty_weight + self.feasibility_weight + self.impact_weight + self.testability_weight
        return abs(total - 1.0) < 0.001

@dataclass
class HypothesisScore:
    """Multi-dimensional scores for a hypothesis"""
    novelty: float
    feasibility: float
    impact: float
    testability: float
    composite: float
    confidence: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "novelty": self.novelty,
            "feasibility": self.feasibility,
            "impact": self.impact,
            "testability": self.testability,
            "composite": self.composite,
            "confidence": self.confidence
        }

class EnhancedTournament:
    """
    Enhanced tournament system with multi-dimensional evaluation.
    Implements sophisticated comparison algorithms and Elo rating system.
    """
    
    def __init__(self, llm_service: LLMService, memory_service: MemoryService):
        self.llm_service = llm_service
        self.memory_service = memory_service
        
        # Default evaluation criteria
        self.default_criteria = EvaluationCriteria()
        
        # Tournament configuration
        self.min_comparisons = 10  # Minimum comparisons per hypothesis
        self.confidence_threshold = 0.7
        self.elo_k_factor = 32  # Elo rating adjustment factor
        
        # Evaluation components
        self.evaluators = {
            "novelty": NoveltyEvaluator(llm_service),
            "feasibility": FeasibilityEvaluator(llm_service),
            "impact": ImpactEvaluator(llm_service),
            "testability": TestabilityEvaluator(llm_service)
        }
        
        # Comparison history for learning
        self.comparison_history: List[TournamentMatch] = []
        
    async def run_tournament(
        self,
        hypotheses: List[Hypothesis],
        criteria: Optional[EvaluationCriteria] = None,
        comparison_type: ComparisonType = ComparisonType.SINGLE_TURN,
        research_goal: str = ""
    ) -> List[Tuple[Hypothesis, HypothesisScore]]:
        """
        Run complete tournament with multi-dimensional evaluation
        
        Args:
            hypotheses: List of hypotheses to compare
            criteria: Evaluation criteria (uses default if None)
            comparison_type: Type of comparison to perform
            research_goal: Original research goal for context
            
        Returns:
            List of (hypothesis, score) tuples sorted by composite score
        """
        if len(hypotheses) < 2:
            logger.warning("Tournament requires at least 2 hypotheses")
            return []
        
        criteria = criteria or self.default_criteria
        if not criteria.validate():
            raise ValueError("Evaluation criteria weights must sum to 1.0")
        
        logger.info(f"Starting tournament with {len(hypotheses)} hypotheses using {comparison_type.value}")
        
        # Step 1: Individual scoring
        scored_hypotheses = []
        for hypothesis in hypotheses:
            score = await self._evaluate_hypothesis(hypothesis, criteria, research_goal)
            scored_hypotheses.append((hypothesis, score))
        
        # Step 2: Pairwise comparisons
        comparison_results = await self._run_pairwise_comparisons(
            scored_hypotheses, comparison_type, criteria, research_goal
        )
        
        # Step 3: Update Elo ratings
        await self._update_elo_ratings(comparison_results)
        
        # Step 4: Final ranking
        final_ranking = await self._compute_final_ranking(scored_hypotheses, comparison_results)
        
        # Step 5: Store tournament results
        await self._store_tournament_results(final_ranking, comparison_results)
        
        logger.info(f"Tournament completed. Winner: {final_ranking[0][0].id}")
        return final_ranking
    
    async def _evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        criteria: EvaluationCriteria,
        research_goal: str
    ) -> HypothesisScore:
        """Evaluate a single hypothesis on all dimensions"""
        
        # Evaluate each dimension
        novelty_score = await self.evaluators["novelty"].evaluate(hypothesis, research_goal)
        feasibility_score = await self.evaluators["feasibility"].evaluate(hypothesis, research_goal)
        impact_score = await self.evaluators["impact"].evaluate(hypothesis, research_goal)
        testability_score = await self.evaluators["testability"].evaluate(hypothesis, research_goal)
        
        # Calculate composite score
        composite_score = (
            novelty_score * criteria.novelty_weight +
            feasibility_score * criteria.feasibility_weight +
            impact_score * criteria.impact_weight +
            testability_score * criteria.testability_weight
        )
        
        # Calculate confidence based on score variance
        scores = [novelty_score, feasibility_score, impact_score, testability_score]
        variance = sum((s - composite_score) ** 2 for s in scores) / len(scores)
        confidence = max(0.1, 1.0 - variance)  # Higher variance = lower confidence
        
        return HypothesisScore(
            novelty=novelty_score,
            feasibility=feasibility_score,
            impact=impact_score,
            testability=testability_score,
            composite=composite_score,
            confidence=confidence
        )
    
    async def _run_pairwise_comparisons(
        self,
        scored_hypotheses: List[Tuple[Hypothesis, HypothesisScore]],
        comparison_type: ComparisonType,
        criteria: EvaluationCriteria,
        research_goal: str
    ) -> List[TournamentMatch]:
        """Run pairwise comparisons between hypotheses"""
        
        comparison_results = []
        n = len(scored_hypotheses)
        
        # Generate all pairwise comparisons
        for i in range(n):
            for j in range(i + 1, n):
                hypothesis1, score1 = scored_hypotheses[i]
                hypothesis2, score2 = scored_hypotheses[j]
                
                # Run comparison
                match_result = await self._compare_hypotheses(
                    hypothesis1, score1,
                    hypothesis2, score2,
                    comparison_type, criteria, research_goal
                )
                
                comparison_results.append(match_result)
        
        return comparison_results
    
    async def _compare_hypotheses(
        self,
        hypothesis1: Hypothesis,
        score1: HypothesisScore,
        hypothesis2: Hypothesis,
        score2: HypothesisScore,
        comparison_type: ComparisonType,
        criteria: EvaluationCriteria,
        research_goal: str
    ) -> TournamentMatch:
        """Compare two hypotheses using specified method"""
        
        if comparison_type == ComparisonType.SINGLE_TURN:
            return await self._single_turn_comparison(
                hypothesis1, score1, hypothesis2, score2, criteria, research_goal
            )
        elif comparison_type == ComparisonType.DEBATE:
            return await self._debate_comparison(
                hypothesis1, score1, hypothesis2, score2, criteria, research_goal
            )
        else:
            return await self._multi_round_comparison(
                hypothesis1, score1, hypothesis2, score2, criteria, research_goal
            )
    
    async def _single_turn_comparison(
        self,
        hypothesis1: Hypothesis,
        score1: HypothesisScore,
        hypothesis2: Hypothesis,
        score2: HypothesisScore,
        criteria: EvaluationCriteria,
        research_goal: str
    ) -> TournamentMatch:
        """Single-turn comparison between two hypotheses"""
        
        prompt = f"""
        Compare these two scientific hypotheses for the research goal: {research_goal}
        
        Hypothesis 1:
        {hypothesis1.content}
        
        Hypothesis 2:
        {hypothesis2.content}
        
        Evaluation Criteria:
        - Novelty (weight: {criteria.novelty_weight}): How original and innovative is the hypothesis?
        - Feasibility (weight: {criteria.feasibility_weight}): How testable with current methods?
        - Impact (weight: {criteria.impact_weight}): Potential significance if proven correct?
        - Testability (weight: {criteria.testability_weight}): How clearly defined are the predictions?
        
        Provide detailed comparison and conclude with:
        "Better hypothesis: 1" or "Better hypothesis: 2"
        
        Include specific scores (0-1) for each criterion for both hypotheses.
        """
        
        comparison_result = await self.llm_service.generate_response(prompt)
        
        # Parse the result to determine winner
        winner_id = self._parse_comparison_result(comparison_result, hypothesis1.id, hypothesis2.id)
        
        # Create tournament match record
        match = TournamentMatch(
            id=f"match_{hypothesis1.id}_{hypothesis2.id}",
            research_goal_id="",  # Will be set by caller
            hypothesis_1_id=hypothesis1.id,
            hypothesis_2_id=hypothesis2.id,
            winner_id=winner_id,
            comparison_type="single_turn",
            comparison_rationale=comparison_result,
            criteria_scores={
                "hypothesis_1": score1.to_dict(),
                "hypothesis_2": score2.to_dict()
            },
            created_at=datetime.utcnow(),
            round_number=1
        )
        
        return match
    
    async def _debate_comparison(
        self,
        hypothesis1: Hypothesis,
        score1: HypothesisScore,
        hypothesis2: Hypothesis,
        score2: HypothesisScore,
        criteria: EvaluationCriteria,
        research_goal: str
    ) -> TournamentMatch:
        """Debate-style comparison between two hypotheses"""
        
        debate_prompt = f"""
        You are an expert panel conducting a structured debate to determine which hypothesis is superior.
        
        Research Goal: {research_goal}
        
        Hypothesis 1: {hypothesis1.content}
        Hypothesis 2: {hypothesis2.content}
        
        Conduct a debate with the following structure:
        1. Opening statements for each hypothesis
        2. Critical evaluation of strengths and weaknesses
        3. Addressing potential counterarguments
        4. Final assessment based on evaluation criteria
        
        Evaluation Criteria:
        - Novelty (weight: {criteria.novelty_weight})
        - Feasibility (weight: {criteria.feasibility_weight})
        - Impact (weight: {criteria.impact_weight})
        - Testability (weight: {criteria.testability_weight})
        
        Conclude with: "Better hypothesis: 1" or "Better hypothesis: 2"
        """
        
        debate_result = await self.llm_service.generate_completion(debate_prompt)
        
        # Parse the result
        winner_id = self._parse_comparison_result(debate_result, hypothesis1.id, hypothesis2.id)
        
        # Create tournament match record
        match = TournamentMatch(
            id=f"debate_{hypothesis1.id}_{hypothesis2.id}",
            research_goal_id="",
            hypothesis_1_id=hypothesis1.id,
            hypothesis_2_id=hypothesis2.id,
            winner_id=winner_id,
            comparison_type="debate",
            debate_transcript=debate_result,
            comparison_rationale=debate_result,
            criteria_scores={
                "hypothesis_1": score1.to_dict(),
                "hypothesis_2": score2.to_dict()
            },
            created_at=datetime.utcnow(),
            round_number=1
        )
        
        return match
    
    async def _multi_round_comparison(
        self,
        hypothesis1: Hypothesis,
        score1: HypothesisScore,
        hypothesis2: Hypothesis,
        score2: HypothesisScore,
        criteria: EvaluationCriteria,
        research_goal: str
    ) -> TournamentMatch:
        """Multi-round comparison with iterative refinement"""
        
        # For now, implement as enhanced single-turn
        # Can be expanded to multiple rounds with refinement
        return await self._single_turn_comparison(
            hypothesis1, score1, hypothesis2, score2, criteria, research_goal
        )
    
    def _parse_comparison_result(self, result: str, hypothesis1_id: str, hypothesis2_id: str) -> str:
        """Parse LLM comparison result to determine winner"""
        result_lower = result.lower()
        
        if "better hypothesis: 1" in result_lower:
            return hypothesis1_id
        elif "better hypothesis: 2" in result_lower:
            return hypothesis2_id
        else:
            # Fallback: use simple heuristics
            if "hypothesis 1" in result_lower and "superior" in result_lower:
                return hypothesis1_id
            elif "hypothesis 2" in result_lower and "superior" in result_lower:
                return hypothesis2_id
            else:
                # Default to first hypothesis if unclear
                logger.warning(f"Could not parse comparison result: {result[:100]}...")
                return hypothesis1_id
    
    async def _update_elo_ratings(self, comparison_results: List[TournamentMatch]) -> None:
        """Update Elo ratings based on tournament results"""
        
        # Get current ratings
        hypothesis_ratings = {}
        for match in comparison_results:
            if match.hypothesis_1_id not in hypothesis_ratings:
                hypothesis_ratings[match.hypothesis_1_id] = 1200.0  # Default Elo
            if match.hypothesis_2_id not in hypothesis_ratings:
                hypothesis_ratings[match.hypothesis_2_id] = 1200.0
        
        # Update ratings based on matches
        for match in comparison_results:
            rating1 = hypothesis_ratings[match.hypothesis_1_id]
            rating2 = hypothesis_ratings[match.hypothesis_2_id]
            
            # Calculate expected scores
            expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
            expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
            
            # Determine actual scores
            if match.winner_id == match.hypothesis_1_id:
                actual1, actual2 = 1.0, 0.0
            else:
                actual1, actual2 = 0.0, 1.0
            
            # Update ratings
            new_rating1 = rating1 + self.elo_k_factor * (actual1 - expected1)
            new_rating2 = rating2 + self.elo_k_factor * (actual2 - expected2)
            
            hypothesis_ratings[match.hypothesis_1_id] = new_rating1
            hypothesis_ratings[match.hypothesis_2_id] = new_rating2
        
        # Store updated ratings (would update database in real implementation)
        for hypothesis_id, rating in hypothesis_ratings.items():
            logger.info(f"Updated Elo rating for {hypothesis_id}: {rating:.1f}")
    
    async def _compute_final_ranking(
        self,
        scored_hypotheses: List[Tuple[Hypothesis, HypothesisScore]],
        comparison_results: List[TournamentMatch]
    ) -> List[Tuple[Hypothesis, HypothesisScore]]:
        """Compute final ranking combining individual scores and pairwise comparisons"""
        
        # Calculate win rates from pairwise comparisons
        win_counts = {}
        total_matches = {}
        
        for match in comparison_results:
            for hyp_id in [match.hypothesis_1_id, match.hypothesis_2_id]:
                if hyp_id not in win_counts:
                    win_counts[hyp_id] = 0
                    total_matches[hyp_id] = 0
                total_matches[hyp_id] += 1
            
            win_counts[match.winner_id] += 1
        
        # Calculate win rates
        win_rates = {}
        for hyp_id in win_counts:
            win_rates[hyp_id] = win_counts[hyp_id] / total_matches[hyp_id] if total_matches[hyp_id] > 0 else 0
        
        # Combine individual scores with tournament performance
        final_scores = []
        for hypothesis, score in scored_hypotheses:
            win_rate = win_rates.get(hypothesis.id, 0)
            
            # Weighted combination: 70% individual score, 30% tournament performance
            combined_score = 0.7 * score.composite + 0.3 * win_rate
            
            # Update the score object
            final_score = HypothesisScore(
                novelty=score.novelty,
                feasibility=score.feasibility,
                impact=score.impact,
                testability=score.testability,
                composite=combined_score,
                confidence=score.confidence
            )
            
            final_scores.append((hypothesis, final_score))
        
        # Sort by composite score
        final_scores.sort(key=lambda x: x[1].composite, reverse=True)
        
        return final_scores
    
    async def _store_tournament_results(
        self,
        final_ranking: List[Tuple[Hypothesis, HypothesisScore]],
        comparison_results: List[TournamentMatch]
    ) -> None:
        """Store tournament results in memory service"""
        
        # Update hypothesis scores
        for i, (hypothesis, score) in enumerate(final_ranking):
            hypothesis.novelty_score = score.novelty
            hypothesis.feasibility_score = score.feasibility
            hypothesis.impact_score = score.impact
            hypothesis.testability_score = score.testability
            hypothesis.composite_score = score.composite
            hypothesis.confidence = score.confidence
            
            # Update tournament stats
            hypothesis.tournament_wins = sum(
                1 for match in comparison_results if match.winner_id == hypothesis.id
            )
            hypothesis.tournament_losses = sum(
                1 for match in comparison_results 
                if match.hypothesis_1_id == hypothesis.id or match.hypothesis_2_id == hypothesis.id
            ) - hypothesis.tournament_wins
        
        # Store comparison results
        self.comparison_history.extend(comparison_results)
        
        logger.info(f"Stored tournament results for {len(final_ranking)} hypotheses")

# Individual evaluators for each dimension

class NoveltyEvaluator:
    """Evaluates hypothesis novelty and originality"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def evaluate(self, hypothesis: Hypothesis, research_goal: str) -> float:
        """Evaluate novelty score (0-1)"""
        
        prompt = f"""
        Evaluate the novelty and originality of this scientific hypothesis:
        
        Research Goal: {research_goal}
        Hypothesis: {hypothesis.content}
        
        Consider:
        - How original is the core idea?
        - Does it challenge existing paradigms?
        - Are the proposed mechanisms novel?
        - Could it open new research directions?
        
        Provide a novelty score from 0.0 (not novel) to 1.0 (highly novel).
        Explain your reasoning.
        """
        
        result = await self.llm_service.generate_completion(prompt)
        
        # Parse score from result
        score = self._extract_score(result)
        return max(0.0, min(1.0, score))
    
    def _extract_score(self, result: str) -> float:
        """Extract numerical score from LLM result"""
        # Simple extraction - look for decimal numbers
        import re
        numbers = re.findall(r'\b0\.\d+\b|\b1\.0\b', result)
        if numbers:
            return float(numbers[0])
        
        # Fallback: look for qualitative indicators
        result_lower = result.lower()
        if "highly novel" in result_lower or "very original" in result_lower:
            return 0.9
        elif "novel" in result_lower or "original" in result_lower:
            return 0.7
        elif "somewhat novel" in result_lower:
            return 0.5
        elif "not novel" in result_lower:
            return 0.2
        else:
            return 0.5  # Default middle score

class FeasibilityEvaluator:
    """Evaluates hypothesis feasibility and testability"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def evaluate(self, hypothesis: Hypothesis, research_goal: str) -> float:
        """Evaluate feasibility score (0-1)"""
        
        prompt = f"""
        Evaluate the feasibility of testing this scientific hypothesis:
        
        Research Goal: {research_goal}
        Hypothesis: {hypothesis.content}
        
        Consider:
        - Can it be tested with current technology?
        - What resources would be required?
        - Are the experimental methods available?
        - Timeline for validation?
        
        Provide a feasibility score from 0.0 (not feasible) to 1.0 (highly feasible).
        Explain your reasoning.
        """
        
        result = await self.llm_service.generate_completion(prompt)
        score = self._extract_score(result)
        return max(0.0, min(1.0, score))
    
    def _extract_score(self, result: str) -> float:
        """Extract numerical score from LLM result"""
        import re
        numbers = re.findall(r'\b0\.\d+\b|\b1\.0\b', result)
        if numbers:
            return float(numbers[0])
        
        result_lower = result.lower()
        if "highly feasible" in result_lower or "very feasible" in result_lower:
            return 0.9
        elif "feasible" in result_lower:
            return 0.7
        elif "somewhat feasible" in result_lower:
            return 0.5
        elif "not feasible" in result_lower or "infeasible" in result_lower:
            return 0.2
        else:
            return 0.5

class ImpactEvaluator:
    """Evaluates potential impact and significance"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def evaluate(self, hypothesis: Hypothesis, research_goal: str) -> float:
        """Evaluate impact score (0-1)"""
        
        prompt = f"""
        Evaluate the potential impact of this scientific hypothesis:
        
        Research Goal: {research_goal}
        Hypothesis: {hypothesis.content}
        
        Consider:
        - Potential to advance the field?
        - Clinical or practical applications?
        - Broader scientific implications?
        - Could it lead to breakthrough discoveries?
        
        Provide an impact score from 0.0 (low impact) to 1.0 (high impact).
        Explain your reasoning.
        """
        
        result = await self.llm_service.generate_completion(prompt)
        score = self._extract_score(result)
        return max(0.0, min(1.0, score))
    
    def _extract_score(self, result: str) -> float:
        """Extract numerical score from LLM result"""
        import re
        numbers = re.findall(r'\b0\.\d+\b|\b1\.0\b', result)
        if numbers:
            return float(numbers[0])
        
        result_lower = result.lower()
        if "high impact" in result_lower or "significant impact" in result_lower:
            return 0.9
        elif "moderate impact" in result_lower:
            return 0.6
        elif "low impact" in result_lower:
            return 0.3
        else:
            return 0.5

class TestabilityEvaluator:
    """Evaluates hypothesis testability and falsifiability"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def evaluate(self, hypothesis: Hypothesis, research_goal: str) -> float:
        """Evaluate testability score (0-1)"""
        
        prompt = f"""
        Evaluate the testability of this scientific hypothesis:
        
        Research Goal: {research_goal}
        Hypothesis: {hypothesis.content}
        
        Consider:
        - Are the predictions specific and measurable?
        - Can the hypothesis be falsified?
        - Are there clear experimental outcomes?
        - How well-defined are the variables?
        
        Provide a testability score from 0.0 (not testable) to 1.0 (highly testable).
        Explain your reasoning.
        """
        
        result = await self.llm_service.generate_completion(prompt)
        score = self._extract_score(result)
        return max(0.0, min(1.0, score))
    
    def _extract_score(self, result: str) -> float:
        """Extract numerical score from LLM result"""
        import re
        numbers = re.findall(r'\b0\.\d+\b|\b1\.0\b', result)
        if numbers:
            return float(numbers[0])
        
        result_lower = result.lower()
        if "highly testable" in result_lower or "very testable" in result_lower:
            return 0.9
        elif "testable" in result_lower:
            return 0.7
        elif "somewhat testable" in result_lower:
            return 0.5
        elif "not testable" in result_lower or "untestable" in result_lower:
            return 0.2
        else:
            return 0.5 