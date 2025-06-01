"""
Determines intent based on similarity search results and context.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

from app.services.search.async_vector_search import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class IntentScore:
    """
    Representation of an intent's performance score with associated metrics.

    This class is used to assess and represent the performance and confidence of a
    specific intent in a system by evaluating its score, evidence count, and success
    rate. It provides a property to compute the overall confidence based on these
    attributes.

    Attributes
    ----------
    intent : str
        Name of the intent being evaluated.
    score : float
        Base performance score of the intent, typically in the range [0.0, 1.0].
    evidence_count : int
        Number of supporting evidence items used for scoring.
    success_rate : float
        Success rate of the intent, typically in the range [0.0, 1.0].
    """

    intent: str
    score: float
    evidence_count: int
    success_rate: float

    @property
    def confidence(self) -> float:
        """Calculate overall confidence."""
        if self.score >= 0.6:
            scaled_score = 0.7 + (self.score - 0.6) * 0.75
        else:
            scaled_score = self.score
        evidence_weight = min(self.evidence_count / 3, 1.0)
        confidence = (
            scaled_score * 0.6 + self.success_rate * 0.3 + evidence_weight * 0.1
        )
        return min(confidence, 0.95)  # Cap at 95%


class IntentDeterminer:
    """
    IntentDeterminer class determines the most likely user intent based on search results,
    contextual data, and similarity scores.

    The purpose of this class is to evaluate search results, analyze them in relation to a
    given context, and calculate scores for potential user intents. It applies different
    weights and adjustments to ensure the most accurate determination of the user's intent.
    This includes functionality to adjust intent scores based on contextual keywords, as
    well as confidence boosting based on similarity scores.

    Attributes
    ----------
    min_evidence_count : int
        The minimum number of pieces of evidence required to consider an intent as valid.
    success_weight : float
        The multiplier weight applied to positive search result feedback.
    failure_weight : float
        The multiplier weight applied to negative search result feedback.
    min_similarity_for_confidence : float
        The minimum average similarity threshold required to boost intent confidence.
    """

    def __init__(self):
        self.min_evidence_count = 1
        self.success_weight = 1.2
        self.failure_weight = 0.8
        self.min_similarity_for_confidence = 0.6

    def determine_intent(
        self,
        search_results: List[SearchResult],
        query_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Determine the most likely intent from search results.
        """
        if not search_results:
            return self._default_intent()

        # Calculate intent scores
        intent_scores = self._calculate_intent_scores(search_results)

        # Apply context adjustments
        if query_context:
            intent_scores = self._apply_context_adjustments(
                intent_scores, query_context
            )

        # Select best intent
        best_intent = self._select_best_intent(intent_scores)

        # BOOST confidence if we have good similarity scores
        avg_similarity = self._calculate_avg_similarity(search_results)
        if avg_similarity >= self.min_similarity_for_confidence:
            # Boost confidence based on similarity
            boost_factor = 1 + (avg_similarity - self.min_similarity_for_confidence)
            best_intent.score = min(best_intent.score * boost_factor, 0.95)
            logger.info(
                f"Boosted intent confidence due to similarity {avg_similarity:.3f}"
            )

        # Build analysis
        analysis = self._build_analysis(best_intent, intent_scores, search_results)

        return best_intent.intent, best_intent.confidence, analysis

    def _calculate_intent_scores(
        self, search_results: List[SearchResult]
    ) -> Dict[str, IntentScore]:
        """Calculate scores for each intent."""
        intent_data = defaultdict(
            lambda: {"total_score": 0.0, "count": 0, "success_count": 0}
        )

        # Aggregate scores by intent
        for result in search_results:
            intent = result.intent
            similarity = result.similarity_score

            # Weight by success/failure
            weight = (
                self.success_weight if result.success_indicator else self.failure_weight
            )

            intent_data[intent]["total_score"] += similarity * weight
            intent_data[intent]["count"] += 1

            if result.success_indicator:
                intent_data[intent]["success_count"] += 1

        # Convert to IntentScore objects
        intent_scores = {}
        for intent, data in intent_data.items():
            avg_score = data["total_score"] / data["count"]
            success_rate = data["success_count"] / data["count"]

            intent_scores[intent] = IntentScore(
                intent=intent,
                score=avg_score,
                evidence_count=data["count"],
                success_rate=success_rate,
            )

        return intent_scores

    @staticmethod
    def _apply_context_adjustments(
        intent_scores: Dict[str, IntentScore], query_context: Dict[str, Any]
    ) -> Dict[str, IntentScore]:
        """Apply context-based adjustments to scores."""
        # Check for explicit intent signals in context
        if "keywords" in query_context:
            keywords = query_context["keywords"]

            # Boost intents matching keywords
            keyword_intents = {
                "transaction": ["transaction", "payment", "expense"],
                "invoice": ["invoice", "bill", "client"],
                "forecast": ["forecast", "predict", "future"],
                "budget": ["budget", "limit", "spending"],
            }

            for intent, score in intent_scores.items():
                for keyword_intent, words in keyword_intents.items():
                    if intent == keyword_intent and any(
                        word in keywords for word in words
                    ):
                        # Boost score by 20%
                        score.score *= 1.2

        return intent_scores

    def _select_best_intent(self, intent_scores: Dict[str, IntentScore]) -> IntentScore:
        """Select the best intent based on scores."""
        if not intent_scores:
            return IntentScore(
                intent="general", score=0.5, evidence_count=0, success_rate=0.5
            )

        # Sort by confidence
        sorted_intents = sorted(
            intent_scores.values(), key=lambda x: x.confidence, reverse=True
        )

        best_intent = sorted_intents[0]

        # Check if we have enough evidence
        if best_intent.evidence_count < self.min_evidence_count:
            # Reduce confidence if insufficient evidence
            best_intent.score *= 0.8

        return best_intent

    def _build_analysis(
        self,
        best_intent: IntentScore,
        all_scores: Dict[str, IntentScore],
        search_results: List[SearchResult],
    ) -> Dict[str, Any]:
        """Build detailed analysis of the determination."""
        # Get alternative intents
        alternatives = []
        for intent, score in all_scores.items():
            if intent != best_intent.intent:
                alternatives.append(
                    {
                        "intent": intent,
                        "confidence": score.confidence,
                        "evidence_count": score.evidence_count,
                    }
                )

        # Sort alternatives by confidence
        alternatives.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "method": "similarity_search",
            "evidence_count": best_intent.evidence_count,
            "success_rate": best_intent.success_rate,
            "alternatives": alternatives[:2],  # Top 2 alternatives
            "search_results_count": len(search_results),
            "avg_similarity": self._calculate_avg_similarity(search_results),
        }

    @staticmethod
    def _calculate_avg_similarity(search_results: List[SearchResult]) -> float:
        """Calculate average similarity score."""
        if not search_results:
            return 0.0

        total = sum(r.similarity_score for r in search_results)
        return total / len(search_results)

    @staticmethod
    def _default_intent() -> Tuple[str, float, Dict[str, Any]]:
        """Return default intent when no results found."""
        return (
            "general",
            0.5,
            {"method": "default", "reason": "No similar conversations found"},
        )
