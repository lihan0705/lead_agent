"""
GAIA Evaluation Metrics

This module provides metrics for evaluating GAIA benchmark performance.
"""

from typing import List, Dict, Any
from .quasi_exact_match import quasi_exact_match, batch_quasi_exact_match


class GAIAMetrics:
    """GAIA benchmark metrics calculator"""
    
    @staticmethod
    def accuracy(matches: List[bool]) -> float:
        """
        Calculate accuracy (percentage of correct predictions)
        
        Args:
            matches: List of boolean match results
            
        Returns:
            Accuracy score (0-1)
        """
        if not matches:
            return 0.0
        return sum(matches) / len(matches)
    
    @staticmethod
    def exact_match(predictions: List[Any], ground_truths: List[Any]) -> float:
        """
        Calculate exact match accuracy using quasi-exact matching
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            
        Returns:
            Exact match accuracy (0-1)
        """
        matches = batch_quasi_exact_match(predictions, ground_truths)
        return GAIAMetrics.accuracy(matches)
    
    @staticmethod
    def calculate_metrics(
        predictions: List[Any], 
        ground_truths: List[Any],
        tolerance: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for GAIA evaluation
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            tolerance: Numeric tolerance for floating point comparisons
            
        Returns:
            Dictionary containing various metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length")
        
        matches = batch_quasi_exact_match(predictions, ground_truths, tolerance)
        
        total = len(predictions)
        correct = sum(matches)
        incorrect = total - correct
        
        metrics = {
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': correct / total if total > 0 else 0.0,
            'matches': matches,
            'tolerance': tolerance
        }
        
        return metrics
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> str:
        """
        Format metrics for display
        
        Args:
            metrics: Metrics dictionary from calculate_metrics()
            
        Returns:
            Formatted string
        """
        lines = [
            "=" * 60,
            "GAIA Benchmark Evaluation Results",
            "=" * 60,
            f"Total examples: {metrics['total']}",
            f"Correct: {metrics['correct']}",
            f"Incorrect: {metrics['incorrect']}",
            f"Accuracy: {metrics['accuracy']:.2%}",
            "=" * 60
        ]
        return "\n".join(lines)
    
    @staticmethod
    def get_detailed_results(
        predictions: List[Any], 
        ground_truths: List[Any],
        questions: List[str] = None,
        tolerance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get detailed results for each example
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            questions: Optional list of questions
            tolerance: Numeric tolerance for floating point comparisons
            
        Returns:
            List of detailed result dictionaries
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length")
        
        if questions and len(questions) != len(predictions):
            raise ValueError("Questions must have the same length as predictions")
        
        results = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            match = quasi_exact_match(pred, gt, tolerance)
            
            result = {
                'index': i,
                'prediction': pred,
                'ground_truth': gt,
                'match': match
            }
            
            if questions:
                result['question'] = questions[i]
            
            results.append(result)
        
        return results
