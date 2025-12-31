"""
GAIA Evaluator

This module provides the main evaluator for the GAIA benchmark.
"""

from typing import List, Dict, Any, Optional, Callable
from .dataset import GAIADataset
from .metrics import GAIAMetrics
from .quasi_exact_match import quasi_exact_match


class GAIAEvaluator:
    """GAIA benchmark evaluator"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize GAIA evaluator
        
        Args:
            data_dir: Path to the GAIA dataset
        """
        self.dataset_loader = GAIADataset(data_dir)
        self.metrics_calculator = GAIAMetrics()
        self.dataset = None
        self.predictions = []
        self.ground_truths = []
        self.questions = []
    
    def load_dataset(self, split: str = "2023_level1", subset: str = "test") -> None:
        """
        Load the GAIA dataset
        
        Args:
            split: Dataset split
            subset: Dataset subset
        """
        self.dataset = self.dataset_loader.load(split, subset)
        self.questions = self.dataset_loader.get_questions()
        self.ground_truths = self.dataset_loader.get_ground_truths()
        print(f"✅ Loaded {len(self.dataset)} examples for evaluation")
    
    def evaluate_predictions(
        self, 
        predictions: List[Any],
        tolerance: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluate a list of predictions against the dataset
        
        Args:
            predictions: List of predictions
            tolerance: Numeric tolerance for floating point comparisons
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if len(predictions) != len(self.ground_truths):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) must match "
                f"number of ground truths ({len(self.ground_truths)})"
            )
        
        self.predictions = predictions
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            predictions, 
            self.ground_truths,
            tolerance
        )
        
        return metrics
    
    def evaluate_agent(
        self, 
        agent: Any,
        tolerance: float = 0.0,
        max_examples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an agent on the GAIA dataset
        
        Args:
            agent: Agent instance with a run() method
            tolerance: Numeric tolerance for floating point comparisons
            max_examples: Maximum number of examples to evaluate (None for all)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        predictions = []
        num_examples = len(self.dataset) if max_examples is None else min(max_examples, len(self.dataset))
        
        print(f"\n🤖 Evaluating agent on {num_examples} examples...")
        print("=" * 60)
        
        for i in range(num_examples):
            example = self.dataset[i]
            question = example['Question']
            
            print(f"\n[{i+1}/{num_examples}] Question: {question[:100]}...")
            
            try:
                # Run the agent
                response = agent.run(question)
                predictions.append(response)
                print(f"✓ Response: {str(response)[:100]}...")
            except Exception as e:
                print(f"✗ Error: {e}")
                predictions.append(f"Error: {e}")
        
        print("\n" + "=" * 60)
        
        # Evaluate predictions
        return self.evaluate_predictions(predictions, tolerance)
    
    def evaluate_single(
        self, 
        prediction: Any, 
        ground_truth: Any,
        tolerance: float = 0.0
    ) -> bool:
        """
        Evaluate a single prediction against ground truth
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            tolerance: Numeric tolerance for floating point comparisons
            
        Returns:
            True if prediction matches ground truth
        """
        return quasi_exact_match(prediction, ground_truth, tolerance)
    
    def get_detailed_results(
        self, 
        predictions: Optional[List[Any]] = None,
        tolerance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get detailed results for each example
        
        Args:
            predictions: Optional list of predictions (uses stored predictions if None)
            tolerance: Numeric tolerance for floating point comparisons
            
        Returns:
            List of detailed result dictionaries
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if predictions is None:
            predictions = self.predictions
        
        if not predictions:
            raise ValueError("No predictions available. Provide predictions or evaluate first.")
        
        return self.metrics_calculator.get_detailed_results(
            predictions,
            self.ground_truths,
            self.questions,
            tolerance
        )
    
    def print_results(self, metrics: Dict[str, Any]) -> None:
        """
        Print formatted evaluation results
        
        Args:
            metrics: Metrics dictionary from evaluate_predictions()
        """
        print(self.metrics_calculator.format_metrics(metrics))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        return self.dataset_loader.get_statistics()
