"""
GAIA Evaluation Tool

This tool provides a simple interface for evaluating agents on the GAIA benchmark.
"""

from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime
from ..benchmarks.gaia import GAIADataset, GAIAEvaluator, GAIAMetrics


class GAIAEvaluationTool:
    """GAIA benchmark evaluation tool"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize GAIA evaluation tool
        
        Args:
            data_dir: Path to the GAIA dataset directory
        """
        self.data_dir = data_dir
        self.evaluator = GAIAEvaluator(data_dir)
        self.results = None
    
    def run(
        self,
        agent: Any,
        level: int = 1,
        max_samples: Optional[int] = None,
        export_results: bool = True,
        generate_report: bool = True,
        output_dir: str = "./gaia_results",
        tolerance: float = 0.0
    ) -> Dict[str, Any]:
        """
        Run GAIA evaluation on the given agent
        
        Args:
            agent: Agent instance with a run() method
            level: GAIA difficulty level (1, 2, or 3)
            max_samples: Maximum number of samples to evaluate (None for all)
            export_results: Whether to export results in GAIA format
            generate_report: Whether to generate evaluation report
            output_dir: Directory to save results
            tolerance: Numeric tolerance for floating point comparisons
            
        Returns:
            Dictionary containing evaluation results
        """
        # Load dataset for the specified level
        split = f"2023_level{level}"
        print(f"\n📊 Loading GAIA dataset: {split}")
        self.evaluator.load_dataset(split=split, subset="test")
        
        # Get dataset statistics
        stats = self.evaluator.get_statistics()
        print(f"📈 Dataset statistics:")
        print(f"   Total examples: {stats['total_examples']}")
        print(f"   Examples with files: {stats['examples_with_files']}")
        print(f"   Examples without files: {stats['examples_without_files']}")
        
        # Evaluate agent
        num_samples = min(max_samples, len(self.evaluator.dataset)) if max_samples else len(self.evaluator.dataset)
        print(f"\n🤖 Evaluating agent on {num_samples} examples...")
        
        metrics = self.evaluator.evaluate_agent(
            agent=agent,
            tolerance=tolerance,
            max_examples=max_samples
        )
        
        # Calculate additional metrics
        total_samples = len(self.evaluator.predictions)
        exact_matches = sum(metrics['matches'])
        exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0.0
        
        # Calculate partial match (if answer contains correct information)
        partial_matches = self._calculate_partial_matches(
            self.evaluator.predictions,
            self.evaluator.ground_truths
        )
        partial_match_rate = partial_matches / total_samples if total_samples > 0 else 0.0
        
        # Compile results
        self.results = {
            'exact_match_rate': exact_match_rate,
            'partial_match_rate': partial_match_rate,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'total_samples': total_samples,
            'level': level,
            'metrics': metrics,
            'detailed_results': self.evaluator.get_detailed_results(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Print results
        self._print_results()
        
        # Export results if requested
        if export_results:
            self._export_results(output_dir)
        
        # Generate report if requested
        if generate_report:
            self._generate_report(output_dir)
        
        return self.results
    
    def _calculate_partial_matches(
        self,
        predictions: List[Any],
        ground_truths: List[Any]
    ) -> int:
        """
        Calculate partial matches (answers that contain correct information)
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            
        Returns:
            Number of partial matches
        """
        partial_matches = 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_str = str(pred).lower()
            gt_str = str(gt).lower()
            
            # Check if prediction contains ground truth (or vice versa)
            if gt_str in pred_str or pred_str in gt_str:
                partial_matches += 1
        
        return partial_matches
    
    def _print_results(self) -> None:
        """Print evaluation results"""
        print("\n" + "=" * 60)
        print("GAIA BENCHMARK EVALUATION RESULTS")
        print("=" * 60)
        print(f"Level: {self.results['level']}")
        print(f"Total samples: {self.results['total_samples']}")
        print(f"Exact matches: {self.results['exact_matches']}")
        print(f"Partial matches: {self.results['partial_matches']}")
        print(f"\nExact match rate: {self.results['exact_match_rate']:.2%}")
        print(f"Partial match rate: {self.results['partial_match_rate']:.2%}")
        print("=" * 60)
    
    def _export_results(self, output_dir: str) -> None:
        """
        Export results in GAIA format
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export in GAIA format
        gaia_format = []
        for i, (pred, gt, match) in enumerate(zip(
            self.evaluator.predictions,
            self.evaluator.ground_truths,
            self.results['metrics']['matches']
        )):
            gaia_format.append({
                'question_id': i,
                'question': self.evaluator.questions[i] if i < len(self.evaluator.questions) else '',
                'model_answer': str(pred),
                'ground_truth': str(gt),
                'correct': match
            })
        
        # Save GAIA format results
        gaia_file = os.path.join(output_dir, f"gaia_results_level{self.results['level']}.json")
        with open(gaia_file, 'w', encoding='utf-8') as f:
            json.dump(gaia_format, f, indent=2, ensure_ascii=False)
        
        # Save detailed results
        detailed_file = os.path.join(output_dir, f"gaia_detailed_results_level{self.results['level']}.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results exported to: {output_dir}")
    
    def _generate_report(self, output_dir: str) -> None:
        """
        Generate evaluation report
        
        Args:
            output_dir: Directory to save report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report_file = os.path.join(output_dir, f"gaia_report_level{self.results['level']}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GAIA BENCHMARK EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Evaluation Time: {self.results['timestamp']}\n")
            f.write(f"Level: {self.results['level']}\n")
            f.write(f"Total Samples: {self.results['total_samples']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("OVERALL RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Exact Matches: {self.results['exact_matches']}/{self.results['total_samples']}\n")
            f.write(f"Exact Match Rate: {self.results['exact_match_rate']:.2%}\n")
            f.write(f"Partial Matches: {self.results['partial_matches']}/{self.results['total_samples']}\n")
            f.write(f"Partial Match Rate: {self.results['partial_match_rate']:.2%}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("-" * 80 + "\n")
            
            for result in self.results['detailed_results']:
                f.write(f"\n[Example {result['index'] + 1}]\n")
                f.write(f"Question: {result.get('question', 'N/A')}\n")
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Match: {'✓' if result['match'] else '✗'}\n")
        
        print(f"✅ Report generated: {report_file}")
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the last evaluation results
        
        Returns:
            Results dictionary or None if no evaluation has been run
        """
        return self.results
