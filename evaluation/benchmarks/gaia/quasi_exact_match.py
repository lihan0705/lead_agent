"""
Quasi Exact Match Algorithm for GAIA Evaluation

This module implements the quasi-exact matching algorithm used in GAIA benchmark evaluation.
"""

import re
import json
from typing import Any, Union


def normalize_string(s: str) -> str:
    """
    Normalize a string for comparison
    
    Args:
        s: Input string
        
    Returns:
        Normalized string
    """
    # Convert to lowercase
    s = s.lower()
    
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s)
    
    # Strip leading/trailing whitespace
    s = s.strip()
    
    # Remove common punctuation at the end
    s = re.sub(r'[.,!?;:]+$', '', s)
    
    return s


def normalize_number(value: Union[str, int, float]) -> str:
    """
    Normalize a numeric value for comparison
    
    Args:
        value: Numeric value (string, int, or float)
        
    Returns:
        Normalized string representation
    """
    if isinstance(value, (int, float)):
        return str(value)
    
    # Remove commas from numbers (e.g., "1,234" -> "1234")
    value = re.sub(r',', '', value)
    
    # Try to parse as float
    try:
        num = float(value)
        # If it's an integer, return without decimal
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return normalize_string(value)


def extract_number(s: str) -> Union[float, None]:
    """
    Extract the first number from a string
    
    Args:
        s: Input string
        
    Returns:
        Extracted number or None
    """
    # Try to find a number in the string
    match = re.search(r'-?\d+\.?\d*', s)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def quasi_exact_match(
    prediction: Any, 
    ground_truth: Any, 
    tolerance: float = 0.0
) -> bool:
    """
    Perform quasi-exact match between prediction and ground truth
    
    This is the main evaluation metric used in GAIA benchmark. It handles:
    - String matching with normalization
    - Numeric matching with optional tolerance
    - List/set matching
    - JSON/dict matching
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        tolerance: Numeric tolerance for floating point comparisons
        
    Returns:
        True if prediction matches ground truth, False otherwise
    """
    # Handle None values
    if prediction is None and ground_truth is None:
        return True
    if prediction is None or ground_truth is None:
        return False
    
    # Handle numeric types
    if isinstance(prediction, (int, float)) and isinstance(ground_truth, (int, float)):
        return abs(prediction - ground_truth) <= tolerance
    
    # Handle string types
    if isinstance(prediction, str) and isinstance(ground_truth, str):
        # Try numeric comparison first
        pred_num = extract_number(prediction)
        gt_num = extract_number(ground_truth)
        
        if pred_num is not None and gt_num is not None:
            return abs(pred_num - gt_num) <= tolerance
        
        # Otherwise do string comparison
        return normalize_string(prediction) == normalize_string(ground_truth)
    
    # Handle mixed string/numeric types
    if isinstance(prediction, str) and isinstance(ground_truth, (int, float)):
        pred_num = extract_number(prediction)
        if pred_num is not None:
            return abs(pred_num - ground_truth) <= tolerance
        return normalize_string(prediction) == normalize_string(str(ground_truth))
    
    if isinstance(prediction, (int, float)) and isinstance(ground_truth, str):
        gt_num = extract_number(ground_truth)
        if gt_num is not None:
            return abs(prediction - gt_num) <= tolerance
        return normalize_string(str(prediction)) == normalize_string(ground_truth)
    
    # Handle list/set types
    if isinstance(prediction, (list, set)) and isinstance(ground_truth, (list, set)):
        pred_set = set(prediction) if isinstance(prediction, list) else prediction
        gt_set = set(ground_truth) if isinstance(ground_truth, list) else ground_truth
        return pred_set == gt_set
    
    # Handle dict types
    if isinstance(prediction, dict) and isinstance(ground_truth, dict):
        return json.dumps(prediction, sort_keys=True) == json.dumps(ground_truth, sort_keys=True)
    
    # Default: convert to string and compare
    return normalize_string(str(prediction)) == normalize_string(str(ground_truth))


def batch_quasi_exact_match(
    predictions: list, 
    ground_truths: list, 
    tolerance: float = 0.0
) -> list[bool]:
    """
    Perform batch quasi-exact matching
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        tolerance: Numeric tolerance for floating point comparisons
        
    Returns:
        List of boolean match results
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    
    return [
        quasi_exact_match(pred, gt, tolerance)
        for pred, gt in zip(predictions, ground_truths)
    ]
