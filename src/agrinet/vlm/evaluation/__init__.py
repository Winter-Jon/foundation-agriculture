"""Stable APIs for formal VLM prediction evaluation.

This package is the supported home for evaluation implementations. Adapters and
CLI commands may depend on it; research-route builders must not embed separate
scoring rules.
"""

from .exact_name import evaluate_predictions
from .paired_bootstrap import paired_review

__all__ = ["evaluate_predictions", "paired_review"]
