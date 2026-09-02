"""Compatibility import for the former VLM evaluation module.

New source must import :mod:`agrinet.vlm.evaluation`. This wrapper preserves
the exact scoring API for existing local callers during the migration window.
"""

from agrinet.vlm.evaluation.exact_name import evaluate_predictions

__all__ = ["evaluate_predictions"]
