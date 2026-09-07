"""Approved canonical-label tooling for the OpenAgri v2 canonical-v1 line."""

from .registry import (
    CANONICAL_VERSION,
    MERGED_SOURCE_TO_CANONICAL,
    Registry,
    build_registry_rows,
    load_registry,
    validate_registry,
)
from .retrieval import CanonicalCandidateCardBackend

__all__ = [
    "CANONICAL_VERSION",
    "MERGED_SOURCE_TO_CANONICAL",
    "Registry",
    "build_registry_rows",
    "load_registry",
    "validate_registry",
    "CanonicalCandidateCardBackend",
]
