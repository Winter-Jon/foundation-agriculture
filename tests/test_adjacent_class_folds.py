from scripts.vision.build_adjacent_class_folds import (
    assignment_violations,
    build_known_rag_graph,
    candidate_assignment,
    load_known_classes,
    solve_assignment,
)
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_known_graph_has_bridge_for_every_class() -> None:
    known = load_known_classes(ROOT / "datasets/AgriNet-1K/open_agri_v3")
    graph, metadata = build_known_rag_graph(
        known, ROOT / "outputs/milvus/wiki_similar_classes_siglip2_report.top12.json"
    )
    assert len(known) == len(graph) == 107
    assert metadata["symmetrized"] is False
    assert all(graph.values())
    assert all(all(known[item]["domain"] == known[code]["domain"] for item in values) for code, values in graph.items())


def test_three_fold_assignment_is_balanced_and_bridge_safe() -> None:
    known = load_known_classes(ROOT / "datasets/AgriNet-1K/open_agri_v3")
    graph, _ = build_known_rag_graph(
        known, ROOT / "outputs/milvus/wiki_similar_classes_siglip2_report.top12.json"
    )
    assignment, report = solve_assignment(known, graph, folds=3, seed="test-e3", attempts=2000)
    assert assignment is not None
    assert report["valid_candidates"] > 0
    assert assignment_violations(assignment, graph) == []
    assert sorted(sum(value == fold for value in assignment.values()) for fold in range(3)) == [35, 36, 36]
    for fold in range(3):
        assert sum(value == fold and known[code]["domain"] == "disease" for code, value in assignment.items()) == 24


def test_candidate_assignment_is_domain_balanced() -> None:
    known = {
        **{f"D{i}": {"domain": "disease", "train_candidate_images": 1} for i in range(7)},
        **{f"P{i}": {"domain": "pest", "train_candidate_images": 1} for i in range(5)},
    }
    assignment = candidate_assignment(known, folds=3, seed="test", attempt=0)
    assert [sum(value == fold and known[code]["domain"] == "disease" for code, value in assignment.items()) for fold in range(3)] == [3, 2, 2]
    assert [sum(value == fold and known[code]["domain"] == "pest" for code, value in assignment.items()) for fold in range(3)] == [2, 2, 1]
