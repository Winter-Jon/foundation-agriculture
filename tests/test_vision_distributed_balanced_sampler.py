from collections import Counter

from agrinet.vision.workflow import DistributedBalancedSampler
from agrinet.vision.workflow import _ARCHITECTURES


def test_distributed_balanced_sampler_is_deterministic_and_globally_balanced() -> None:
    rows = [{"label": 0}] * 2 + [{"label": 1}] * 4 + [{"label": 2}] * 3
    first = DistributedBalancedSampler(rows, rank=0, world_size=2, seed=17)
    second = DistributedBalancedSampler(rows, rank=1, world_size=2, seed=17)
    global_indices = first._global_indices()
    assert len(global_indices) == 10
    assert Counter(rows[index]["label"] for index in global_indices).values()
    counts = Counter(rows[index]["label"] for index in global_indices)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert list(first) == global_indices[0::2]
    assert list(second) == global_indices[1::2]
    repeat = DistributedBalancedSampler(rows, rank=0, world_size=2, seed=17)
    assert list(repeat) == list(first)


def test_distributed_balanced_sampler_rotates_remainder_classes() -> None:
    rows = [{"label": label} for label in range(3) for _ in range(2)] + [{"label": 0}]
    sampler = DistributedBalancedSampler(rows, rank=0, world_size=2)
    first = Counter(rows[index]["label"] for index in sampler._global_indices())
    sampler.set_epoch(1)
    second = Counter(rows[index]["label"] for index in sampler._global_indices())
    assert first == {0: 3, 1: 3, 2: 2}
    assert second == {0: 2, 1: 3, 2: 3}


def test_vith_architecture_maps_to_timm_huge_patch14() -> None:
    assert _ARCHITECTURES["mae_vit_huge_patch14_224"] == "vit_huge_patch14_224"
