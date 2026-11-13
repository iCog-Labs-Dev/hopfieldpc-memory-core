from __future__ import annotations

import itertools
import math
from pathlib import Path
import pytest
import torch

from benchmarks.mnist_prototype_retrieval import covering_radius, k_center_greedy


def test_k_center_greedy_toy() -> None:
    # Proves that on a hand-computed 1D toy instance, greedy k-center selects indices [0, 6, 5] with covering radius 7.
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [8.0], [15.0], [30.0]])
    gen = torch.Generator().manual_seed(0)
    idx = k_center_greedy(x, 3, generator=gen, first=0)
    assert idx.tolist() == [0, 6, 5]
    assert covering_radius(x, x[idx]) == 7.0


def test_k_center_greedy_distinct_and_real_rows() -> None:
    # Proves that picked indices are strictly distinct and index into real rows of x.
    x = torch.randn(25, 4)
    gen = torch.Generator().manual_seed(42)
    idx = k_center_greedy(x, 10, generator=gen)
    assert len(idx) == 10
    assert len(set(idx.tolist())) == 10
    assert idx.dtype == torch.long
    for i in idx.tolist():
        assert 0 <= i < len(x)
        assert any(torch.equal(x[i], row) for row in x)


def test_k_center_greedy_step_by_step_farthest() -> None:
    # Proves that each pick maximizes the exact distance to the set of centers chosen so far.
    x = torch.randn(30, 5)
    gen = torch.Generator().manual_seed(123)
    idx = k_center_greedy(x, 7, generator=gen, first=2)
    for t in range(1, 7):
        current_centers = x[idx[:t]]
        dists = torch.cdist(x, current_centers).min(dim=1).values
        unpicked_mask = torch.ones(len(x), dtype=torch.bool)
        unpicked_mask[idx[:t]] = False
        max_dist = dists[unpicked_mask].max()
        assert torch.isclose(dists[idx[t]], max_dist)


def test_k_center_greedy_prefix_and_monotonic_radius() -> None:
    # Proves that choices share a common prefix for fixed seed/first and covering radius is non-increasing.
    x = torch.randn(20, 3)
    gen1 = torch.Generator().manual_seed(99)
    gen2 = torch.Generator().manual_seed(99)
    idx_k5 = k_center_greedy(x, 5, generator=gen1, first=1)
    idx_k8 = k_center_greedy(x, 8, generator=gen2, first=1)
    assert idx_k8[:5].tolist() == idx_k5.tolist()
    radii = [covering_radius(x, x[idx_k8[:k]]) for k in range(1, 9)]
    for prev, cur in zip(radii[:-1], radii[1:]):
        assert cur <= prev + 1e-6


def test_k_center_greedy_2_approximation_brute_force() -> None:
    # Proves that the greedy covering radius is within 2*OPT across 20 synthetic 2D instances.
    gen = torch.Generator().manual_seed(2026)
    n, d, k = 12, 2, 3
    for _ in range(20):
        x = torch.randn(n, d, generator=gen)
        combos = list(itertools.combinations(range(n), k))
        opt_radius = min(covering_radius(x, x[list(c)]) for c in combos)
        greedy_idx = k_center_greedy(x, k, generator=gen)
        greedy_radius = covering_radius(x, x[greedy_idx])
        assert greedy_radius <= 2.0 * opt_radius + 1e-6


def test_k_center_greedy_determinism_and_first_override() -> None:
    # Proves that identical seeds produce identical picks and first= overrides random initialization.
    x = torch.randn(15, 3)
    idx1 = k_center_greedy(x, 4, generator=torch.Generator().manual_seed(7))
    idx2 = k_center_greedy(x, 4, generator=torch.Generator().manual_seed(7))
    assert torch.equal(idx1, idx2)
    idx3 = k_center_greedy(x, 4, generator=torch.Generator().manual_seed(7), first=9)
    assert idx3[0].item() == 9


def test_k_center_greedy_k_equals_n() -> None:
    # Proves that k == N selects all rows and yields zero covering radius.
    x = torch.randn(10, 4)
    idx = k_center_greedy(x, 10, generator=torch.Generator().manual_seed(0))
    assert len(set(idx.tolist())) == 10
    assert covering_radius(x, x[idx]) == 0.0


def test_k_center_greedy_duplicate_rows() -> None:
    # Proves that duplicate rows still result in distinct selected indices without error.
    x = torch.tensor([[1.0, 2.0], [1.0, 2.0], [5.0, 6.0], [5.0, 6.0]])
    idx = k_center_greedy(x, 4, generator=torch.Generator().manual_seed(0), first=0)
    assert len(set(idx.tolist())) == 4


def test_k_center_greedy_invalid_args() -> None:
    # Proves that invalid dimensions, k out of bounds, or invalid first argument raise ValueError.
    x = torch.randn(10, 3)
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="2-D"):
        k_center_greedy(torch.randn(10), 3, generator=gen)
    with pytest.raises(ValueError, match="2-D"):
        k_center_greedy(torch.randn(10, 3, 2), 3, generator=gen)
    with pytest.raises(ValueError, match="integer in"):
        k_center_greedy(x, 0, generator=gen)
    with pytest.raises(ValueError, match="integer in"):
        k_center_greedy(x, 11, generator=gen)
    with pytest.raises(ValueError, match="integer in"):
        k_center_greedy(x, True, generator=gen)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="first must be an integer"):
        k_center_greedy(x, 3, generator=gen, first=-1)
    with pytest.raises(ValueError, match="first must be an integer"):
        k_center_greedy(x, 3, generator=gen, first=10)
    with pytest.raises(ValueError, match="first must be an integer"):
        k_center_greedy(x, 3, generator=gen, first=False)  # type: ignore[arg-type]


def test_k_center_greedy_input_not_modified() -> None:
    # Proves that k_center_greedy does not modify the input tensor x in-place.
    x = torch.randn(15, 3)
    clone = x.clone()
    _ = k_center_greedy(x, 5, generator=torch.Generator().manual_seed(42))
    assert torch.equal(x, clone)


def test_build_memory_kcenter_shape_labels_center() -> None:
    # Proves that build_memory(method="kcenter") produces expected shape, sorted labels, and center.
    from benchmarks.mnist_prototype_retrieval import build_memory
    z_train = torch.randn(100, 8)
    labels = torch.repeat_interleave(torch.arange(10), 10)
    mem, mem_labels, center = build_memory(z_train, labels, per_class=3, center=True, seed=0, method="kcenter")
    assert mem.shape == (30, 8)
    assert mem_labels.tolist() == sorted(list(range(10)) * 3)
    assert center is not None
    assert torch.allclose(center, z_train.mean(dim=0))


def test_build_memory_kcenter_real_rows_matching_class() -> None:
    # Proves that every row in kcenter memory equals some z_train row of the same class.
    from benchmarks.mnist_prototype_retrieval import build_memory
    z_train = torch.randn(100, 8)
    labels = torch.repeat_interleave(torch.arange(10), 10)
    mem, mem_labels, _ = build_memory(z_train, labels, per_class=3, center=True, seed=0, method="kcenter")
    for i in range(len(mem)):
        c = mem_labels[i].item()
        zc = z_train[labels == c]
        assert any(torch.equal(mem[i], row) for row in zc)


def test_build_memory_kcenter_deterministic() -> None:
    # Proves that same seed produces identical kcenter memory.
    from benchmarks.mnist_prototype_retrieval import build_memory
    z_train = torch.randn(100, 8)
    labels = torch.repeat_interleave(torch.arange(10), 10)
    m1, l1, c1 = build_memory(z_train, labels, per_class=3, center=True, seed=42, method="kcenter")
    m2, l2, c2 = build_memory(z_train, labels, per_class=3, center=True, seed=42, method="kcenter")
    assert torch.equal(m1, m2)
    assert torch.equal(l1, l2)
    assert center_equal(c1, c2)


def center_equal(c1: torch.Tensor | None, c2: torch.Tensor | None) -> bool:
    if c1 is None or c2 is None:
        return c1 is c2
    return bool(torch.equal(c1, c2))


def test_kcenter_vs_kmeans_covering_radius_outliers() -> None:
    # Proves that on dense blob + far points, per-class covering radius of kcenter <= kmeans.
    from benchmarks.mnist_prototype_retrieval import build_memory
    torch.manual_seed(0)
    blocks = []
    labels_list = []
    for c in range(10):
        # Two dense sub-clusters with 250 points each at -5 and +5, plus two outliers at -50 and +50
        c1 = torch.randn(250, 2) * 0.1 + torch.tensor([-5.0, 0.0]) + c * 200.0
        c2 = torch.randn(250, 2) * 0.1 + torch.tensor([5.0, 0.0]) + c * 200.0
        o1 = torch.tensor([[-50.0, 0.0]]) + c * 200.0
        o2 = torch.tensor([[50.0, 0.0]]) + c * 200.0
        zc = torch.cat([c1, c2, o1, o2])
        blocks.append(zc)
        labels_list.append(torch.full((len(zc),), c, dtype=torch.long))
    z_train = torch.cat(blocks)
    labels = torch.cat(labels_list)

    mem_km, l_km, _ = build_memory(z_train, labels, per_class=3, center=False, seed=0, method="kmeans")
    mem_kc, l_kc, _ = build_memory(z_train, labels, per_class=3, center=False, seed=0, method="kcenter")

    for c in range(10):
        zc = z_train[labels == c]
        r_km = covering_radius(zc, mem_km[l_km == c])
        r_kc = covering_radius(zc, mem_kc[l_kc == c])
        assert r_kc <= r_km + 1e-4


def test_build_memory_per_class_1_and_unknown_method() -> None:
    # Proves that per_class == 1 yields class mean for both methods, and unknown method raises.
    from benchmarks.mnist_prototype_retrieval import build_memory
    z_train = torch.randn(50, 6)
    labels = torch.repeat_interleave(torch.arange(5), 10)
    m_km, _, _ = build_memory(z_train, labels, per_class=1, center=False, seed=0, method="kmeans", num_classes=5)
    m_kc, _, _ = build_memory(z_train, labels, per_class=1, center=False, seed=0, method="kcenter", num_classes=5)
    for c in range(5):
        mean_c = z_train[labels == c].mean(dim=0, keepdim=True)
        assert torch.allclose(m_km[c:c+1], mean_c)
        assert torch.allclose(m_kc[c:c+1], mean_c)
    with pytest.raises(ValueError, match="Unknown memory method"):
        build_memory(z_train, labels, per_class=2, center=False, seed=0, method="invalid")


def test_build_memory_default_method_is_kmeans() -> None:
    # Proves that default method in build_memory is bit-identical to method="kmeans".
    from benchmarks.mnist_prototype_retrieval import build_memory
    z_train = torch.randn(80, 8)
    labels = torch.repeat_interleave(torch.arange(10), 8)
    m_def, l_def, c_def = build_memory(z_train, labels, per_class=3, center=True, seed=123)
    m_km, l_km, c_km = build_memory(z_train, labels, per_class=3, center=True, seed=123, method="kmeans")
    assert torch.equal(m_def, m_km)
    assert torch.equal(l_def, l_km)
    assert center_equal(c_def, c_km)


def test_config_memory_method_and_yaml_equality() -> None:
    # Proves that config validates memory_method and the two YAML files match modulo reports_dir and memory_method.
    import yaml
    from benchmarks.mnist_baseline import validate_mnist_config

    cfg_def = validate_mnist_config({})
    assert cfg_def["memory_method"] == "kmeans"

    cfg_kc = validate_mnist_config({"memory_method": "kcenter"})
    assert cfg_kc["memory_method"] == "kcenter"

    with pytest.raises(ValueError, match="memory_method must be one of"):
        validate_mnist_config({"memory_method": "bogus"})

    with open("configs/mnist_prototype_retrieval.yaml", "r", encoding="utf-8") as f:
        y1 = yaml.safe_load(f)
    with open("configs/mnist_kcenter.yaml", "r", encoding="utf-8") as f:
        y2 = yaml.safe_load(f)

    del y1["reports_dir"]
    del y2["reports_dir"]
    y1.get("quick", {}).pop("reports_dir", None)
    y2.get("quick", {}).pop("reports_dir", None)
    y2.pop("memory_method", None)
    y2.pop("backbone_dir", None)
    y2.get("quick", {}).pop("backbone_dir", None)
    assert y1 == y2


def test_mean_quantization_error_toy() -> None:
    # Proves that mean quantization error correctly averages distance to nearest center.
    from benchmarks.kcenter_analysis import mean_quantization_error
    x = torch.tensor([[0.0], [2.0], [10.0]])
    centers = torch.tensor([[0.0], [10.0]])
    mqe = mean_quantization_error(x, centers)
    assert math.isclose(mqe, 2.0 / 3.0, rel_tol=1e-5)


def test_coverage_fraction_toy() -> None:
    # Proves that coverage fraction correctly computes fraction within radius r.
    from benchmarks.kcenter_analysis import coverage_fraction
    x = torch.tensor([[0.0], [2.0], [10.0]])
    centers = torch.tensor([[0.0], [10.0]])
    assert math.isclose(coverage_fraction(x, centers, 1.0), 2.0 / 3.0, rel_tol=1e-5)
    assert math.isclose(coverage_fraction(x, centers, 2.0), 1.0, rel_tol=1e-5)
    assert math.isclose(coverage_fraction(x, centers, 0.0), 2.0 / 3.0, rel_tol=1e-5)


def test_cell_balance_entropy_toy() -> None:
    # Proves that cell balance entropy correctly computes Shannon entropy of Voronoi partitions.
    import math
    from benchmarks.kcenter_analysis import cell_balance_entropy
    x = torch.tensor([[0.0], [0.1], [10.0], [10.1]])
    centers = torch.tensor([[0.0], [10.0]])
    entropy = cell_balance_entropy(x, centers)
    assert math.isclose(entropy, math.log(2.0), rel_tol=1e-4)


def test_pairwise_distances_and_margin() -> None:
    # Proves that pairwise distance and inter-class margin compute correct statistics.
    from benchmarks.kcenter_analysis import inter_class_margin, pairwise_distances
    centers = torch.tensor([[0.0], [5.0], [15.0]])
    mean_d, min_d = pairwise_distances(centers)
    assert math.isclose(mean_d, 10.0, rel_tol=1e-5)
    assert math.isclose(min_d, 5.0, rel_tol=1e-5)

    mem = torch.tensor([[0.0], [1.0], [10.0], [11.0]])
    labels = torch.tensor([0, 0, 1, 1])
    margin = inter_class_margin(mem, labels)
    assert math.isclose(margin, 9.5, rel_tol=1e-5)


def test_extract_beta_star_all_structures() -> None:
    # Proves that extract_beta_star handles floats, ints, dicts, and raises ValueError for missing/empty/invalid values.
    from benchmarks.kcenter_analysis import extract_beta_star
    assert extract_beta_star(50.0) == 50.0
    assert extract_beta_star(10) == 10.0
    assert extract_beta_star("25.5") == 25.5
    assert extract_beta_star({"embedding": 50.0}) == 50.0
    assert extract_beta_star({"pixel": 5.0, "embedding": 20.0}) == 20.0
    assert extract_beta_star({"custom": 15.0}) == 15.0

    with pytest.raises(ValueError, match="missing"):
        extract_beta_star(None)
    with pytest.raises(ValueError, match="empty"):
        extract_beta_star({})
    with pytest.raises(ValueError, match="no non-null values"):
        extract_beta_star({"embedding": None})
    with pytest.raises(ValueError, match="Cannot extract beta"):
        extract_beta_star("invalid")


def test_run_full_analysis_rejects_mismatched_configs(tmp_path: Path) -> None:
    # Proves that run_full_analysis rejects runs with mismatched seeds or grids before pairing.
    import json
    from benchmarks.kcenter_analysis import run_full_analysis

    dir_km = tmp_path / "km"
    dir_kc = tmp_path / "kc"
    dir_km.mkdir()
    dir_kc.mkdir()

    base_res = {
        "config": {
            "seeds": [0, 1],
            "gaussian_sigmas": [0.0, 0.5],
            "salt_pepper_probs": [0.0, 0.2],
            "occlusion_sizes": [0, 7],
            "beta_values": [1.0, 10.0],
            "num_updates_values": [1],
            "beta_selection_sets": [["clean", 0]],
            "prototypes_per_class": 8,
            "center_prototypes": True,
            "eval_batch_size": 1000,
            "noisy_sigma": 0.5,
            "placements": ["embedding"],
            "val_size": 100,
            "train_subset": 200,
            "test_subset": 100,
        },
        "results": {
            "E1": [{"value": "clean", "baseline": [{"head_acc": 0.95}, {"head_acc": 0.96}], "hopfield": [{"head_acc": 0.94, "proto_acc": 0.94}, {"head_acc": 0.95, "proto_acc": 0.95}]}],
            "E3a": [{"value": 0.0, "baseline": [{"head_acc": 0.95}, {"head_acc": 0.96}], "hopfield": [{"head_acc": 0.94}, {"head_acc": 0.95}]}],
            "E3b": [{"value": 0.0, "baseline": [{"head_acc": 0.95}, {"head_acc": 0.96}], "hopfield": [{"head_acc": 0.94}, {"head_acc": 0.95}]}],
            "E3c": [{"value": 0, "baseline": [{"head_acc": 0.95}, {"head_acc": 0.96}], "hopfield": [{"head_acc": 0.94}, {"head_acc": 0.95}]}],
            "E2": [{"value": 1.0, "baseline": [{"head_acc": 0.95}, {"head_acc": 0.96}], "hopfield": [{"head_acc": 0.94}, {"head_acc": 0.95}]}],
            "E4": [{"value": 1, "baseline": [{"head_acc": 0.95}, {"head_acc": 0.96}], "hopfield": [{"head_acc": 0.94}, {"head_acc": 0.95}]}],
            "E5": [{"value": 1.0, "baseline": [{"head_acc": 0.95}, {"head_acc": 0.96}], "hopfield": [{"head_acc": 0.94}, {"head_acc": 0.95}]}],
        },
        "beta_star": 10.0,
        "cost": {"e8": {}},
    }

    # Write valid KM
    (dir_km / "results.json").write_text(json.dumps(base_res), encoding="utf-8")

    # Mismatched seeds in KC
    mismatched_seeds = json.loads(json.dumps(base_res))
    mismatched_seeds["config"]["seeds"] = [0, 2]
    (dir_kc / "results.json").write_text(json.dumps(mismatched_seeds), encoding="utf-8")

    with pytest.raises(ValueError, match="Mismatched seeds"):
        run_full_analysis(str(dir_km), str(dir_kc), output_dir=str(tmp_path / "out"))

    # Mismatched grids in KC
    mismatched_grid = json.loads(json.dumps(base_res))
    mismatched_grid["config"]["gaussian_sigmas"] = [0.0, 0.8]
    (dir_kc / "results.json").write_text(json.dumps(mismatched_grid), encoding="utf-8")

    with pytest.raises(ValueError, match="Mismatched configuration for 'gaussian_sigmas'"):
        run_full_analysis(str(dir_km), str(dir_kc), output_dir=str(tmp_path / "out"))

    # Baseline mismatch in E1
    mismatched_baseline = json.loads(json.dumps(base_res))
    mismatched_baseline["results"]["E1"][0]["baseline"][0]["head_acc"] = 0.50
    (dir_kc / "results.json").write_text(json.dumps(mismatched_baseline), encoding="utf-8")

    with pytest.raises(ValueError, match="Fairness violation"):
        run_full_analysis(str(dir_km), str(dir_kc), output_dir=str(tmp_path / "out"))
