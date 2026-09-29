"""
Smoke tests for the astronaut health monitoring pipeline.

Fast, no network, no GPU.

    pytest tests/ -q
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets  # noqa: E402
import run  # noqa: E402

MODELS = list(run.MODEL_KEYS)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


def test_synthetic_corpus_loads_with_expected_shape():
    window_set = datasets.load("synthetic_crew_gait", split="test", seed=7)
    assert len(window_set) > 0
    assert window_set.x.ndim == 3
    assert window_set.x.shape[1] == 6  # 6 IMU channels
    assert window_set.x.shape[2] == 500  # 10s @ 50Hz
    assert set(np.unique(window_set.y)) <= {0, 1}
    assert window_set.label_source == "provided"


def test_splits_are_subject_disjoint():
    """No crew member may appear in more than one split -- the whole point."""
    subjects = {
        split: set(np.unique(datasets.load("synthetic_crew_gait", split=split, seed=7).subject_ids))
        for split in ("train", "val", "test")
    }
    assert not subjects["train"] & subjects["test"]
    assert not subjects["train"] & subjects["val"]
    assert not subjects["val"] & subjects["test"]


def test_missing_corpus_raises_actionable_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError) as caught:
        datasets.load("osdr_human_spaceflight", split="test", data_root=tmp_path)
    message = str(caught.value)
    assert "DATASETS.md" in message


def test_longitudinal_panel_without_label_is_circular_and_flagged(tmp_path):
    root = tmp_path / "toy_panel"
    root.mkdir(parents=True)
    rows = []
    for subject in range(6):
        for timepoint, phase in (("L-3", "preflight"), ("FD1", "inflight"), ("R+1", "postflight")):
            rows.append({"subject_id": f"S{subject}", "timepoint": timepoint, "phase": phase,
                        "heart_rate": 70 + subject + (5 if phase != "preflight" else 0)})
    pd.DataFrame(rows).to_csv(root / "panel.csv", index=False)

    with pytest.warns(UserWarning, match="CIRCULAR"):
        window_set = datasets.load("toy_panel", split="test", data_root=tmp_path)
    assert window_set.label_source.startswith("derived")
    assert "validity_warning" in window_set.statistics()


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------


@pytest.mark.parametrize("model_key", MODELS)
def test_models_fit_and_score(model_key):
    train_set = datasets.load("synthetic_crew_gait", split="train", seed=7)
    test_set = datasets.load("synthetic_crew_gait", split="test", seed=7)

    model = run.build_model(model_key, seed=7)
    model.fit(train_set.x, train_set.y, train_set.subject_ids)
    scores = model.predict_proba(test_set.x)

    assert scores.shape == (len(test_set),)
    assert np.all((scores >= 0) & (scores <= 1))
    assert model.n_parameters >= 1, f"{model_key} reported no parameters"


def test_numpy_mlp_is_deterministic_and_needs_no_torch():
    assert "torch" not in sys.modules
    train_set = datasets.load("synthetic_crew_gait", split="train", seed=7)
    test_set = datasets.load("synthetic_crew_gait", split="test", seed=7)

    scores = []
    for _ in range(2):
        model = run.NumpyMLPMonitor(sampling_rate_hz=50.0, seed=7, epochs=20)
        model.fit(train_set.x, train_set.y, train_set.subject_ids)
        scores.append(model.predict_proba(test_set.x))

    assert np.allclose(scores[0], scores[1]), "same seed must give identical scores"
    assert "torch" not in sys.modules


def test_numpy_mlp_beats_chance_on_the_simulator():
    train_set = datasets.load("synthetic_crew_gait", split="train", seed=7)
    test_set = datasets.load("synthetic_crew_gait", split="test", seed=7)

    model = run.build_model("descriptor_mlp", seed=7)
    model.fit(train_set.x, train_set.y, train_set.subject_ids)
    score = run.compute_metrics(test_set.y, model.predict_proba(test_set.x))[run.PRIMARY_METRIC]
    assert score > 0.6, f"MLP scored {score:.3f} on the simulator -- likely a training bug"


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def test_perfect_and_inverted_predictions_bracket_the_metric():
    y_true = np.array([0, 0, 1, 1])
    assert run.compute_metrics(y_true, np.array([0.0, 0.1, 0.9, 1.0]))[run.PRIMARY_METRIC] == 1.0
    assert run.compute_metrics(y_true, np.array([1.0, 0.9, 0.1, 0.0]))[run.PRIMARY_METRIC] == 0.0


def test_constant_predictor_scores_half_not_high_on_imbalanced_data():
    """A monitor that never alerts must not look good -- why balanced accuracy is primary."""
    y_true = np.array([0] * 90 + [1] * 10)
    metrics = run.compute_metrics(y_true, np.zeros(100))
    assert metrics["accuracy"] == 0.90
    assert metrics[run.PRIMARY_METRIC] == 0.50


def test_per_subject_breakdown_is_emitted():
    y_true = np.array([0, 1, 0, 1])
    metrics = run.compute_metrics(y_true, np.array([0.1, 0.9, 0.2, 0.8]), subject_ids=np.array([1, 1, 2, 2]))
    assert set(metrics["per_subject"]) == {"1", "2"}


def test_single_class_split_does_not_crash():
    metrics = run.compute_metrics(np.zeros(10), np.random.rand(10))
    assert np.isnan(metrics["roc_auc"])


# ----------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------


def test_evaluate_completes_a_cell():
    row = run.evaluate("gait_logreg", "synthetic_crew_gait", seed=7)
    assert row["status"] == "completed"
    assert 0.0 <= row[run.PRIMARY_METRIC] <= 1.0
    assert row["n_parameters"] > 0


def test_main_reports_missing_data_without_raising(tmp_path, capsys):
    exit_code = run.main([
        "--dataset", "osdr_human_spaceflight", "--models", "gait_logreg",
        "--data-root", str(tmp_path), "--results-csv", str(tmp_path / "results.csv"),
    ])
    assert exit_code == 1
    frame = pd.read_csv(tmp_path / "results.csv")
    assert frame.loc[0, "status"] == "missing_data"
