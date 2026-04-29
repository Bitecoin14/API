# tests/test_asl_model.py
"""Tests for the ASL model and ASLStage fallback chain."""
import sys
from pathlib import Path

import numpy as np
import pytest

MODEL_PATH = Path("models/asl_classifier.pkl")


def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Run models/train_asl.py first — {MODEL_PATH} not found"


def test_model_loads():
    import joblib
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")


def test_model_predict_returns_letter():
    import joblib
    model = joblib.load(MODEL_PATH)
    features = np.zeros((1, 63), dtype=np.float32)
    pred = model.predict(features)
    assert len(pred) == 1
    assert pred[0] in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_model_predicts_all_letters():
    """Each letter in the training set can be predicted by some input."""
    import joblib
    from models.train_asl import generate_dataset
    model = joblib.load(MODEL_PATH)
    X, y = generate_dataset(n_per_letter=50)
    preds = model.predict(X)
    predicted_set = set(preds)
    letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # At least 20 of 26 letters should appear in predictions
    assert len(predicted_set & letters) >= 20, f"Only predicted: {predicted_set}"


def test_asl_stage_fallback_no_crash(tmp_path):
    """ASLStage initialises without error when model file is missing."""
    from stages.asl_stage import ASLStage
    stage = ASLStage(model_path=tmp_path / "missing.pkl")
    assert stage is not None


def test_normalize_landmarks_shape():
    """_normalize_landmarks returns 63-element vector."""
    from stages.asl_stage import _normalize_landmarks
    from types import SimpleNamespace

    # Mock a hand_landmarks object with 21 landmarks
    lm_list = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(21)]
    lm_list[9] = SimpleNamespace(x=0.0, y=-1.0, z=0.0)   # middle MCP for scale
    mock_hand = SimpleNamespace(landmark=lm_list)
    result = _normalize_landmarks(mock_hand)
    assert result.shape == (63,)
