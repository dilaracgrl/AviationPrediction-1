"""
Minimal regression tests for the AviationPrediction pipeline.
Requires data/processed/ artifacts (run notebooks 01, 03, 04 first).

Run with: pytest tests/test_pipeline.py -v
(or from project root: python -m pytest tests/test_pipeline.py -v)
"""

from pathlib import Path

import pandas as pd
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"


def _model_dataset_path():
    import pytest
    p = PROCESSED / "model_dataset.parquet"
    if not p.exists():
        pytest.skip("data/processed/model_dataset.parquet not found (run notebook 01 first)")
    return p


def _preprocessor_path():
    import pytest
    p = PROCESSED / "preprocessor.joblib"
    if not p.exists():
        pytest.skip("data/processed/preprocessor.joblib not found (run notebook 03 first)")
    return p


def _best_model_path():
    import pytest
    p = PROCESSED / "best_model.joblib"
    if not p.exists():
        pytest.skip("data/processed/best_model.joblib not found (run notebook 04 first)")
    return p


def test_model_dataset_no_leakage_columns():
    """Model dataset must not contain post-departure leakage columns."""
    path = _model_dataset_path()
    df = pd.read_parquet(path)
    leakage = ["DepDelay", "ArrDelay", "DepDel15"]
    leaked = [c for c in leakage if c in df.columns]
    assert len(leaked) == 0, f"LEAKAGE: found columns {leaked} in model dataset"


def test_model_dataset_has_target_and_shape():
    """Model dataset must have ArrDel15 and a reasonable number of columns."""
    path = _model_dataset_path()
    df = pd.read_parquet(path)
    assert "ArrDel15" in df.columns
    assert df.shape[1] >= 10
    assert df.shape[0] >= 1000


def test_preprocessor_output_shape():
    """Preprocessor loads and saved train_X has expected number of features (59)."""
    _preprocessor_path()
    train_x_path = PROCESSED / "train_X.parquet"
    if not train_x_path.exists():
        import pytest
        pytest.skip("train_X.parquet not found (run notebook 03 first)")
    pp = joblib.load(PROCESSED / "preprocessor.joblib")
    X = pd.read_parquet(train_x_path)
    assert X.shape[1] == 59, f"Expected 59 features in train_X, got {X.shape[1]}"
    # Preprocessor output dimension matches (get_feature_names_out exists in sklearn 1.0+)
    if hasattr(pp, "get_feature_names_out"):
        out_names = pp.get_feature_names_out()
        assert len(out_names) == 59, f"Preprocessor outputs {len(out_names)} features, expected 59"


def test_best_model_load_and_predict():
    """Best model bundle must load and predict_proba returns correct shape."""
    bundle_path = _best_model_path()
    test_x_path = PROCESSED / "test_X.parquet"
    if not test_x_path.exists():
        import pytest
        pytest.skip("test_X.parquet not found (run notebook 03 first)")
    bundle = joblib.load(bundle_path)
    assert "model" in bundle
    assert "threshold" in bundle
    X = pd.read_parquet(test_x_path)
    if hasattr(X, "values"):
        X = X.values.astype(np.float32)
    else:
        X = X.astype(np.float32)
    probs = bundle["model"].predict_proba(X[:5])
    assert probs.shape[0] == 5
    assert probs.shape[1] == 2
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_train_val_test_splits_exist_and_align():
    """Train/val/test X and y parquets must exist and have matching row counts."""
    for name in ["train", "val", "test"]:
        x_path = PROCESSED / f"{name}_X.parquet"
        y_path = PROCESSED / f"{name}_y.parquet"
        if not x_path.exists() or not y_path.exists():
            import pytest
            pytest.skip(f"{name}_X or {name}_y not found (run notebook 03 first)")
        X = pd.read_parquet(x_path)
        y = pd.read_parquet(y_path)
        assert len(X) == len(y), f"{name}: X rows {len(X)} != y rows {len(y)}"
        if name == "train":
            assert "ArrDel15" in y.columns or y.shape[1] == 1


if __name__ == "__main__":
    try:
        import pytest
    except ImportError:
        print("pytest is not installed in this Python.")
        print("Install it: pip install pytest")
        print("Then run tests with: pytest tests/test_pipeline.py -v")
        exit(1)
    exit(pytest.main([__file__, "-v"]))
