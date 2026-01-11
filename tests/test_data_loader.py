from src.data_loader import load_model_features


def test_load_model_features_non_empty():
    df = load_model_features()
    assert len(df) > 0


def test_model_features_has_targets():
    df = load_model_features()
    assert "XLE_target" in df.columns
    assert "ICLN_target" in df.columns
