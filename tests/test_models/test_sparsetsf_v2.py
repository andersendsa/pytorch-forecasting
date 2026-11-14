import pytest
import torch

from pytorch_forecasting.models import SparseTSF, SparseTSF_pkg_v2

METADATA = {"max_encoder_length": 8, "max_prediction_length": 4}


@pytest.mark.parametrize("model_type", ["linear", "mlp"])
def test_reference_computation_and_gradients(model_type):
    torch.manual_seed(1)
    model = SparseTSF(period_len=2, model_type=model_type, d_model=3, metadata=METADATA)
    history = torch.randn(2, 8, 3, requires_grad=True)
    prediction = model({"target_past": history})["prediction"]

    mean = history.mean(1, keepdim=True)
    values = (history - mean).permute(0, 2, 1)
    values = model.conv1d(values.reshape(-1, 1, 8)).reshape(2, 3, 8) + values
    values = values.reshape(-1, 4, 2).permute(0, 2, 1)
    expected = model.predictor(values).permute(0, 2, 1).reshape(2, 3, 4)
    expected = expected.permute(0, 2, 1) + mean
    torch.testing.assert_close(prediction, expected)
    prediction.sum().backward()
    assert history.grad is not None
    assert all(p.grad is not None for p in model.parameters())


def test_univariate_and_channel_independence():
    model = SparseTSF(period_len=2, metadata=METADATA)
    history = torch.randn(2, 8, 2)
    prediction = model({"target_past": history})["prediction"]
    assert prediction.shape == (2, 4, 2)
    single = model({"target_past": history[:, :, 0]})["prediction"]
    torch.testing.assert_close(single, prediction[:, :, :1])
    assert SparseTSF.pkg is SparseTSF_pkg_v2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"period_len": 0},
        {"period_len": 3},
        {"model_type": "invalid"},
        {"d_model": 0},
        {"metadata": {**METADATA, "min_encoder_length": 4}},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SparseTSF(**({"metadata": METADATA} | kwargs))
