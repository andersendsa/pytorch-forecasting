"""SparseTSF model for time series forecasting."""

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class SparseTSF(BaseModel):
    """Cross-period sparse forecasting with channel-independent weights.

    Based on `SparseTSF <https://arxiv.org/abs/2405.00946>`_ and its
    `reference implementation <https://github.com/lss-1138/SparseTSF>`_.
    Uses past targets only, ignoring covariates. Produces point forecasts.

    Parameters
    ----------
    period_len : int, default=1
        Data period. Both context and prediction lengths must be divisible by it.
    model_type : str, default="linear"
        Cross-period predictor: "linear" or "mlp".
    d_model : int, default=128
        Hidden size of the MLP predictor.
    loss : nn.Module, optional
        Point forecast loss; defaults to MAE.
    logging_metrics : list of nn.Module, optional
        Additional metrics to log.
    optimizer : Optimizer or str, default="adam"
        Optimizer for training.
    optimizer_params : dict, optional
        Optimizer parameters.
    lr_scheduler : str, optional
        Learning rate scheduler.
    lr_scheduler_params : dict, optional
        Scheduler parameters.
    metadata : dict
        Metadata from EncoderDecoderTimeSeriesDataModule, with fixed history length.
    """

    @classmethod
    def _pkg(cls):
        from pytorch_forecasting.models.sparsetsf._sparsetsf_pkg_v2 import (
            SparseTSF_pkg_v2,
        )

        return SparseTSF_pkg_v2

    def __init__(
        self,
        period_len: int = 1,
        model_type: str = "linear",
        d_model: int = 128,
        loss: nn.Module | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
    ):
        if metadata is None:
            raise ValueError("metadata is required")
        context_length = metadata["max_encoder_length"]
        prediction_length = metadata["max_prediction_length"]
        if (
            not isinstance(period_len, int)
            or isinstance(period_len, bool)
            or period_len < 1
        ):
            raise ValueError("period_len must be a positive integer")
        if (
            context_length < 1
            or prediction_length < 1
            or context_length % period_len
            or prediction_length % period_len
        ):
            raise ValueError(
                "context and prediction lengths must be positive "
                "multiples of period_len"
            )
        if metadata.get("min_encoder_length", context_length) != context_length:
            raise ValueError("SparseTSF requires a fixed encoder length")
        if model_type not in ("linear", "mlp"):
            raise ValueError("model_type must be 'linear' or 'mlp'")
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model < 1:
            raise ValueError("d_model must be a positive integer")

        super().__init__(
            loss=loss if loss is not None else MAE(),
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self.metadata = metadata
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.period_len = period_len
        self.conv1d = nn.Conv1d(
            1, 1, 1 + 2 * (period_len // 2), padding=period_len // 2, bias=False
        )
        input_periods = context_length // period_len
        output_periods = prediction_length // period_len
        if model_type == "linear":
            self.predictor = nn.Linear(input_periods, output_periods, bias=False)
        else:
            self.predictor = nn.Sequential(
                nn.Linear(input_periods, d_model),
                nn.ReLU(),
                nn.Linear(d_model, output_periods),
            )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forecast each target channel from ``target_past`` independently."""
        history = x["target_past"]
        if history.ndim == 2:
            history = history.unsqueeze(-1)
        batch, length, channels = history.shape
        if length != self.context_length:
            raise ValueError(
                "target_past length must equal the configured context length"
            )
        mean = history.mean(dim=1, keepdim=True)
        values = (history - mean).permute(0, 2, 1).reshape(-1, 1, length)
        values = values + self.conv1d(values)
        values = values.reshape(-1, length // self.period_len, self.period_len)
        values = self.predictor(values.permute(0, 2, 1))
        values = values.permute(0, 2, 1).reshape(
            batch, channels, self.prediction_length
        )
        return {"prediction": values.permute(0, 2, 1) + mean}
