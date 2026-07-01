"""
Moirai-MoE Model for PyTorch Forecasting
----------------------------------------
"""

from typing import Any, Optional, Union
from warnings import warn

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class MoiraiMoE(TslibBaseModel):
    """
    Salesforce Moirai-MoE Model wrapper for PyTorch Forecasting.

    Parameters
    ----------
    module: MoiraiMoEFinetune or MoiraiMoEForecast or similar, optional
        The underlying MoiraiMoE module from uni2ts.
    loss : Descendants of ``pytorch_forecasting.metrics.Metric`` class
        Loss function to use for training.
    """

    def __init__(
        self,
        module: Any | None = None,
        loss: Metric | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
            **kwargs,
        )
        self.module = module

    @classmethod
    def _pkg(cls):
        """Return the package class for this model."""
        from pytorch_forecasting.models.moirai._moirai_moe_pkg_v2 import (
            MoiraiMoE_pkg_v2,
        )

        return MoiraiMoE_pkg_v2

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing output tensors
        """
        if self.module is not None:
            return self.module(x)
        raise NotImplementedError("Underlying module is not provided.")
