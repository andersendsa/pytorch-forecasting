"""
Moirai Model for PyTorch Forecasting
------------------------------------
"""

from typing import Any, Optional, Union
from warnings import warn

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class Moirai(TslibBaseModel):
    """
    Salesforce Moirai Model wrapper for PyTorch Forecasting.

    Parameters
    ----------
    module: MoiraiFinetune or MoiraiForecast or similar, optional
        The underlying Moirai module from uni2ts.
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
        from pytorch_forecasting.models.moirai._moirai_pkg_v2 import Moirai_pkg_v2

        return Moirai_pkg_v2

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
            # We expect module to be a PyTorch model like MoiraiFinetune
            # We extract necessary components from x

            # This is a generic pass-through if the user structured their
            # module to accept the PTF data format.
            # In PTF, x usually has "encoder_cont", "target_past", etc.

            # Since Moirai has its own expectations (e.g. past_target),
            # adaptation is normally done in an adapter. But since we use
            # module directly here as requested by soft-dependency wrappers,
            # we rely on the module or adapter handling it.
            return self.module(x)

        raise NotImplementedError("Underlying module is not provided.")
