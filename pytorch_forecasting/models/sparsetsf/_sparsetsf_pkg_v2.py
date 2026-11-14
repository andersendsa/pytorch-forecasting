"""SparseTSF v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class SparseTSF_pkg_v2(Base_pkg):
    """SparseTSF package container."""

    _tags = {
        "info:name": "SparseTSF",
        "info:compute": 2,
        "info:y_type": ["numeric"],
        "authors": ["lss-1138"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Return model class."""
        from pytorch_forecasting.models.sparsetsf._sparsetsf_v2 import SparseTSF

        return SparseTSF

    @classmethod
    def get_datamodule_cls(cls):
        """Return data module class."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls):
        """Return parameters for package integration tests."""
        return [
            {
                "period_len": 2,
                "datamodule_cfg": {"max_encoder_length": 8, "max_prediction_length": 4},
            },
            {
                "period_len": 2,
                "model_type": "mlp",
                "d_model": 8,
                "datamodule_cfg": {"max_encoder_length": 8, "max_prediction_length": 4},
            },
        ]
