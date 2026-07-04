import torch
import torch.nn as nn
from pytorch_forecasting.models.moirai._moirai_v2 import Moirai
from pytorch_forecasting.models.moirai._moirai_moe_v2 import MoiraiMoE
from pytorch_forecasting.metrics import MAE

class DummyModule(nn.Module):
    def forward(self, x):
        return {"prediction": x["encoder_cont"] * 2}

loss = MAE()
mock_module = DummyModule()

try:
    # Test Moirai
    moirai = Moirai(module=mock_module, loss=loss)
    dummy_input = {"encoder_cont": torch.tensor([1.0, 2.0, 3.0])}
    out = moirai(dummy_input)
    print(f"Moirai forward output: {out}")

    # Test MoiraiMoE
    moirai_moe = MoiraiMoE(module=mock_module, loss=loss)
    out_moe = moirai_moe(dummy_input)
    print(f"MoiraiMoE forward output: {out_moe}")
except Exception as e:
    print(f"Error: {e}")
