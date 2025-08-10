import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from mkcpu import MegakernelModel

torch.manual_seed(0)
torch.set_grad_enabled(False)

class TinyConvChain(nn.Module):
    def __init__(self, cin, cout1, cout2):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout1, kernel_size=3, stride=1, padding=1, bias=True)
        self.act1  = nn.ReLU()
        self.conv2 = nn.Conv2d(cout1, cout2, kernel_size=3, stride=1, padding=1, bias=True)
        self.act2  = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x

def export_onnx(model: nn.Module, x: torch.Tensor, path: str):
    model.eval()
    torch.onnx.export(
        model, x, path,
        input_names=["input"], output_names=["output"],
        opset_version=14, do_constant_folding=True,
        dynamic_axes=None  # static NCHW (N=1) expected by our MVP
    )

@pytest.mark.parametrize(
    "cin,h,w,cout1,cout2",
    [
        (3, 56, 56, 20, 7),     # tail channels (not multiples of 16)
        (8, 28, 28, 16, 13),    # one multiple-of-16, one not
        (32, 14, 14, 48, 32),   # all multiples of 16
    ],
)
def test_conv_relu_conv_relu_matches_pytorch(cin, h, w, cout1, cout2):
    # Build model & input
    model = TinyConvChain(cin, cout1, cout2).eval()
    x_torch = torch.randn(1, cin, h, w, dtype=torch.float32)
    y_ref = model(x_torch).detach().cpu().numpy()

    # Export ONNX to a temp file
    with tempfile.TemporaryDirectory() as td:
        onnx_path = os.path.join(td, "tiny.onnx")
        export_onnx(model, x_torch, onnx_path)

        # Run our engine
        print("Running constructor...")
        mk = MegakernelModel(onnx_path)
        print("Successfully ran constructor")
        x_np = x_torch.detach().cpu().numpy()
        print("Running model.run function")
        y_me = mk.run(x_np)
        print("Run Successful")

    assert y_me.shape == y_ref.shape

    # Tolerances: reference conv is scalar and may sum in a different order than PyTorch
    atol, rtol = 1e-4, 1e-3
    np.testing.assert_allclose(y_me, y_ref, rtol=rtol, atol=atol)

if __name__ == "__main__":
    test_conv_relu_conv_relu_matches_pytorch(17, 257, 257, 17, 181)
