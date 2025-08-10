# tests/test_smoke.py
import numpy as np
import torch
import torch.nn as nn
from mkcpu import MegakernelModel
import tempfile, os

def test_smoke():
    cin, h, w, cout1, cout2 = 3, 32, 32, 8, 5
    m = nn.Sequential(
        nn.Conv2d(cin, cout1, 3, 1, 1, bias=True),
        nn.ReLU(),
        nn.Conv2d(cout1, cout2, 3, 1, 1, bias=True),
        nn.ReLU(),
    ).eval()
    x = torch.randn(1, cin, h, w, dtype=torch.float32)
    y_ref = m(x).detach().numpy()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tiny.onnx")
        torch.onnx.export(m, x, path, opset_version=14)

        print("→ constructing")
        mk = MegakernelModel(path)            # if it segfaults here, it’s ONNX/protobuf/pack
        print("→ running")
        y = mk.run(x.numpy())              # if it segfaults here, it’s our executor
        print("ok", y.shape)

    np.testing.assert_allclose(y, y_ref, rtol=1e-3, atol=1e-4)

if __name__ == "__main__":
    test_smoke()
