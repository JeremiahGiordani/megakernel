import os
import time
import argparse
import tempfile
import numpy as np

# Threads: default to OMP_NUM_THREADS or 1
DEF_THREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))

def timer(fn, warmup=10, iters=50):
    # Warmup
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0  # ms/iter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cin", type=int, default=64)
    parser.add_argument("--cout1", type=int, default=64)
    parser.add_argument("--cout2", type=int, default=64)
    parser.add_argument("--h", type=int, default=56)
    parser.add_argument("--w", type=int, default=56)
    parser.add_argument("--threads", type=int, default=DEF_THREADS, help="Number of intra-op threads for all runtimes")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    # Set global env for libs that read OMP_NUM_THREADS
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

    import torch
    import torch.nn as nn
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    # Build model
    class TinyConvChain(nn.Module):
        def __init__(self, cin, cout1, cout2):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout1, 3, 1, 1, bias=True)
            self.act1  = nn.ReLU()
            self.conv2 = nn.Conv2d(cout1, cout2, 3, 1, 1, bias=True)
            self.act2  = nn.ReLU()
        def forward(self, x):
            x = self.act1(self.conv1(x))
            x = self.act2(self.conv2(x))
            return x

    cin, h, w, cout1, cout2 = args.cin, args.h, args.w, args.cout1, args.cout2
    model = TinyConvChain(cin, cout1, cout2).eval()
    x_t = torch.randn(1, cin, h, w, dtype=torch.float32)

    # Export ONNX
    with tempfile.TemporaryDirectory() as td:
        onnx_path = os.path.join(td, "tiny.onnx")
        torch.onnx.export(
            model, x_t, onnx_path,
            input_names=["input"], output_names=["output"],
            opset_version=14, do_constant_folding=True, dynamic_axes=None
        )

        # -------- Our runtime --------
        from mkcpu import MegakernelModel
        mk = MegakernelModel(onnx_path)

        x_np = x_t.detach().cpu().numpy()
        mk_out = mk.run(x_np)
        def run_mk():
            mk.run(x_np)

        mk_ms = timer(run_mk, warmup=args.warmup, iters=args.iters)

        # -------- PyTorch (eager, oneDNN underneath) --------
        def run_torch():
            with torch.inference_mode():
                _ = model(x_t)
        torch_out = model(x_t)
        torch_ms = timer(run_torch, warmup=args.warmup, iters=args.iters)

        # -------- ONNX Runtime (CPUExecutionProvider) --------
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Execution is SEQUENTIAL for fairness in single-thread scenarios
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.intra_op_num_threads  = args.threads
        so.inter_op_num_threads  = args.threads

        providers = ["CPUExecutionProvider"]
        # provider_options = [{
        #     "intra_op_num_threads": args.threads,
        #     "inter_op_num_threads": 1,
        #     # Depending on the wheel, these may be ignored; harmless.
        #     "arena_extend_strategy": "kSameAsRequested",
        # }]

        sess = ort.InferenceSession(onnx_path, sess_options=so,
                                    providers=providers,)

        input_name = sess.get_inputs()[0].name
        def run_ort():
            _ = sess.run(None, {input_name: x_np})
        onnx_out = sess.run(None, {input_name: x_np})

        ort_ms = timer(run_ort, warmup=args.warmup, iters=args.iters)

    print("=== Verifying correctness ===")
    print("Torch vs ONNX :", np.allclose(torch_out.numpy(), onnx_out, atol=1e-4))
    print("Torch vs MK :", np.allclose(torch_out.numpy(), mk_out, atol=1e-4))

    print("\n=== Conv→ReLU→Conv→ReLU B=1 ===")
    print(f"Shape: N=1, C_in={cin}, HxW={h}x{w}, C_mid={cout1}, C_out={cout2}")
    print(f"Threads: {args.threads} (OMP_NUM_THREADS)")
    print(f"PyTorch:        {torch_ms:8.3f} ms")
    print(f"ONNX Runtime:   {ort_ms:8.3f} ms")
    print(f"Our runtime:    {mk_ms:8.3f} ms")

if __name__ == "__main__":
    main()
