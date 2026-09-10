"""Is the GPU the forward's lever? (docs/24 section 6 -- the probe, not the build)

`docs/21` section 8.13 closed the CPU question: nothing short of changing the model buys more
than ~1.13x, so the forward's remaining lever is a GPU. This script measures that lever on the
forward alone, in isolation from the load, so the number is available before any tile is run.

Why it is not obvious that a GPU wins: 14,535 parameters of depthwise-separable Conv1d over a
100-step curve is very low arithmetic intensity. The expected limit is **kernel launch
overhead**, not FLOPs -- a member is ~20 kernels and the ensemble runs five of them per batch,
so at batch 8,192 the GPU may well spend more time launching than computing. Two arms attack
exactly that, and they are the reason this script exists rather than a one-line `--device cuda`:

  loop      the production path: five members, batched, one after another
  grouped   the five members as one grouped convolution (`bench_forward.GroupedTrunk`), so
            the trunk is one set of launches instead of five. Measured **slower** on CPU
            (section 8.13), which is not evidence about a launch-bound device.
  graph     the same work captured as a CUDA graph and replayed: launch cost collapses to one
            replay per batch. If the loop is launch-bound this is where it shows.

Every arm is timed the way production would pay for it -- numpy in, numpy out, host-to-device
and device-to-host included, `synchronize` before stopping the clock -- and `--no-transfer`
additionally reports the same arms with the data already resident, which separates "the GPU is
slow at this" from "the PCIe round trip is the cost".

Bit-exactness does not apply across devices (`docs/24` section 6): float32 reductions do not
associate the same way, so the deviation is reported and the tolerance is a decision, not a
test. `grouped` is bit-identical to `loop` on the same device.

Usage:
    PYTHONPATH=src python scripts/bench/bench_forward_gpu.py [--n 46553] [--batch 8192]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_forward import CKPT_DIR, NGS, GroupedTrunk, load_members     # noqa: E402


def timeit(fn, repeat: int, sync: bool) -> float:
    """Best of ``repeat``; the minimum, so a noisy neighbour cannot inflate it."""
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


def batches(n: int, batch: int):
    return [(s, min(s + batch, n)) for s in range(0, n, batch)]


def make_loop(members, x, ctx, batch, dev, transfer: bool):
    """The production path: `FacetEnsemble.predict_scaled`, minus the sklearn context transform.

    Member-outer, batch-inner, exactly as `mapinfer.predict_scaled` runs it, so that what is
    measured here is what a tile-year would pay.
    """
    def run():
        out = []
        for k, m in enumerate(members):
            preds = []
            for s, e in batches(len(x), batch):
                xb, cb = x[s:e], ctx[k][s:e]
                if transfer:
                    xb, cb = xb.to(dev, non_blocking=True), cb.to(dev, non_blocking=True)
                p = m(xb, cb)
                preds.append(p.cpu() if transfer else p)
            out.append(torch.cat(preds))
        return torch.stack(out)
    return run


def make_grouped(trunk, members, x_rep, ctx, batch, dev, transfer: bool):
    def run():
        out = []
        for s, e in batches(len(x_rep), batch):
            xb = x_rep[s:e].to(dev, non_blocking=True) if transfer else x_rep[s:e]
            emb = trunk(xb)
            heads = []
            for k, m in enumerate(members):
                cb = ctx[k][s:e].to(dev, non_blocking=True) if transfer else ctx[k][s:e]
                p = m.head(torch.cat([emb[:, k], cb], 1))
                heads.append(p.cpu() if transfer else p)
            out.append(torch.stack(heads))
        return torch.cat(out, dim=1)
    return run


def graph_runner(fn_of_static, x_static, ctx_static, warmup: int = 3):
    """Capture ``fn_of_static`` as a CUDA graph over fixed input buffers.

    Requires static shapes, so the caller feeds it one full batch and copies each real batch
    into the buffers before replaying. That copy is a device-to-device one and is included.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn_of_static()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn_of_static()
    return g, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=46553,
                    help="pixels in a tile-year; default is the measured production median")
    ap.add_argument("--batch", type=int, default=8192, help="production default")
    ap.add_argument("--batch-sweep", default="8192,32768,131072",
                    help="comma list of batch sizes for the cuda arms")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--threads", default="1,4", help="cpu arms, comma list of torch threads")
    ap.add_argument("--ckpt", type=Path, default=CKPT_DIR)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this script is the GPU probe of docs/24 section 6")
    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
          f"cuda {torch.version.cuda}")

    members_cpu = load_members(a.ckpt)
    n_ctx = (members_cpu[0].head[1].weight.shape[1]
             - members_cpu[0].blocks[-1].pw.weight.shape[0])
    rng = np.random.default_rng(0)
    x_np = rng.random((a.n, 1, NGS), dtype=np.float32)
    ctx_np = [rng.random((a.n, n_ctx), dtype=np.float32) for _ in members_cpu]
    x_cpu = torch.from_numpy(x_np)
    ctx_cpu = [torch.from_numpy(c) for c in ctx_np]
    x_rep_cpu = x_cpu.expand(-1, len(members_cpu), -1).contiguous()

    print(f"n={a.n:,} px, {len(members_cpu)} seeds, {n_ctx} context features, "
          f"best of {a.repeat}\n")
    rows: list[tuple[str, float, float]] = []
    ref = None

    with torch.no_grad():
        # ---- CPU baselines, same harness, same machine (section 8.6: only same-hardware
        #      comparisons are meaningful) ----------------------------------------------
        for t in [int(v) for v in a.threads.split(",")]:
            torch.set_num_threads(t)
            fn = make_loop(members_cpu, x_cpu, ctx_cpu, a.batch, "cpu", transfer=False)
            secs = timeit(fn, max(2, a.repeat // 2), sync=False)
            if ref is None:
                ref = fn().numpy()
            rows.append((f"cpu loop, {t} thread(s), batch {a.batch}", secs, 0.0))
        torch.set_num_threads(1)

        members = [m.to(dev) for m in load_members(a.ckpt)]
        trunk = GroupedTrunk(load_members(a.ckpt), fold=False).to(dev)
        x_dev, ctx_dev = x_cpu.to(dev), [c.to(dev) for c in ctx_cpu]
        x_rep_dev = x_rep_cpu.to(dev)

        for b in [int(v) for v in a.batch_sweep.split(",")]:
            for name, fn, dv in [
                ("cuda loop", make_loop(members, x_cpu, ctx_cpu, b, dev, True), None),
                ("cuda grouped",
                 make_grouped(trunk, members, x_rep_cpu, ctx_cpu, b, dev, True), None),
                ("cuda loop, resident",
                 make_loop(members, x_dev, ctx_dev, b, "cuda", False), None),
                ("cuda grouped, resident",
                 make_grouped(trunk, members, x_rep_dev, ctx_dev, b, "cuda", False), None),
            ]:
                fn()                                        # warm up kernels and autotuning
                torch.cuda.synchronize()
                secs = timeit(fn, a.repeat, sync=True)
                out = fn()
                out = out.cpu().numpy() if out.is_cuda else out.numpy()
                d = float(np.abs(out - ref).max())
                rows.append((f"{name}, batch {b:,}", secs, d))

        # ---- CUDA graph: one replay per batch instead of ~100 launches ----------------
        for b in [int(v) for v in a.batch_sweep.split(",")]:
            if b > a.n:
                continue
            xs = x_dev[:b].clone()
            cs = [c[:b].clone() for c in ctx_dev]
            static = make_loop(members, xs, cs, b, "cuda", False)
            g, out_buf = graph_runner(static, xs, cs)
            full = batches(a.n, b)

            def run(g=g, xs=xs, cs=cs, full=full, out_buf=out_buf):
                acc = []
                for s, e in full:
                    m = e - s
                    xs[:m].copy_(x_dev[s:e])
                    for k, c in enumerate(cs):
                        c[:m].copy_(ctx_dev[k][s:e])
                    g.replay()
                    acc.append(out_buf[:, :m].clone())
                return torch.cat(acc, dim=1)

            run()
            torch.cuda.synchronize()
            secs = timeit(run, a.repeat, sync=True)
            d = float(np.abs(run().cpu().numpy() - ref).max())
            rows.append((f"cuda loop, CUDA graph, resident, batch {b:,}", secs, d))

    base = rows[0][1]
    print(f"  {'arm':46s} {'s':>8s} {'vs cpu1':>8s} {'px/s':>12s} {'max abs':>10s}")
    for name, secs, d in rows:
        print(f"  {name:46s} {secs:8.3f} {base / secs:7.2f}x {a.n / secs:12,.0f} "
              f"{d:10.2e}")

    print(f"\nreference for 'max abs' is the cpu arm; cross-device float32 is not "
          f"bit-identical by construction (docs/24 section 6)")


if __name__ == "__main__":
    main()
