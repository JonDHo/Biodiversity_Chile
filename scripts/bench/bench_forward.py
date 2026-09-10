"""Does the CNN forward have a big CPU lever left? (docs/21 §8.13 -- answer: no)

The forward is ~49 % of a tile's core-seconds, so it is the largest single term. This script
tests the two structural ideas that were outstanding, and then bounds what is removable at all.

Part 1, attribution -- four arms so the two transforms are separable:

  A  loop      the current path: five `Pheno1D` forwards over an identical input
  B  folded    A, with each BatchNorm folded into the convolution before it
  C  grouped   one grouped-conv trunk + per-seed heads, instead of five narrow passes
  D  both

The grouping idea was that five narrow passes over identical input are latency- and
cache-bound (14,535 parameters, very low arithmetic intensity), so one pass at 5x channel
width should be faster. It is not: PyTorch's grouped convolutions do not reach the same oneDNN
paths as dense ones, and C measures slightly *slower* than A.

Part 2, the ceiling -- successively delete work until nothing is left to delete. This is what
closes the question: the last row removes the activation and the depthwise convolutions
entirely, which changes the model rather than its implementation, and still only reaches ~2x.
So there is no large CPU lever here and the forward's real lever is a GPU.

Neither B nor D is bit-identical to A -- both change floating-point operation order, and are
mathematically exact, which is not the same thing. C *is* bit-identical, for whatever that is
worth given it is slower. Deviations are reported as max absolute and max relative; prefer the
absolute one, because these are transformed-space targets that pass through zero, so relative
error explodes on values that are near zero and mean nothing.

Usage:
    PYTHONPATH=src python scripts/bench/bench_forward.py [--n 16384] [--repeat 2]
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CKPT_DIR = Path("results/models_unified_topofix/"
                "C1D01_curve1d_kndvi_raw100_pg-all_unified_ctr_FINAL_alldata/final")
NGS = 100


def load_members(ckpt_dir: Path, width: str = "B"):
    """The five deployed members, built exactly the way `FacetEnsemble` builds them."""
    from biodiv.models_conv import build_model

    members = []
    for p in sorted(ckpt_dir.glob("model_seed*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = build_model("C1D", c_in=ck["input_shape"][0], n_out=len(ck["targets"]),
                            n_ctx=len(ck["ctx_preprocessor"].feature_names), width=width)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        members.append(model)
    if not members:
        raise SystemExit(f"no checkpoints in {ckpt_dir}")
    return members


def _fold_bn(w: torch.Tensor, bn: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor]:
    """Conv weight (bias-free) + BatchNorm in eval -> (scaled weight, bias).

    In eval a BatchNorm is the fixed affine map ``gamma*(z-mean)/sqrt(var+eps) + beta``, so it
    folds into the convolution that produced ``z``: scale each output channel's kernel by
    ``gamma/sqrt(var+eps)`` and carry the rest as a bias the convolution did not have.
    """
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return w * s.reshape(-1, *([1] * (w.dim() - 1))), bn.bias - bn.running_mean * s


def _conv_like(ref: nn.Conv1d, cin, cout, k, stride, groups, weight, bias) -> nn.Conv1d:
    c = nn.Conv1d(cin, cout, k, stride=stride, padding=ref.padding, groups=groups,
                  bias=bias is not None, padding_mode=ref.padding_mode)
    with torch.no_grad():
        c.weight.copy_(weight)
        if bias is not None:
            c.bias.copy_(bias)
    return c


def fold_member(m: nn.Module) -> nn.Module:
    """A copy of one `Pheno1D` with every BatchNorm folded into the convolution before it."""
    f = copy.deepcopy(m).eval()
    with torch.no_grad():
        w, b = _fold_bn(f.stem[0].weight, f.stem[1])
        f.stem[0] = _conv_like(f.stem[0], f.stem[0].in_channels, f.stem[0].out_channels,
                               f.stem[0].kernel_size[0], f.stem[0].stride[0], 1, w, b)
        f.stem[1] = nn.Identity()
        for blk in f.blocks:
            w, b = _fold_bn(blk.pw.weight, blk.bn)
            blk.pw = _conv_like(blk.pw, blk.pw.in_channels, blk.pw.out_channels, 1, 1, 1, w, b)
            blk.bn = nn.Identity()
    return f


class GroupedTrunk(nn.Module):
    """The `Pheno1D.embed` trunks of every member, as one grouped convolution.

    ``(B, S, ngs)`` -- the curve repeated once per seed -- to ``(B, S, w3)``. Every convolution
    carries ``groups`` such that seed ``s`` only sees seed ``s``'s channels, so this computes
    exactly what the loop computes. ``fold`` additionally folds the BatchNorms.
    """

    def __init__(self, members: list[nn.Module], fold: bool = False):
        super().__init__()
        self.n_seeds = s = len(members)
        self.fold = fold
        w1 = members[0].stem[0].weight.shape[0]
        self.w3 = members[0].blocks[-1].pw.weight.shape[0]
        cin = members[0].stem[0].weight.shape[1]

        stem_w, stem_b = torch.cat([m.stem[0].weight for m in members], 0), None
        if fold:
            parts = [_fold_bn(m.stem[0].weight, m.stem[1]) for m in members]
            stem_w = torch.cat([p[0] for p in parts], 0)
            stem_b = torch.cat([p[1] for p in parts], 0)
        self.stem = _conv_like(members[0].stem[0], s * cin, s * w1, 5, 1, s, stem_w, stem_b)
        self.stem_bn = None if fold else self._bn([m.stem[1] for m in members])

        self.dw, self.pw = nn.ModuleList(), nn.ModuleList()
        bns = []
        for j, ref in enumerate(members[0].blocks):
            blocks = [m.blocks[j] for m in members]
            ci, co = ref.dw.weight.shape[0], ref.pw.weight.shape[0]
            # depthwise is already fully grouped, so concatenating channels is all it needs
            self.dw.append(_conv_like(ref.dw, s * ci, s * ci, ref.dw.kernel_size[0],
                                      ref.dw.stride[0], s * ci,
                                      torch.cat([b.dw.weight for b in blocks], 0), None))
            pw_w, pw_b = torch.cat([b.pw.weight for b in blocks], 0), None
            if fold:
                parts = [_fold_bn(b.pw.weight, b.bn) for b in blocks]
                pw_w, pw_b = torch.cat([p[0] for p in parts], 0), torch.cat([p[1] for p in parts])
            self.pw.append(_conv_like(ref.pw, s * ci, s * co, 1, 1, s, pw_w, pw_b))
            bns.append(self._bn([b.bn for b in blocks]))
        self.bn = None if fold else nn.ModuleList(bns)
        self.act, self.pool = nn.GELU(), nn.AdaptiveAvgPool1d(1)
        self.eval()

    @staticmethod
    def _bn(bns: list[nn.BatchNorm1d]) -> nn.BatchNorm1d:
        out = nn.BatchNorm1d(sum(b.weight.numel() for b in bns), eps=bns[0].eps)
        with torch.no_grad():
            out.weight.copy_(torch.cat([b.weight for b in bns]))
            out.bias.copy_(torch.cat([b.bias for b in bns]))
            out.running_mean.copy_(torch.cat([b.running_mean for b in bns]))
            out.running_var.copy_(torch.cat([b.running_var for b in bns]))
        return out.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.stem(x) if self.fold else self.stem_bn(self.stem(x)))
        for j in range(len(self.dw)):
            z = self.pw[j](self.dw[j](z))
            z = self.act(z if self.fold else self.bn[j](z))
        return self.pool(z).flatten(1).reshape(-1, self.n_seeds, self.w3)


def ablate(m: nn.Module, gelu=None, drop_dw: bool = False, zero_pad: bool = False) -> nn.Module:
    """A member with pieces deleted, to bound what is removable.

    These are **not** valid models -- `drop_dw` removes a convolution and `gelu=Identity`
    removes the non-linearity. The point is the ceiling: if deleting the work outright does not
    buy much, no faster implementation of it will either.
    """
    f = copy.deepcopy(m).eval()
    if gelu is not None:
        for mod in f.modules():
            for n, c in list(mod.named_children()):
                if isinstance(c, nn.GELU):
                    setattr(mod, n, gelu())
    if drop_dw:
        for b in f.blocks:
            st = b.dw.stride[0]
            b.dw = nn.Identity() if st == 1 else nn.AvgPool1d(1, stride=st)
    if zero_pad:
        for mod in f.modules():
            if isinstance(mod, nn.Conv1d) and mod.padding_mode != "zeros":
                mod.padding_mode = "zeros"
    return f


def timeit(fn, repeat: int) -> float:
    """Best of ``repeat`` -- the minimum, not the mean, so a noisy neighbour cannot inflate it."""
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16384, help="pixels; production median ~46k")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1, help="torch threads; Argo pins 1")
    ap.add_argument("--ckpt", type=Path, default=CKPT_DIR)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    members = load_members(a.ckpt)
    n_ctx = members[0].head[1].weight.shape[1] - members[0].blocks[-1].pw.weight.shape[0]
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.random((a.n, 1, NGS), dtype=np.float32))
    ctx = [torch.from_numpy(rng.random((a.n, n_ctx), dtype=np.float32)) for _ in members]
    x_rep = x.expand(-1, len(members), -1).contiguous()

    print(f"n={a.n:,} px, {len(members)} seeds, {a.threads} torch thread(s), "
          f"best of {a.repeat}\n")

    with torch.no_grad():
        folded = [fold_member(m) for m in members]
        tc, td = GroupedTrunk(members, False), GroupedTrunk(members, True)
        def loop(ms):
            return lambda: torch.stack([m(x, ctx[k]) for k, m in enumerate(ms)])

        def grp(tr):
            # The trunk runs ONCE for all five seeds -- that is the whole point of the arm.
            # Calling it inside the comprehension would run it per seed and measure the
            # opposite of what this tests.
            def run():
                emb = tr(x_rep)
                return torch.stack([m.head(torch.cat([emb[:, k], ctx[k]], 1))
                                    for k, m in enumerate(members)])
            return run

        print("Part 1 -- attribution (all five seeds)")
        arms = {"A loop (current)": loop(members), "B loop + BN folded": loop(folded),
                "C grouped": grp(tc), "D grouped + BN folded": grp(td)}
        ref, base = arms["A loop (current)"]().numpy(), None
        print(f"  {'arm':26s} {'s':>7s} {'speedup':>8s} {'max abs':>10s} {'max rel':>10s}")
        for name, fn in arms.items():
            secs = timeit(fn, a.repeat)
            base = secs if base is None else base
            d = np.abs(fn().numpy() - ref)
            print(f"  {name:26s} {secs:7.2f} {base / secs:7.2f}x {d.max():10.3e} "
                  f"{(d / np.maximum(np.abs(ref), 1e-12)).max():10.3e}")

        print("\nPart 2 -- ceiling (one seed; last rows are not valid models)")
        one = members[0]
        tanh = lambda: nn.GELU(approximate="tanh")
        variants = [
            ("current", {}),
            ("BN folded", None),
            ("BN folded + GELU tanh approx", dict(gelu=tanh)),
            ("BN folded + zero pad (not circular)", dict(zero_pad=True)),
            ("CEILING: also GELU and depthwise deleted",
             dict(gelu=nn.Identity, drop_dw=True, zero_pad=True)),
        ]
        r0 = None
        for name, kw in variants:
            m = one if kw == {} else (fold_member(one) if kw is None
                                      else ablate(fold_member(one), **kw))
            secs = timeit(lambda: m(x, ctx[0]), a.repeat)
            r0 = secs if r0 is None else r0
            print(f"  {name:42s} {secs:7.2f} {r0 / secs:7.2f}x")


if __name__ == "__main__":
    main()
