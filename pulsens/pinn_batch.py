"""Batched (ensemble) version of pulsens/pinn.py: E independent inverse fits trained in one GPU program.
Every member has its own network weights, beta, A0, data and collocation points; losses are summed, and since
members share no parameters (and Adam is element-wise) this equals E separate fits.
Same equations, scalings and loss terms as pinn.fit (the pilot was run with that), checked in scripts/check_batch.py.
"""
import time, numpy as np, torch
from .pinn import MMHG, RHO, KR, P_REF, LX, SIG_A, SIG_U, SIG_P, metrics


def _lin(E, din, dout, g):
    k = 1 / np.sqrt(din)
    W = (torch.rand(E, din, dout, generator=g) * 2 - 1) * k
    b = (torch.rand(E, 1, dout, generator=g) * 2 - 1) * k
    return [torch.nn.Parameter(W), torch.nn.Parameter(b)]


class EnsNet(torch.nn.Module):
    """kind: B1 tanh MLP | B2 Fourier-feature MLP | B3 modified MLP. Input z (E, N, 2) -> (E, N, 2)."""

    def __init__(self, E, kind="B1", width=64, depth=4, nf=32, sigma=2.0, seeds=None, K=8):
        super().__init__()
        g = torch.Generator().manual_seed(int(seeds[0]) if seeds is not None else 0)
        self.kind, self.K = kind, K
        din = 2
        if kind == "B2":
            self.register_buffer("B", torch.randn(E, 2, nf, generator=g) * sigma)
            din = 2 * nf
        if kind == "CPC":                       # periodic harmonic time embedding + modified-MLP trunk
            din = 1 + 2 * K
        if kind in ("B3", "CPC"):
            self.U, self.V, self.H0 = (torch.nn.ParameterList(_lin(E, din, width, g)) for _ in range(3))
            self.hs = torch.nn.ModuleList([torch.nn.ParameterList(_lin(E, width, width, g)) for _ in range(depth - 1)])
        else:
            dims = [din] + [width] * depth
            self.hs = torch.nn.ModuleList([torch.nn.ParameterList(_lin(E, a, b, g)) for a, b in zip(dims[:-1], dims[1:])])
        self.out = torch.nn.ParameterList(_lin(E, width, 2, g))

    @staticmethod
    def _f(p, h):
        return torch.baddbmm(p[1], h, p[0])

    def forward(self, z):
        if self.kind == "B2":
            zb = 2 * np.pi * torch.bmm(z, self.B)
            z = torch.cat([torch.sin(zb), torch.cos(zb)], -1)
        if self.kind == "CPC":
            k = torch.arange(1, self.K + 1, device=z.device, dtype=z.dtype)
            ph = 2 * np.pi * z[..., 1:2] * k
            z = torch.cat([z[..., 0:1], torch.sin(ph), torch.cos(ph)], -1)
        if self.kind in ("B3", "CPC"):
            U, V = torch.tanh(self._f(self.U, z)), torch.tanh(self._f(self.V, z))
            h = torch.tanh(self._f(self.H0, z))
            for L in self.hs:
                gt = torch.tanh(self._f(L, h)); h = (1 - gt) * U + gt * V
        else:
            h = z
            for L in self.hs:
                h = torch.tanh(self._f(L, h))
        return self._f(self.out, h)


def fit_batch(ds, levels, kind="B1", iters=8000, n_col=2048, seeds=None, dev="cuda", lr=1e-3, log_every=0):
    """ds: list of case dicts from pinn.load_case (each already built for its level); levels: list of 'a'..'d'."""
    E = len(ds)
    seeds = seeds if seeds is not None else [0] * E
    torch.manual_seed(int(seeds[0]))
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    net = EnsNet(E, kind, seeds=seeds).to(dev)
    A_s = f([d["A_s"] for d in ds])[:, None]
    Tc = f([d["Tc"] for d in ds])[:, None]
    Q_s = 1e-5
    beta0 = 40 * MMHG / (0.10 * torch.sqrt(A_s))
    M_mom = (Tc / Q_s) * (A_s / RHO) * beta0 / (2 * torch.sqrt(A_s)) * (A_s / LX) * 1e-2
    lb = torch.zeros(E, 1, device=dev, requires_grad=True)
    la0 = f([[np.log(np.sqrt(d["A_min"] / d["A_s"]))] for d in ds]).requires_grad_(True)
    opt = torch.optim.Adam(list(net.parameters()) + [lb, la0], lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1 / iters))
    zd = torch.stack([f(np.stack([d["xi"] for d in ds])), f(np.stack([d["tau"] for d in ds]))], -1)
    Ad = f(np.stack([d["A"] for d in ds])) / A_s
    isb = f([lv == "b" for lv in levels])
    isc = f([lv in ("c", "d") for lv in levels])
    nu = len(ds[0]["tau_mid"])
    zu = torch.stack([torch.zeros(E, nu, device=dev), f(np.stack([d.get("u_tau", d["tau_mid"]) for d in ds]))], -1)
    ud = f(np.stack([d.get("u", np.zeros(nu)) for d in ds]))
    cuff = f([d.get("cuff", (0.0, 0.0)) for d in ds])
    zc_eval = torch.stack([torch.zeros(E, 64, device=dev), torch.linspace(0, 1, 65, device=dev)[:-1].expand(E, 64)], -1)

    cpc = kind == "CPC"
    if cpc:   # Psi potential: A_hat = 1 + 0.2 dPsi/dxi, Q_hat = -0.2 kappa dPsi/dtau + qbar  => mass exact
        kappa = A_s * LX / (Q_s * Tc)
        qbar = torch.full((E, 1), 0.6, device=dev, requires_grad=True)
        opt.add_param_group(dict(params=[qbar]))

    def fields(z):
        if not cpc:
            o = net(z)
            return 1.0 + 0.2 * o[..., 0], o[..., 1]
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            psi = net(z)[..., 0]
            gp = torch.autograd.grad(psi.sum(), z, create_graph=True)[0]
        return 1.0 + 0.2 * gp[..., 0], -0.2 * kappa * gp[..., 1] + qbar

    def pressure(Ah):
        beta = beta0 * torch.exp(lb)
        return P_REF + beta * (torch.sqrt(Ah * A_s) - torch.sqrt(A_s) * torch.exp(la0)), beta

    t0 = time.time()
    for it in range(iters):
        zc = torch.rand(E, n_col, 2, device=dev); zc[..., 0] = 2 * zc[..., 0] - 1
        zc.requires_grad_(True)
        Ah, Qh = fields(zc)
        gA = torch.autograd.grad(Ah.sum(), zc, create_graph=True)[0]
        gQ = torch.autograd.grad(Qh.sum(), zc, create_graph=True)[0]
        g2 = torch.autograd.grad((Qh ** 2 / Ah).sum(), zc, create_graph=True)[0]
        beta = beta0 * torch.exp(lb)
        r_mass = gA[..., 1] + (Q_s * Tc / (A_s * LX)) * gQ[..., 0]
        A = Ah * A_s
        dpdx = beta / (2 * torch.sqrt(A)) * gA[..., 0] * A_s / LX
        r_mom = gQ[..., 1] + (Tc / Q_s) * ((Q_s ** 2 / (A_s * LX)) * g2[..., 0] + A / RHO * dpdx + KR * Q_s * Qh / A)
        r_mom = r_mom / M_mom
        Ahd, _ = fields(zd)
        loss_d = torch.mean(((Ahd - Ad) / SIG_A) ** 2, 1)
        loss = loss_d + torch.mean(r_mass ** 2, 1) + torch.mean(r_mom ** 2, 1)
        Au, Qu = fields(zu)
        loss = loss + isb * torch.mean(((Qu * Q_s / (Au * A_s) - ud) / SIG_U) ** 2, 1)
        pc, _ = pressure(fields(zc_eval)[0])
        loss = loss + isc * (((pc.max(1).values - cuff[:, 0]) / SIG_P) ** 2 + ((pc.min(1).values - cuff[:, 1]) / SIG_P) ** 2)
        opt.zero_grad(); loss.sum().backward(); opt.step(); sched.step()
        if log_every and it % log_every == 0:
            print(f"  it {it} loss median {float(loss.median()):.3f} ({time.time() - t0:.0f} s)", flush=True)
    outs = []
    with torch.no_grad():
        for i, d in enumerate(ds):
            ze = torch.stack([torch.zeros(len(d["eval_tau"]), device=dev), f(d["eval_tau"])], -1)[None]
            Ah = fields(ze.expand(E, -1, -1).contiguous())[0][i].detach()
            b = float(beta0[i, 0] * torch.exp(lb[i, 0]))
            sqA0 = float(torch.sqrt(A_s[i, 0]) * torch.exp(la0[i, 0]))
            pe = (P_REF + b * (torch.sqrt(Ah * A_s[i, 0]) - sqA0)).cpu().numpy()
            c0 = np.sqrt(b / (2 * RHO)) * sqA0 ** 0.5
            m = metrics(d, pe, b, c0)
            m.update(loss=float(loss[i].detach()), loss_data=float(loss_d[i].detach()), secs=round((time.time() - t0) / E, 1))
            outs.append(m)
    return outs
