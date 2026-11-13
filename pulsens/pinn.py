"""Phase 4: baseline PINNs for the L1 inverse problem on a 2 cm scan window, one cardiac cycle.

Unknowns: A(x,t), Q(x,t) from the network; wall stiffness beta and reference area A0 as trainable scalars;
p = p_ref + beta (sqrt A - sqrt A0) (tube law, p_ref = 80 mmHg as in the twin model).
Physics residuals: mass and momentum of the 1D model (same equations as pulsens/l1.py).
Data levels:
  a  D(x,t) only
  b  a + mean velocity at the mid column
  c  a + cuff SBP/DBP at the mid column (ideal, same site)
  d  a + biased cuff (distant site: SBP +5..10, DBP +-2 mmHg)
Models: B1 tanh MLP, B2 Fourier-feature MLP, B3 modified MLP (Wang, Teng & Perdikaris 2021).
B0 (no PINN): linear cuff scaling of the diameter waveform (levels c, d).
"""
import json, time, numpy as np, torch, torch.nn as nn

MMHG, RHO, MU = 133.322, 1060.0, 0.0035
KR = 8 * np.pi * MU / RHO
P_REF = 80 * MMHG
LX = 0.01                         # half window (m); xi = (x - 0.075) / LX
SIG_A, SIG_U, SIG_P = 0.012, 0.02, 2 * MMHG


class MLP(nn.Module):
    def __init__(self, din=2, width=64, depth=4, dout=2):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]; d = width
        layers += [nn.Linear(d, dout)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class FourierMLP(nn.Module):
    def __init__(self, width=64, depth=4, nf=32, sigma=2.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(2, nf, generator=g) * sigma)
        self.mlp = MLP(2 * nf, width, depth)

    def forward(self, z):
        zb = 2 * np.pi * z @ self.B
        return self.mlp(torch.cat([torch.sin(zb), torch.cos(zb)], -1))


class ModifiedMLP(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        self.U, self.V, self.H0 = nn.Linear(2, width), nn.Linear(2, width), nn.Linear(2, width)
        self.hs = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, 2)

    def forward(self, z):
        U, V = torch.tanh(self.U(z)), torch.tanh(self.V(z))
        h = torch.tanh(self.H0(z))
        for L in self.hs:
            g = torch.tanh(L(h)); h = (1 - g) * U + g * V
        return self.out(h)


MODELS = dict(B1=MLP, B2=FourierMLP, B3=ModifiedMLP)


def load_case(path, level, rng):
    z = np.load(path)
    m, pr = json.loads(str(z["metrics"])), json.loads(str(z["params"]))
    Tc = 60.0 / pr["HR"]
    tau = (z["scan_t"] % Tc) / Tc
    xi = (z["scan_x"] - 0.075) / LX
    D = z["scan_D"] * 1e-3
    A = np.pi * D ** 2 / 4
    T, X = np.meshgrid(tau, xi, indexing="ij")
    ft = z["t"] / Tc
    mid = int(np.argmin(np.abs(z["x"] - 0.075)))
    p_mid = z["p"][:, mid]
    u_mid = z["Q"][:, mid] / z["A"][:, mid]
    d = dict(Tc=Tc, tau=T.ravel(), xi=X.ravel(), A=A.ravel(), A_s=float(A.mean()), A_min=float(A.min()),
             eval_tau=ft, eval_p=p_mid, beta_true=m["beta"], c0_true=m["c0"], SBP=m["SBP"], DBP=m["DBP"], case=pr["case"],
             D_mid=z["scan_D"][:, 4] * 1e-3, tau_mid=tau)
    if level == "b":
        ui = np.interp(tau, ft, u_mid, period=1.0)
        d["u_tau"], d["u"] = tau, ui + rng.normal(0, SIG_U, ui.shape)
    if level in ("c", "d"):
        s, dd = float(p_mid.max()), float(p_mid.min())
        if level == "d":
            s += rng.uniform(5, 10) * MMHG; dd += rng.uniform(-2, 2) * MMHG
        d["cuff"] = (s, dd)
    return d


def b0_cuff_scaling(d, nb=40):
    """Phase-binned mid-column diameter, linearly scaled to cuff SBP/DBP."""
    bins = np.minimum((d["tau_mid"] * nb).astype(int), nb - 1)
    Db = np.array([d["D_mid"][bins == k].mean() if np.any(bins == k) else np.nan for k in range(nb)])
    ok = ~np.isnan(Db)
    tc = (np.arange(nb) + 0.5) / nb
    Db = np.interp(tc, tc[ok], Db[ok], period=1.0)
    Db = np.convolve(np.r_[Db[-2:], Db, Db[:2]], np.ones(5) / 5, "valid")
    s, dd = d["cuff"]
    p = dd + (s - dd) * (Db - Db.min()) / (Db.max() - Db.min())
    pe = np.interp(d["eval_tau"], tc, p, period=1.0)
    return metrics(d, pe, None, None)


def metrics(d, p_pred, beta, c0):
    e = (p_pred - d["eval_p"]) / MMHG
    out = dict(rmse_mmHg=float(np.sqrt(np.mean(e ** 2))), sbp_err=float((p_pred.max() - d["eval_p"].max()) / MMHG),
               dbp_err=float((p_pred.min() - d["eval_p"].min()) / MMHG),
               shape_corr=float(np.corrcoef(p_pred, d["eval_p"])[0, 1]))
    if beta is not None:
        out.update(beta_rel_err=float(beta / d["beta_true"] - 1), c0_rel_err=float(c0 / d["c0_true"] - 1))
    return out


def fit(d, level, model="B1", iters=8000, n_col=2048, seed=0, dev="cuda", lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    net = (MODELS[model](seed=seed) if model == "B2" else MODELS[model]()).to(dev)
    A_s, Tc = d["A_s"], d["Tc"]
    Q_s = 1e-5
    beta0 = 40 * MMHG / (0.10 * np.sqrt(A_s))
    # momentum residual scale: size of its pressure-gradient term for a 1% change of A across the half window
    M_mom = (Tc / Q_s) * (A_s / RHO) * beta0 / (2 * np.sqrt(A_s)) * (A_s / LX) * 1e-2
    lb = torch.zeros((), device=dev, requires_grad=True)
    la0 = torch.tensor(float(np.log(np.sqrt(d["A_min"] / A_s))), device=dev, requires_grad=True)
    opt = torch.optim.Adam(list(net.parameters()) + [lb, la0], lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1 / iters))
    zd = torch.stack([f(d["xi"]), f(d["tau"])], -1)
    Ad = f(d["A"]) / A_s
    if level == "b":
        zu = torch.stack([torch.zeros(len(d["u_tau"]), device=dev), f(d["u_tau"])], -1); ud = f(d["u"])
    zc_eval = torch.stack([torch.zeros(64, device=dev), torch.linspace(0, 1, 65, device=dev)[:-1]], -1)

    def fields(z):
        o = net(z)
        return 1.0 + 0.2 * o[:, 0], o[:, 1]                       # A_hat, Q_hat

    def pressure(Ah):
        beta = beta0 * torch.exp(lb)
        return P_REF + beta * (torch.sqrt(Ah * A_s) - np.sqrt(A_s) * torch.exp(la0)), beta

    t0 = time.time()
    for it in range(iters):
        zc = torch.rand(n_col, 2, device=dev); zc[:, 0] = 2 * zc[:, 0] - 1
        zc.requires_grad_(True)
        Ah, Qh = fields(zc)
        gA = torch.autograd.grad(Ah.sum(), zc, create_graph=True)[0]
        gQ = torch.autograd.grad(Qh.sum(), zc, create_graph=True)[0]
        g2 = torch.autograd.grad((Qh ** 2 / Ah).sum(), zc, create_graph=True)[0]
        beta = beta0 * torch.exp(lb)
        # mass: dA/dt + dQ/dx = 0, scaled by A_s/Tc
        r_mass = gA[:, 1] + (Q_s * Tc / (A_s * LX)) * gQ[:, 0]
        # momentum, scaled by Q_s/Tc
        A = Ah * A_s
        dpdx = beta / (2 * torch.sqrt(A)) * gA[:, 0] * A_s / LX
        r_mom = gQ[:, 1] + (Tc / Q_s) * ((Q_s ** 2 / (A_s * LX)) * g2[:, 0] + A / RHO * dpdx + KR * Q_s * Qh / A)
        r_mom = r_mom / M_mom
        Ahd, _ = fields(zd)
        loss_d = torch.mean(((Ahd - Ad) / SIG_A) ** 2)
        loss = loss_d + torch.mean(r_mass ** 2) + torch.mean(r_mom ** 2)
        if level == "b":
            Au, Qu = fields(zu)
            loss = loss + torch.mean(((Qu * Q_s / (Au * A_s) - ud) / SIG_U) ** 2)
        if level in ("c", "d"):
            pc, _ = pressure(fields(zc_eval)[0])
            loss = loss + ((pc.max() - d["cuff"][0]) / SIG_P) ** 2 + ((pc.min() - d["cuff"][1]) / SIG_P) ** 2
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        ze = torch.stack([torch.zeros(len(d["eval_tau"]), device=dev), f(d["eval_tau"])], -1)
        pe, beta = pressure(fields(ze)[0])
        A0 = (np.sqrt(A_s) * torch.exp(la0)) ** 2
        c0 = torch.sqrt(beta / (2 * RHO)) * A0 ** 0.25
        out = metrics(d, pe.cpu().numpy(), float(beta), float(c0))
    out.update(loss=float(loss.detach()), loss_data=float(loss_d.detach()), secs=round(time.time() - t0, 1))
    return out
