"""PulseNS L1: nonlinear 1D blood flow in a compliant vessel.

Equations (SI units), U = [A, Q]:
    dA/dt + dQ/dx = 0
    dQ/dt + d(Q^2/A)/dx + (A/rho) dp/dx = -K_R Q/A,   K_R = 8*pi*mu/rho (Poiseuille profile, alpha = 1)
Tube law (beta and A0 uniform along the segment):
    p = p_ref + beta * (sqrt(A) - sqrt(A0))
    so (A/rho) dp/dx = d/dx( beta/(3 rho) * A^1.5 ), and the momentum flux is Q^2/A + beta/(3 rho) A^1.5
Wave speed c = sqrt(beta/(2 rho)) * A^(1/4); Riemann invariants W1,2 = u +- 4c.

Scheme: two-step Lax-Wendroff (Richtmyer), 2nd order in space and time.
Boundaries: characteristic extrapolation.
    inlet  = prescribed flow Q_in(t)
    outlet = 3-element Windkessel (R1, C, R2) or pure resistance (C = 0)
"""
import numpy as np


class Vessel:
    def __init__(self, L, A0, beta, rho=1060.0, mu=0.0035, p_ref=0.0):
        self.L, self.A0, self.beta, self.rho, self.mu, self.p_ref = L, A0, beta, rho, mu, p_ref
        self.k = np.sqrt(beta / (2.0 * rho))      # c = k * A^(1/4)
        self.KR = 8.0 * np.pi * mu / rho

    def c(self, A):
        return self.k * A ** 0.25

    def p(self, A):
        return self.p_ref + self.beta * (np.sqrt(A) - np.sqrt(self.A0))

    def A_of_p(self, p):
        return (np.sqrt(self.A0) + (p - self.p_ref) / self.beta) ** 2

    @property
    def c0(self):
        return self.c(self.A0)

    @property
    def Z0(self):  # characteristic impedance
        return self.rho * self.c0 / self.A0

    def flux(self, A, Q):
        return Q, Q * Q / A + self.beta / (3.0 * self.rho) * A ** 1.5

    def source(self, A, Q):
        return 0.0 * A, -self.KR * Q / A


def _newton(f, x0, tol=1e-13, it=50):
    x = x0
    for _ in range(it):
        h = 1e-7 * x
        fx = f(x)
        x_new = x - fx * h / (f(x + h) - fx)
        if abs(x_new - x) < tol * x:
            return x_new
        x = x_new
    return x


class Solver:
    """outlet: dict(R1=..., C=..., R2=..., p_out=0.0); C=0 -> pure resistance R1."""

    def __init__(self, vessel, N, Q_in, outlet, cfl=0.5, A_init=None, Q_init=None, mms_source=None, exact=None):
        self.v, self.N, self.Q_in, self.out = vessel, N, Q_in, dict(outlet)
        self.x = np.linspace(0.0, vessel.L, N)
        self.dx = self.x[1] - self.x[0]
        self.A = np.full(N, vessel.A0) if A_init is None else np.array(A_init, float)
        self.Q = np.zeros(N) if Q_init is None else np.array(Q_init, float)
        self.cfl, self.t = cfl, 0.0
        self.pc = self.out.get("pc0", self.v.p(self.A[-1]))  # Windkessel capacitor pressure
        self.mms_source, self.exact = mms_source, exact     # verification only (Dirichlet ends from exact)

    def dt_cfl(self):
        u = self.Q / self.A
        return self.cfl * self.dx / np.max(np.abs(u) + self.v.c(self.A))

    def _S(self, A, Q, x, t):
        s0, s1 = self.v.source(A, Q)
        if self.mms_source is not None:
            m0, m1 = self.mms_source(x, t)
            s0, s1 = s0 + m0, s1 + m1
        return s0, s1

    def step(self, dt):
        v, A, Q, x, t, dx = self.v, self.A, self.Q, self.x, self.t, self.dx
        F0, F1 = v.flux(A, Q)
        S0, S1 = self._S(A, Q, x, t)
        xh = 0.5 * (x[1:] + x[:-1])
        Ah = 0.5 * (A[1:] + A[:-1]) - dt / (2 * dx) * (F0[1:] - F0[:-1]) + dt / 4 * (S0[1:] + S0[:-1])
        Qh = 0.5 * (Q[1:] + Q[:-1]) - dt / (2 * dx) * (F1[1:] - F1[:-1]) + dt / 4 * (S1[1:] + S1[:-1])
        G0, G1 = v.flux(Ah, Qh)
        T0, T1 = self._S(Ah, Qh, xh, t + dt / 2)
        An, Qn = A.copy(), Q.copy()
        An[1:-1] = A[1:-1] - dt / dx * (G0[1:] - G0[:-1]) + dt / 2 * (T0[1:] + T0[:-1])
        Qn[1:-1] = Q[1:-1] - dt / dx * (G1[1:] - G1[:-1]) + dt / 2 * (T1[1:] + T1[:-1])
        tn = t + dt
        if self.exact is not None:
            for i in (0, -1):
                An[i], Qn[i] = self.exact(x[i], tn)
        else:
            self._bc_inlet(An, Qn, dt, tn)
            self._bc_outlet(An, Qn, dt)
        self.A, self.Q, self.t = An, Qn, tn

    def _foot(self, i, j, lam, dt):
        """Interpolate state at the foot of the characteristic from boundary node i towards interior node j."""
        s = min(abs(lam) * dt / self.dx, 1.0)
        A = (1 - s) * self.A[i] + s * self.A[j]
        Q = (1 - s) * self.Q[i] + s * self.Q[j]
        return A, Q

    def _bc_inlet(self, An, Qn, dt, tn):
        v = self.v
        A0, Q0 = self.A[0], self.Q[0]
        lam2 = Q0 / A0 - v.c(A0)
        Af, Qf = self._foot(0, 1, lam2, dt)
        W2 = Qf / Af - 4 * v.c(Af)
        q = self.Q_in(tn)
        An[0] = _newton(lambda a: q / a - 4 * v.c(a) - W2, A0)
        Qn[0] = q

    def _bc_outlet(self, An, Qn, dt):
        v, o = self.v, self.out
        A1, Q1 = self.A[-1], self.Q[-1]
        lam1 = Q1 / A1 + v.c(A1)
        Af, Qf = self._foot(-1, -2, lam1, dt)
        W1 = Qf / Af + 4 * v.c(Af)
        R1, C, R2, pout = o["R1"], o.get("C", 0.0), o.get("R2", 0.0), o.get("p_out", 0.0)
        pc = self.pc if C > 0 else pout
        a = _newton(lambda a: a * (W1 - 4 * v.c(a)) - (v.p(a) - pc) / R1, A1)
        An[-1] = a
        Qn[-1] = a * (W1 - 4 * v.c(a))
        if C > 0:
            self.pc += dt * (Qn[-1] - (self.pc - pout) / R2) / C

    def run(self, T_end, probes=(), every=1, dt=None):
        """Advance to T_end. probes = x positions to record (A, Q, p). Returns dict of arrays."""
        idx = [int(round(p / self.dx)) for p in probes]
        rec = dict(t=[], A=[], Q=[], p=[], vol=[], qin=[], qout=[])
        n = 0
        while self.t < T_end - 1e-15:
            h = min(dt or self.dt_cfl(), T_end - self.t)
            self.step(h)
            n += 1
            if n % every == 0 or self.t >= T_end - 1e-15:
                rec["t"].append(self.t)
                rec["A"].append(self.A[idx]); rec["Q"].append(self.Q[idx]); rec["p"].append(self.v.p(self.A[idx]))
                rec["vol"].append(np.trapezoid(self.A, self.x)); rec["qin"].append(self.Q[0]); rec["qout"].append(self.Q[-1])
        return {k: np.array(val) for k, val in rec.items()}
