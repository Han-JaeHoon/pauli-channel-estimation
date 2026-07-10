"""
Reproduction of the core method from:

    Wagner, Kampermann, Bruss, Kliesch,
    "Pauli channels can be estimated from syndrome measurements in
     quantum error correction", Quantum (2022), arXiv:2107.14252.

General classical method-of-moments estimator (Sections 2.2, 2.4).

Idea
----
An error distribution P over F_2^n is described by its Fourier coefficients
(= moments)  E(a) = sum_e (-1)^{a.e} P(e).  For s in the dual code C^perp
(= row span of the parity-check matrix H) the moment E(s) equals the
expectation value of the corresponding parity/stabilizer measurement, so it
is directly observable from the SYNDROME STATISTICS.

If the noise is a convolution of independent channels supported on the sets
in Gamma, the distribution is fully described by a few "transformed moments"
F(a), a in Gamma_hat, related to the observable moments by the *binomial*
system

        E(s) = prod_{a subseteq s, a in Gamma_hat} F(a)      (Eq. 18)

Taking logs turns this into a linear system  D . logF = log E , where D is
the coefficient matrix  D[s,a] = 1 iff a subseteq s.  If D has full column
rank (Thm 3 / Cor 4: guaranteed when the code distance d >= 2t+1 for
t-qubit correlations) the transformed moments -- and hence all error rates
-- are identifiable up to signs, and uniquely if all rates < 1/2.
"""

import itertools
import numpy as np


# --------------------------------------------------------------------------
# Linear-algebra helpers over F_2
# --------------------------------------------------------------------------
def row_span(H):
    """All 2^rank vectors in the row span of a binary matrix H (over F_2)."""
    H = np.array(H, dtype=int) % 2
    basis = []
    for row in H:
        r = row.copy()
        for b in basis:
            if r[np.argmax(b)] and np.array_equal(np.maximum(r, 0), r):
                pass
        basis.append(r)
    # Simpler & robust: enumerate all F_2 combinations of the rows, dedupe.
    rows = [r for r in H]
    span = set()
    k = len(rows)
    for mask in range(2 ** k):
        v = np.zeros(H.shape[1], dtype=int)
        for i in range(k):
            if (mask >> i) & 1:
                v = (v + rows[i]) % 2
        span.add(tuple(v))
    return np.array(sorted(span))


def subsets_up_to(elements, tmax):
    """All non-empty subsets of `elements` of size 1..tmax (as frozensets)."""
    out = []
    for r in range(1, tmax + 1):
        for c in itertools.combinations(elements, r):
            out.append(frozenset(c))
    return out


def set_to_vec(s, n):
    v = np.zeros(n, dtype=int)
    for i in s:
        v[i] = 1
    return v


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------
class MomentEstimator:
    """Method-of-moments estimator for a classical linear code.

    Parameters
    ----------
    H     : (n-k, n) binary parity-check matrix.  C^perp = row span of H.
    gamma_hat : list of frozensets -- the set Gamma_hat of "independently
                occurring" error supports (Eq. 16).  For single-bit noise
                this is {{0},...,{n-1}}; for t-correlated noise it is every
                subset of size <= t.
    """

    def __init__(self, H, gamma_hat):
        self.H = np.array(H, dtype=int) % 2
        self.n = self.H.shape[1]
        self.gamma_hat = list(gamma_hat)

        # dual code C^perp minus the zero element -> observable moments
        span = row_span(self.H)
        self.dual = np.array([s for s in span if s.any()])

        # coefficient matrix  D[s,a] = 1 iff a subseteq s          (Eq. 19)
        self.D = np.zeros((len(self.dual), len(self.gamma_hat)), dtype=float)
        for i, s in enumerate(self.dual):
            s_idx = set(np.nonzero(s)[0])
            for j, a in enumerate(self.gamma_hat):
                self.D[i, j] = 1.0 if a.issubset(s_idx) else 0.0

    # -- identifiability diagnostics ---------------------------------------
    def rank(self):
        return np.linalg.matrix_rank(self.D)

    def is_identifiable(self):
        return self.rank() == len(self.gamma_hat)

    # -- estimation --------------------------------------------------------
    def empirical_moments(self, errors):
        """Observable moments E(s), s in C^perp, from raw error samples.

        Only the syndrome s.e (mod 2) of each dual codeword is used, i.e.
        exactly the information available from syndrome measurements.
        """
        errs = np.array(errors, dtype=int) % 2          # (M, n)
        # parity of each dual codeword on each sample: (num_dual, M)
        parity = (self.dual @ errs.T) % 2
        signs = 1.0 - 2.0 * parity                       # (-1)^parity
        return signs.mean(axis=1)                        # E_hat(s)

    def solve(self, E_obs):
        """Solve the (log-linearised) binomial system for F, then rates.

        Assumes all moments positive (all channel error probs < 1/2), so we
        may take logs directly.  Returns dict {frozenset: F(a)}.
        """
        # guard against empirical zeros/negatives from finite sampling
        E_clip = np.clip(E_obs, 1e-9, None)
        logF, *_ = np.linalg.lstsq(self.D, np.log(E_clip), rcond=None)
        F = np.exp(logF)
        return {a: F[j] for j, a in enumerate(self.gamma_hat)}

    def recover_single_bit_rates(self, F):
        """For single-bit supports, E({i}) = F({i}) = 1 - 2 p_i."""
        rates = np.zeros(self.n)
        for a, val in F.items():
            if len(a) == 1:
                (i,) = tuple(a)
                rates[i] = (1.0 - val) / 2.0
        return rates
