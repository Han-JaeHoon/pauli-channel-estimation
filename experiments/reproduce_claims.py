"""
Reproduce the central claims of arXiv:2107.14252.

Run:  python experiments.py
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pauli_syndrome import (
    MomentEstimator,
    subsets_up_to,
    NoiseModel,
    single_bit_channel,
    pair_flip_channel,
)


# ==========================================================================
# Experiment 1 -- Toric code with independent bit-flip (X) noise.
#   Reproduces Section 2.1 / Eq.(1): estimate p_e of every edge from the
#   three stabilizer expectations of its two adjacent star operators, using
#   ONLY the syndrome statistics.  This is the Spitz-et-al. estimator.
# ==========================================================================
def build_toric(L):
    """L x L toric code.  Qubits on edges, Z-stars on vertices.
    Returns (n_edges, stars, edge_stars) where
        stars[v]       = list of edge indices in star operator v
        edge_stars[e]  = the two vertices whose stars contain edge e
    """
    def vid(i, j):
        return (i % L) * L + (j % L)

    # edge indexing: horizontal edges 0..L^2-1, vertical edges L^2..2L^2-1
    def h(i, j):
        return (i % L) * L + (j % L)

    def v(i, j):
        return L * L + (i % L) * L + (j % L)

    n = 2 * L * L
    stars = {}
    for i in range(L):
        for j in range(L):
            # star at (i,j): right & left horizontal, down & up vertical
            stars[vid(i, j)] = [h(i, j), h(i, j - 1), v(i, j), v(i - 1, j)]

    edge_stars = {}
    for i in range(L):
        for j in range(L):
            edge_stars[h(i, j)] = (vid(i, j), vid(i, j + 1))
            edge_stars[v(i, j)] = (vid(i, j), vid(i + 1, j))
    return n, stars, edge_stars


def experiment_toric(L=3, M=200_000, seed=1):
    print("=" * 74)
    print(f"EXPERIMENT 1  Toric code (L={L}), independent X noise -- Eq.(1)")
    print("=" * 74)
    rng = np.random.default_rng(seed)
    n, stars, edge_stars = build_toric(L)

    true_p = rng.uniform(0.03, 0.30, size=n)            # true edge error rates

    # --- simulate M rounds, record only the syndrome (star parities) ------
    star_ids = sorted(stars)
    star_mat = np.zeros((len(star_ids), n), dtype=int)
    for r, vtx in enumerate(star_ids):
        for e in stars[vtx]:
            star_mat[r, e] = 1

    errs = (rng.random((M, n)) < true_p).astype(int)    # X errors
    synd = (errs @ star_mat.T) % 2                       # (M, num_stars)
    star_index = {vtx: r for r, vtx in enumerate(star_ids)}

    def Emeas(cols):                                     # E(product of stars)
        parity = synd[:, cols].sum(axis=1) % 2
        return (1 - 2 * parity).mean()

    est_p = np.zeros(n)
    for e in range(n):
        v1, v2 = edge_stars[e]
        c1, c2 = star_index[v1], star_index[v2]
        E1, E2, E12 = Emeas([c1]), Emeas([c2]), Emeas([c1, c2])
        val = E1 * E2 / E12                              # = (1 - 2 p_e)^2
        EZ = np.sqrt(max(val, 0.0))                      # + sign: p < 1/2
        est_p[e] = (1 - EZ) / 2

    err = np.abs(est_p - true_p)
    print(f"  edges (qubits)         : {n}")
    print(f"  syndrome rounds (M)    : {M:,}")
    print(f"  mean |p_hat - p_true|  : {err.mean():.4e}")
    print(f"  max  |p_hat - p_true|  : {err.max():.4e}")
    print("  sample edges (true -> est):")
    for e in range(min(6, n)):
        print(f"     edge {e:2d}:  {true_p[e]:.4f}  ->  {est_p[e]:.4f}")
    return true_p, est_p


# ==========================================================================
# Experiment 2 -- General method-of-moments pipeline on the [7,4] Hamming
#   code with independent single-bit noise.  Builds the coefficient matrix
#   D, checks identifiability (Cor. 4: d=3 >= 2*1+1), solves the binomial
#   system and recovers all 7 error rates from syndrome statistics.
# ==========================================================================
HAMMING_H = [
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
]


def experiment_hamming(M=200_000, seed=2):
    print()
    print("=" * 74)
    print("EXPERIMENT 2  [7,4] Hamming code, single-bit noise -- full pipeline")
    print("=" * 74)
    rng = np.random.default_rng(seed)
    n = 7
    gamma_hat = subsets_up_to(range(n), 1)              # single-bit supports

    est = MomentEstimator(HAMMING_H, gamma_hat)
    print(f"  |C^perp \\ 0|           : {len(est.dual)}")
    print(f"  |Gamma_hat| (unknowns) : {len(gamma_hat)}")
    print(f"  rank(D)                : {est.rank()}  "
          f"-> identifiable: {est.is_identifiable()}  (Cor.4: d=3 >= 2t+1)")

    true_p = rng.uniform(0.02, 0.30, size=n)
    model = NoiseModel(n, [single_bit_channel(n, i, true_p[i]) for i in range(n)])

    errors = model.sample(M, rng)
    E_obs = est.empirical_moments(errors)
    F = est.solve(E_obs)
    est_p = est.recover_single_bit_rates(F)

    err = np.abs(est_p - true_p)
    print(f"  syndrome rounds (M)    : {M:,}")
    print(f"  mean |p_hat - p_true|  : {err.mean():.4e}")
    print("  qubit  true_p   est_p")
    for i in range(n):
        print(f"    {i}    {true_p[i]:.4f}   {est_p[i]:.4f}")
    return true_p, est_p


# ==========================================================================
# Experiment 3 -- Correlated noise and the distance threshold d >= 2t+1.
#   (a) rank table: which codes make 2-qubit-correlated noise identifiable.
#   (b) recover an actual 2-qubit correlation on a code with d=5.
# ==========================================================================
def repetition_H(n):
    """Parity-check of the [n,1,n] repetition code: consecutive-bit parities.
    Dual code = even-weight vectors (the [n,n-1] parity code)."""
    H = np.zeros((n - 1, n), dtype=int)
    for i in range(n - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


def experiment_correlated(M=400_000, seed=3):
    print()
    print("=" * 74)
    print("EXPERIMENT 3  Correlated (t=2) noise & threshold d >= 2t+1 (Cor.4)")
    print("=" * 74)

    print("  (a) Identifiability vs code distance for t=2 correlations:")
    print("      code            n   d   |Gamma_hat|  rank(D)  identifiable")
    for n in (3, 4, 5, 6, 7):
        H = repetition_H(n)                 # [n,1,n] repetition code, d = n
        gamma_hat = subsets_up_to(range(n), 2)
        est = MomentEstimator(H, gamma_hat)
        ok = est.is_identifiable()
        need = "yes" if n >= 5 else "no "   # d = n >= 2*2+1 = 5
        print(f"      repetition[{n}]   {n}   {n}      {len(gamma_hat):3d}"
              f"        {est.rank():3d}      {str(ok):5s}  (expect {need})")

    print()
    print("  (b) Recovering a real 2-qubit correlation (repetition[5], d=5):")
    rng = np.random.default_rng(seed)
    n = 5
    H = repetition_H(n)
    # Gamma = singletons + the correlated pair {0,1}
    gamma_hat = [frozenset([i]) for i in range(n)] + [frozenset([0, 1])]
    est = MomentEstimator(H, gamma_hat)
    print(f"      rank(D) = {est.rank()} / {len(gamma_hat)}  "
          f"-> identifiable: {est.is_identifiable()}")

    true_p = np.array([0.10, 0.15, 0.08, 0.12, 0.05])
    q = 0.18                                            # 2-qubit corr. strength
    channels = [single_bit_channel(n, i, true_p[i]) for i in range(n)]
    channels.append(pair_flip_channel(n, 0, 1, q))
    model = NoiseModel(n, channels)

    errors = model.sample(M, rng)
    E_obs = est.empirical_moments(errors)
    F = est.solve(E_obs)

    # ground-truth transformed moments for comparison
    def true_F(a):
        # F(a) = prod_{b subseteq a} E(b)^{(-1)^{|a|-|b|}}   (inclusion-excl.)
        a = sorted(a)
        val = 1.0
        for r in range(len(a) + 1):
            for b in itertools.combinations(a, r):
                sign = (-1) ** (len(a) - len(b))
                vec = np.zeros(n, dtype=int)
                for i in b:
                    vec[i] = 1
                val *= model.true_moment(vec) ** sign
        return val

    print("      support        F_true     F_est")
    for a in gamma_hat:
        print(f"      {str(sorted(a)):12s}  {true_F(a):8.4f}   {F[a]:8.4f}")

    # translate the pair's transformed moment back to q:  F({0,1}) = (1-2q)^-2
    F01 = F[frozenset([0, 1])]
    q_est = (1 - 1 / np.sqrt(F01)) / 2
    print(f"      correlation q:  true = {q:.4f}   estimated = {q_est:.4f}")

    # contrast: if we WRONGLY assume single-bit-only noise, the pair
    # correlation is invisible and single-rate estimates are biased.
    est_sb = MomentEstimator(H, subsets_up_to(range(n), 1))
    F_sb = est_sb.solve(est_sb.empirical_moments(errors))
    p_sb = est_sb.recover_single_bit_rates(F_sb)
    print("      (misspecified single-bit model ignores the correlation:)")
    print(f"        p_hat[0,1] = {p_sb[0]:.4f}, {p_sb[1]:.4f}"
          f"   (marginal true = {(1-model.true_moment([1,0,0,0,0]))/2:.4f},"
          f" {(1-model.true_moment([0,1,0,0,0]))/2:.4f})")


if __name__ == "__main__":
    experiment_toric()
    experiment_hamming()
    experiment_correlated()
    print()
    print("Done.")
