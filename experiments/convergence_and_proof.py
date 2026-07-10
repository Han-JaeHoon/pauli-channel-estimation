"""
(1) Convergence of the syndrome-only estimator as sample size grows.
(2) Numerical check of Lemma 6: the intersection matrix M_t[a,b]=2^{|a cap b|}
    is positive-definite -- the crux that makes D^T D full rank.
"""

import itertools
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pauli_syndrome import MomentEstimator, subsets_up_to
from pauli_syndrome import NoiseModel, single_bit_channel

HAMMING_H = [
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
]


# --------------------------------------------------------------------------
# Lemma 6 : intersection matrix M_t[a,b] = 2^{|a cap b|} is positive-definite
# --------------------------------------------------------------------------
def check_lemma6(nmax=8):
    print("Lemma 6 check -- min eigenvalue of intersection matrix M_t:")
    print("   n   t   size   min eig   positive-definite")
    for n in range(2, nmax + 1):
        for t in (1, 2, 3):
            if t > n:
                continue
            subs = subsets_up_to(range(n), t)
            M = np.array([[2.0 ** len(a & b) for b in subs] for a in subs])
            lam = np.linalg.eigvalsh(M).min()
            print(f"   {n}   {t}   {len(subs):4d}   {lam:8.4f}   {lam > 0}")


# --------------------------------------------------------------------------
# Convergence: estimation error vs number of syndrome rounds
# --------------------------------------------------------------------------
def convergence(seed=7):
    rng = np.random.default_rng(seed)
    n = 7
    est = MomentEstimator(HAMMING_H, subsets_up_to(range(n), 1))
    true_p = rng.uniform(0.05, 0.30, size=n)
    model = NoiseModel(n, [single_bit_channel(n, i, true_p[i]) for i in range(n)])

    Ms = np.array([200, 500, 1000, 2000, 5000, 10000, 20000,
                   50000, 100000, 200000, 500000])
    reps = 20
    mean_err = []
    for M in Ms:
        errs_rep = []
        for _ in range(reps):
            errors = model.sample(int(M), rng)
            F = est.solve(est.empirical_moments(errors))
            p_hat = est.recover_single_bit_rates(F)
            errs_rep.append(np.abs(p_hat - true_p).mean())
        mean_err.append(np.mean(errs_rep))
    mean_err = np.array(mean_err)

    # reference 1/sqrt(M) line, anchored in the well-sampled regime
    anchor = np.where(Ms == 5000)[0][0]
    ref = mean_err[anchor] * np.sqrt(Ms[anchor] / Ms)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(Ms, mean_err, "o-", lw=2, ms=6, label="syndrome-only estimator")
    ax.loglog(Ms[anchor:], ref[anchor:], "--", color="gray",
              label=r"$\propto 1/\sqrt{M}$ (method-of-moments)")
    ax.set_ylim(1e-3, 5e-2)
    ax.set_xlim(3e3, 7e5)
    ax.set_xlabel("number of syndrome rounds $M$")
    ax.set_ylabel(r"mean $|\hat p_i - p_i|$")
    ax.set_title("Estimation error vs syndrome samples  ([7,4] Hamming)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = "convergence.png"
    fig.savefig(out, dpi=130)
    print(f"\nConvergence data (mean over {reps} trials):")
    for M, e in zip(Ms, mean_err):
        print(f"   M = {M:>7,}   mean|dp| = {e:.5e}")
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    check_lemma6()
    convergence()
