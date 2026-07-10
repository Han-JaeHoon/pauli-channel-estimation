# Estimating Pauli channels from syndrome measurements

A self-contained reproduction of the central results of

> T. Wagner, H. Kampermann, D. Bruß, M. Kliesch,
> **"Pauli channels can be estimated from syndrome measurements in quantum
> error correction"**, *Quantum* **6**, 809 (2022), [arXiv:2107.14252](https://arxiv.org/abs/2107.14252).

The paper's claim: the error rates of a (correlated) Pauli channel can be
identified **from the syndrome statistics alone** — the measurements already
performed during quantum error correction — without ever measuring the logical
state. Identifiability is governed by the code's (pure) distance: a code can
estimate noise correlated across up to `⌊(d_p − 1)/2⌋` qubits.

This repo implements the method from scratch (`numpy` only) and verifies every
claim by Monte-Carlo simulation.

## The method in one paragraph

An error distribution `P` over `F_2^n` is described by its Fourier coefficients,
the **moments** `E(a) = Σ_e (−1)^{a·e} P(e)`. For a dual codeword `s ∈ C⊥`
(a stabilizer), `E(s)` is exactly the expectation value of that stabilizer
measurement, so it is **observable from the syndrome statistics**. If the noise
is a convolution of independent channels supported on the sets in `Γ`, the whole
distribution is fixed by a few *transformed moments* `F(a)` obeying the
**binomial system**

```
E(s) = ∏_{a ⊆ s, a ∈ Γ̂} F(a)          (Eq. 18)
```

Taking logs linearises it: `D · log F = log E`, with `D[s,a] = 1 ⇔ a ⊆ s`. When
`D` has full column rank — guaranteed by `d ≥ 2t + 1` for `t`-qubit correlations
(Cor. 4) — the rates are identifiable, and unique if all rates `< 1/2`.

## Layout

```
pauli_syndrome/
  moment_estimator.py   # MomentEstimator: builds D, solves the binomial system
  noise.py              # ground-truth noise = convolution of independent channels
experiments/
  reproduce_claims.py         # Experiments 1–3 (see below)
  convergence_and_proof.py    # 1/√M convergence + Lemma 6 (positive-definiteness)
```

## Running

```bash
pip install -r requirements.txt
python experiments/reproduce_claims.py
python experiments/convergence_and_proof.py   # also writes convergence.png
```

## What is reproduced

**Experiment 1 — Toric code, independent X noise (Sec. 2.1, Eq. 1).**
Every edge rate on a 3×3 toric code is recovered from 200k syndrome rounds via
`E(Z_e) = √(E(S₁)E(S₂)/E(S₁S₂))` (the Spitz-et-al. estimator, a special case of
the general system). Mean error ≈ 3×10⁻³.

**Experiment 2 — [7,4] Hamming code, full pipeline.**
The coefficient matrix `D` is 7×7 full rank (Cor. 4: `d = 3 ≥ 2t+1`); solving the
binomial system recovers all 7 single-qubit rates (mean error ≈ 1×10⁻³).

**Experiment 3 — correlated noise & the threshold `d ≥ 2t+1` (Cor. 4).**
For `t = 2` correlations, repetition codes with `d < 5` give a rank-deficient `D`
(not identifiable) while `d ≥ 5` are full rank. A genuine 2-qubit correlation
(`q = 0.18`) is then recovered from syndromes (`q̂ ≈ 0.182`), whereas a
misspecified single-qubit model absorbs it into biased marginal rates.

**Convergence & proof core.** The estimation error scales as `1/√M`
(method-of-moments), and the intersection matrix `Mₜ[a,b] = 2^{|a∩b|}` is verified
positive-definite for all tested `n, t` — the linear-algebra crux (Lemma 6)
behind `DᵀD` having full rank.