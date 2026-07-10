"""Ground-truth noise models = convolutions of independent channels."""

import numpy as np


class Channel:
    """An independent error source acting on a subset `support` of bits.

    `patterns` : list of binary error patterns (length n) it can produce.
    `probs`    : matching probabilities (must sum to 1, patterns[0] = 0).
    """

    def __init__(self, patterns, probs):
        self.patterns = np.array(patterns, dtype=int) % 2
        self.probs = np.array(probs, dtype=float)
        assert abs(self.probs.sum() - 1.0) < 1e-9

    def sample(self, M, rng):
        idx = rng.choice(len(self.probs), size=M, p=self.probs)
        return self.patterns[idx]                       # (M, n)


def single_bit_channel(n, i, p):
    """Bit i flips with probability p."""
    z = np.zeros(n, dtype=int)
    e = z.copy(); e[i] = 1
    return Channel([z, e], [1 - p, p])


def pair_flip_channel(n, i, j, q):
    """Bits i and j flip TOGETHER with probability q (correlated source)."""
    z = np.zeros(n, dtype=int)
    e = z.copy(); e[i] = 1; e[j] = 1
    return Channel([z, e], [1 - q, q])


class NoiseModel:
    """Total error = XOR (Boolean convolution) of independent channels."""

    def __init__(self, n, channels):
        self.n = n
        self.channels = channels

    def sample(self, M, rng):
        tot = np.zeros((M, self.n), dtype=int)
        for ch in self.channels:
            tot = (tot + ch.sample(M, rng)) % 2
        return tot

    def true_moment(self, a):
        """Exact E(a) = prod over channels of E_channel(a)."""
        a = np.array(a, dtype=int) % 2
        val = 1.0
        for ch in self.channels:
            parity = (ch.patterns @ a) % 2
            val *= float((ch.probs * (1 - 2 * parity)).sum())
        return val
