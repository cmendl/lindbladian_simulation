import numpy as np


def crandn(size=None, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


def random_density_matrix(d: int, rng: np.random.Generator):
    """
    Construct a random density matrix.
    """
    rho = crandn((d, d), rng)
    rho = rho @ rho.conj().T
    rho /= np.trace(rho)
    return rho
