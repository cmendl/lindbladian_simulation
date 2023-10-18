import numpy as np
from scipy.special import erf


def construct_shoveling_lindblad_operator(A, H, f):
    """
    Construct the "shoveling" Lindblad operator, Eq. (4).

    Reference:
        Zhiyan Ding, Chi-Fang (Anthony) Chen, Lin Lin
        Single-ancilla ground state preparation via Lindbladians
        arXiv:2308.15676
    """
    A = np.asarray(A)
    H = np.asarray(H)

    # diagonalize Hamiltonian
    w, v = np.linalg.eigh(H)

    # construct K operator, Eq. (4)
    return sum((f(w[i] - w[j]) * np.vdot(v[:, i], A @ v[:, j]))
               * np.outer(v[:, i], v[:, j].conj())
                   for i in range(len(w))
                   for j in range(len(w)))


def filter_function(w: float, a: float, da: float, b: float, db: float):
    """
    Filter function in Fourier representation, Eq. (E2).

    Reference:
        Zhiyan Ding, Chi-Fang (Anthony) Chen, Lin Lin
        Single-ancilla ground state preparation via Lindbladians
        arXiv:2308.15676
    """
    return 0.5 * (erf((w + a) / da) - erf((w + b) / db))
