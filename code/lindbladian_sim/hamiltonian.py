import numpy as np


def comm(a, b):
    """
    Evaluate the commutator [a, b].
    """
    return a @ b - b @ a


def hamiltonian_superoperator(H):
    """
    Construct the superoperator representation of the Hamiltonian commutator.
    """
    H = np.asarray(H)
    n = H.shape[0]
    return -1j * (np.kron(H, np.identity(n)) - np.kron(np.identity(n), H.T))
