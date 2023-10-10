import numpy as np


def anti_comm(a, b):
    """
    Evaluate the anti-commutator {a, b}.
    """
    return a @ b + b @ a


def apply_lindblad_operator(L, rho):
    """
    Apply the Lindblad jump operator 'L' to the density matrix 'rho'.
    """
    return L @ rho @ L.conj().T - 0.5 * anti_comm(L.conj().T @ L, rho)


def lindblad_operator_matrix(L):
    """
    Construct the matrix ("superoperator") representation of a Lindblad jump operator.
    """
    L = np.asarray(L)
    n = L.shape[0]
    return (np.kron(L, L.conj())
            - 0.5 * (np.kron(L.conj().T @ L, np.identity(n))
                   + np.kron(np.identity(n), L.T @ L.conj())))


def apply_lindbladian(Llist, rho):
    """
    Apply a Lindbladian (consisting of several jump operators) to the density matrix 'rho'.
    """
    return sum(apply_lindblad_operator(L, rho) for L in Llist)


def lindbladian_matrix(Llist):
    """
    Construct the matrix ("superoperator") representation of a Lindbladian.
    """
    return sum(lindblad_operator_matrix(L) for L in Llist)
