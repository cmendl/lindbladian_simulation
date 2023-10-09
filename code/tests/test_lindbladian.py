import unittest
import numpy as np
import lindbladian_sim as lbs


class TestLindbladian(unittest.TestCase):

    def test_lindblad_operator_matrix(self):

        rng = np.random.default_rng()

        # system size
        d = 5

        # fictitious density matrix
        rho = lbs.crandn((d, d), rng)
        rho = 0.5 * (rho + rho.conj().T)

        # Lindblad jump operator
        L = lbs.crandn((d, d), rng)

        # direct application of Lindblad operator must match superoperator representation
        self.assertTrue(np.allclose(
            lbs.apply_lindblad_operator(L, rho),
            np.reshape(lbs.lindblad_operator_matrix(L) @ rho.reshape(-1), rho.shape)))


    def test_lindbladian_matrix(self):

        rng = np.random.default_rng()

        # system size
        d = 5
        # number of operators
        numops = 4

        # fictitious density matrix
        rho = lbs.crandn((d, d), rng)
        rho = 0.5 * (rho + rho.conj().T)

        # Lindblad jump operators
        Llist = [lbs.crandn((d, d), rng) for _ in range(numops)]

        # direct application of Lindbladian must match superoperator representation
        self.assertTrue(np.allclose(
            lbs.apply_lindbladian(Llist, rho),
            np.reshape(lbs.lindbladian_matrix(Llist) @ rho.reshape(-1), rho.shape)))


if __name__ == '__main__':
    unittest.main()
