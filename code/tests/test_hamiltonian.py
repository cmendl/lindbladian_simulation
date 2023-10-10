import unittest
import numpy as np
import lindbladian_sim as lbs


class TestHamiltonian(unittest.TestCase):

    def test_hamiltonian_superoperator(self):

        rng = np.random.default_rng()

        # system size
        d = 5

        # fictitious density matrix
        rho = lbs.random_density_matrix(d, rng)

        # Hamiltonian
        H = lbs.crandn((d, d), rng)
        H = 0.5 * (H + H.conj().T)

        # direct application of Lindblad operator must match superoperator representation
        self.assertTrue(np.allclose(
            -1j * lbs.comm(H, rho),
            np.reshape(lbs.hamiltonian_superoperator(H) @ rho.reshape(-1), rho.shape)))


if __name__ == '__main__':
    unittest.main()
