import unittest

from spheroidal import *
from spheroidal_particles import *
from properties import *

#main integration tests for non-absorbing particle
class testSVM(unittest.TestCase):

    def test_IntegrationCase(self):
        nmax = 10
        c1 = 1.3
        c2 = 2
        alpha = pi / 4
        k = 1
        particle = ProlateSpheroid(psi=2,c=c1,derivative=0,eps=1)
        b_sca = getSolution(particle, TMInputWave(alpha), c1, c2, nmax)[0]
        C_ext = -getCext(particle, alpha, k, b_sca, nmax)[0]
        C_sca = getCsca(k, b_sca, nmax)[0]
        delta = (C_ext-C_sca)/(C_ext+C_sca)
        self.assertAlmostEqual(delta,0,5)
        self.assertAlmostEqual(C_ext,C_sca,5)
        self.assertTrue(C_ext>0.)
        self.assertTrue(C_sca>0.)

if __name__ == '__main__':
    #import cProfile
    #cProfile.run('unittest.main()','profiler_results')
    unittest.main()


  