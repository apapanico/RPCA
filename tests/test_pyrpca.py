# -*- coding: utf-8 -*-

"""
test_pyrpca
----------------------------------

Tests for `pyrpca` module.
"""

import unittest

from .context import pyrpca

import numpy as np

rtol = 1e-5
atol = 1e-7

test_matrix = np.loadtxt("tests/data/test_matrix.csv",delimiter=',')
test_matrix_L = np.loadtxt("tests/data/test_matrix_L.csv",delimiter=',')
test_matrix_S = np.loadtxt("tests/data/test_matrix_S.csv",delimiter=',')

larger_test_matrix = np.loadtxt("tests/data/larger_test_matrix.csv",delimiter=',')
larger_test_matrix_L = np.loadtxt("tests/data/larger_test_matrix_L.csv",delimiter=',')
larger_test_matrix_S = np.loadtxt("tests/data/larger_test_matrix_S.csv",delimiter=',')

tau = .7
sv = 5
expected_r = 4
shrink_matrix = np.array([[-0.18478, -0.37229,  0.37837,  0.30094,  0.45731,  0.32194],
       [-0.20504, -0.48457, -0.06991, -0.44309,  0.48167,  0.27382],
       [-0.27867, -0.31667,  0.23383, -0.21251,  0.02528,  0.20926],
       [-0.23485, -0.48691, -0.14035,  0.21444, -0.1288 , -0.15544],
       [-0.4933 ,  0.00614,  0.31232, -0.44262, -0.2111 , -0.1911 ],
       [ 0.03347, -0.31605, -0.20282, -0.20231,  0.35343, -0.01137],
       [ 0.00089, -0.20698,  0.12799,  0.28986,  0.34088, -0.05929],
       [ 0.07821, -0.09166, -0.36637, -0.20773,  0.20448, -0.27406],
       [ 0.49957, -0.40724, -0.11624, -0.28267, -0.48699, -0.27938],
       [-0.16146,  0.05043, -0.45943, -0.14569,  0.11812,  0.42471]])

shrunk_matrix = np.array([[-0.13499583, -0.13626589,  0.10894484,  0.08382217,  0.22734707, 0.13642911],
       [-0.13019554, -0.22657192, -0.0270351 , -0.12759786,  0.21484689, 0.11736671],
       [-0.11282303, -0.11717745,  0.05949101, -0.06438603,  0.08248535, 0.05405519],
       [-0.00561618, -0.11268731,  0.01369264, -0.01794372,  0.02484353, -0.01206326],
       [-0.10131444, -0.02556668,  0.07805214, -0.11481419, -0.06539163, -0.01459519],
       [-0.02260088, -0.13871069, -0.06090765, -0.0684355 ,  0.11304767, 0.04479198],
       [-0.02892052, -0.07085122,  0.03913939,  0.07290316,  0.10978982, 0.05297923],
       [ 0.04524661, -0.05840063, -0.0976409 , -0.07342859,  0.00991532, -0.01605863],
       [ 0.13383438, -0.06561628, -0.08996525, -0.10471575, -0.17649834, -0.15706847],
       [-0.03321871, -0.01658297, -0.07597524, -0.05753887,  0.10368384, 0.07390201]])
shrunk_vector = np.array([[-0.     , -0.07229,  0.07837,  0.00094,  0.15731,  0.02194],
				       [-0.     , -0.18457, -0.     , -0.14309,  0.18167,  0.     ],
				       [-0.     , -0.01667,  0.     , -0.     ,  0.     ,  0.     ],
				       [-0.     , -0.18691, -0.     ,  0.     , -0.     , -0.     ],
				       [-0.1933 ,  0.     ,  0.01232, -0.14262, -0.     , -0.     ],
				       [ 0.     , -0.01605, -0.     , -0.     ,  0.05343, -0.     ],
				       [ 0.     , -0.     ,  0.     ,  0.     ,  0.04088, -0.     ],
				       [ 0.     , -0.     , -0.06637, -0.     ,  0.     , -0.     ],
				       [ 0.19957, -0.10724, -0.     , -0.     , -0.18699, -0.     ],
				       [-0.     ,  0.     , -0.15943, -0.     ,  0.     ,  0.12471]])




class Test_rpca_alm(unittest.TestCase):

	def setUp(self):
		self.vector = np.array([0.97524828,  0.22805508, -1.09904465,  1.01910579, -0.77785785])
		self.matrix = shrink_matrix
		self.tau = tau
		self.sv = sv

		self.test_matrix = test_matrix
		self.larger_test_matrix = larger_test_matrix


	def test_vector_shrink(self):
		tau = .3
		expected = shrunk_vector
		out = expected.copy()

		pyrpca.vector_shrink(self.matrix, tau, out=out)
		np.testing.assert_allclose(out,expected)

	def test_matrix_shrink1(self):
		IN = self.matrix.copy()
		OUT = shrunk_matrix.copy()
		
		OUT,r = pyrpca.matrix_shrink(IN,self.tau,self.sv,out=OUT)

		np.testing.assert_allclose(OUT,shrunk_matrix,rtol=rtol,atol=atol)
		np.testing.assert_allclose(r,expected_r)

	def test_rpca_alm(self):
		expected_L = test_matrix_L
		expected_S = test_matrix_S
		expected_niter = 355
		L,S,niter = pyrpca.rpca_alm(self.test_matrix,verbose=False)

		np.testing.assert_allclose(L,expected_L,rtol=rtol,atol=atol)
		np.testing.assert_allclose(S,expected_S,rtol=rtol,atol=atol)
		np.testing.assert_allclose(niter,expected_niter)

	def test_rpca_alm_large(self):
		expected_L = larger_test_matrix_L
		expected_S = larger_test_matrix_S
		L,S,niter = pyrpca.rpca_alm(self.larger_test_matrix,verbose=False)

		np.testing.assert_allclose(L,expected_L,rtol=rtol,atol=atol)
		np.testing.assert_allclose(S,expected_S,rtol=rtol,atol=atol)

	def tearDown(self):
		del self.vector
		del self.matrix
		del self.test_matrix
		del self.larger_test_matrix

class Test_rpca_alm_old(unittest.TestCase):

	def setUp(self):
		self.vector = np.array([0.97524828,  0.22805508, -1.09904465,  1.01910579, -0.77785785])
		self.matrix = shrink_matrix
		self.tau = tau
		self.sv = sv

		self.test_matrix = test_matrix
		self.larger_test_matrix = larger_test_matrix


	def test_vector_shrink(self):
		tau = .5
		expected = np.array([0.47524828,  0.0, -0.59904465,  0.51910579, -0.27785785])
		result = pyrpca.vector_shrink_old(self.vector,tau)
		np.testing.assert_allclose(result,expected)

	def test_matrix_shrink1(self):
		Y,r = pyrpca.matrix_shrink_old(self.matrix,self.tau,self.sv)

		np.testing.assert_allclose(Y,shrunk_matrix,rtol=rtol,atol=atol)
		np.testing.assert_allclose(r,expected_r,rtol=rtol,atol=atol)

	def test_rpca_alm(self):
		expected_L = test_matrix_L
		expected_S = test_matrix_S
		expected_niter = 355
		L,S,niter = pyrpca.rpca_alm_old(self.test_matrix)

		np.testing.assert_allclose(L,expected_L,rtol=rtol,atol=atol)
		np.testing.assert_allclose(S,expected_S,rtol=rtol,atol=atol)
		np.testing.assert_allclose(niter,expected_niter)

	def test_rpca_alm_large(self):
		expected_L = larger_test_matrix_L
		expected_S = larger_test_matrix_S
		L,S,niter = pyrpca.rpca_alm_old(self.larger_test_matrix,verbose=False)

		np.testing.assert_allclose(L,expected_L,rtol=rtol,atol=atol)
		np.testing.assert_allclose(S,expected_S,rtol=rtol,atol=atol)

	def tearDown(self):
		del self.vector
		del self.matrix
		del self.test_matrix
		del self.larger_test_matrix

if __name__ == '__main__':
	nose.runmodule(argv=[__file__],exit=False)