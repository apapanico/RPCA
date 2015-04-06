# -*- coding: utf-8 -*-

"""
test_pyrpca
----------------------------------

Tests for `pyrpca` module.
"""

import unittest

from .context import pyrpca

import numpy as np

rtol = 1e-8
atol = 1e-8

test_matrix = np.loadtxt("tests/data/test_matrix.csv",delimiter=',')
test_matrix_L = np.loadtxt("tests/data/test_matrix_L.csv",delimiter=',')
test_matrix_S = np.loadtxt("tests/data/test_matrix_S.csv",delimiter=',')

larger_test_matrix = np.loadtxt("tests/data/larger_test_matrix.csv",delimiter=',')
larger_test_matrix_L = np.loadtxt("tests/data/larger_test_matrix_L.csv",delimiter=',')
larger_test_matrix_S = np.loadtxt("tests/data/larger_test_matrix_S.csv",delimiter=',')


class Test_rpca_alm(unittest.TestCase):

	def setUp(self):
		self.vector = np.array([0.97524828,  0.22805508, -1.09904465,  1.01910579, -0.77785785])
		self.matrix = np.array([[ 0.18922838, -0.98083816],
							   [-0.34132646, -0.31252715],
							   [ 0.27660279, -0.03665899]])
		self.test_matrix = test_matrix
		self.larger_test_matrix = larger_test_matrix


	def test_vector_shrink(self):
		tau = .5
		expected = np.array([0.47524828,  0.0, -0.59904465,  0.51910579, -0.27785785])
		np.testing.assert_allclose(pyrpca.vector_shrink(self.vector,tau),expected)

	def test_matrix_shrink1(self):
		tau = .5
		sv = 1
		expected_Y = np.array([[ 0.05412482, -0.51145982],
						   [ 0.01494755, -0.14124894],
						   [ 0.00356539, -0.0336916 ]])
		expected_r = 1

		Y,r = pyrpca.matrix_shrink(self.matrix,tau,sv)

		np.testing.assert_allclose(Y,expected_Y,rtol=rtol,atol=atol)
		np.testing.assert_allclose(r,expected_r)

	def test_rpca_alm(self):
		expected_L = test_matrix_L
		expected_S = test_matrix_S
		expected_niter = 355
		L,S,niter = pyrpca.rpca_alm(self.test_matrix)

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



if __name__ == '__main__':
	nose.runmodule(argv=[__file__],exit=False)