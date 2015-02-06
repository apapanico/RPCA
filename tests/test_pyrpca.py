# -*- coding: utf-8 -*-

"""
test_pyrpca
----------------------------------

Tests for `pyrpca` module.
"""

import unittest

from .context import pyrpca

import numpy as np

test_tol = 1e-7

test_matrix = np.load("tests/data/test_matrix.csv")


class TestPyrpca(unittest.TestCase):

	def setUp(self):
		self.vector = np.array([0.97524828,  0.22805508, -1.09904465,  1.01910579, -0.77785785])
		self.matrix1 = np.array([[ 0.18922838, -0.98083816],
							   [-0.34132646, -0.31252715],
							   [ 0.27660279, -0.03665899]])


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

		Y,r = pyrpca.matrix_shrink(self.matrix1,tau,sv)

		np.testing.assert_allclose(Y,expected_Y,atol=test_tol)
		np.testing.assert_allclose(r,expected_r)

	def test_rpca(self):
		

		L,S,niter = rpca_alm(M,gamma=None,tol=1e-7,maxiter=500)

		np.testing.assert_allclose(Y,expected_Y,atol=test_tol)
		np.testing.assert_allclose(S,expected_S,atol=test_tol)
		np.testing.assert_allclose(niter,expected_niter)

	def tearDown(self):
		del self.vector
		del self.matrix1



if __name__ == '__main__':
	nose.runmodule(argv=[__file__],exit=False)