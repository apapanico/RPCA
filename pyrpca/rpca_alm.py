import numpy as np
from numpy import linalg as LA
from pypropack import svdp
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import diags

def rpca_alm(M,gamma=None,tol=1e-7,maxiter=500,verbose=True,use_rand_svd=False):
	"""
	Finds the Principal Component Pursuit solution 
	# %
	minmize		 ||L||_* + gamma || S ||_1 
	# %
	subject to	  L + S = M
	# %
	using an augmented Lagrangian approach
	# %
	Usage:  L,S,iter  = rpca_alm(M,gamma=None,tol=1e-7,maxiter=500)
	# %
	Inputs:
	# %
	M	   - input matrix of size n1 x n2 
	# %
	gamma  - parameter defining the objective functional 

	tol	 - algorithm stops when ||M - L - S||_F <= delta ||M||_F 
	# %
	maxiter - maximum number of iterations
	# %
	Outputs: 

	L		- low-rank component

	S		- sparse component
	# %
	iter	 - number of iterations to reach convergence

	Reference:
	# %
	   Candes, Li, Ma, and Wright 
	   Robust Principal Component Analysis? 
	   Submitted for publication, December 2009.
	# %
	Written by: Alex Papanicolaou
	Created: January 2015"""
	n = M.shape 
	Frob_norm = LA.norm(M,'fro');
	two_norm = LA.norm(M,2)
	one_norm = np.sum(np.abs(M))
	inf_norm = np.max(np.abs(M))

	if gamma is None:
		gamma = 1/np.sqrt(np.max(n))

	K = 1
	if verbose and isinstance(verbose,int):
		K = verbose

	mu_inv = 4*one_norm/np.prod(n)

	# Kicking
	k = np.min([np.floor(mu_inv/two_norm), np.floor(gamma*mu_inv/inf_norm)])
	Y = k*M
	sv = 10

	# Variable init
	zero_mat = np.zeros(n)
	S = zero_mat.copy()
	L = zero_mat.copy()
	R = M.copy()
	T1 = zero_mat.copy()
	T2 = zero_mat.copy()

	np.multiply(Y,mu_inv,out=T1)
	np.add(T1,M,out=T1)

	for k in range(maxiter):
		# Shrink entries
		np.subtract(T1,L,out=T2)
		S = vector_shrink(T2, gamma*mu_inv, out = S)
		
		# Shrink singular values 
		np.subtract(T1,S,out=T2)
		L,r = matrix_shrink(T2, mu_inv, sv, out=L, use_rand_svd=use_rand_svd)

		if r < sv:
			sv = np.min([r + 1, np.min(n)])
		else:
			sv = np.min([r + np.round(0.05*np.min(n)), np.min(n)])
		
		np.subtract(M,L,out=R)
		np.subtract(R,S,out=R)
		stopCriterion = LA.norm(R,'fro')/Frob_norm
		
		if verbose and k % K == 0:
			print "iter: {0}, rank(L) {1}, |S|_0: {2}, stopCriterion {3}".format(k,r,np.sum(np.abs(S) > 0),stopCriterion)
		
		# Check convergence
		if stopCriterion < tol:
			break
		
		# Update dual variable
		np.multiply(R,1./mu_inv,out=T2)
		np.add(T2,Y,out=Y)
		# Y += R/mu_inv

		np.add(T1,R,out=T1)

	niter = k+1
	if verbose:
		print "iter: {0}, rank(L) {1}, |S|_0: {2}, stopCriterion {3}".format(k,r,np.sum(np.abs(S) > 0),stopCriterion)
	
	return (L,S,niter)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Auxilliary functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def matrix_shrink(X,tau,sv,out=None,use_rand_svd=False):

	m = np.min(X.shape)

	if use_rand_svd:
		U,sig,V = randomized_svd(X,int(sv))
	elif choosvd(m,sv):
		U,sig,V = svdp(X,int(sv))
	else:
		U,sig,V = LA.svd(X,full_matrices=0)
	
	r = np.sum(sig > tau);
	if r > 0:
		np.multiply(U[:,:r],(sig[:r]-tau),out=X[:,:r])
		Z = np.dot(X[:,:r],V[:r,:],out=out)
	else:
		out[:] = 0
		Z = out
	return (Z,r)


def vector_shrink(X, tau, out=None):
	np.absolute(X,out=out)
	np.subtract(out,tau,out=out)
	np.maximum(out, 0.0,out=out)
	return np.multiply(np.sign(X),out,out=out)

def choosvd(n_int,d_int):
	n = float(n_int)
	d = float(d_int)
	if n <= 100:
		if d / n <= 0.02:
			return True
		else:
			return False
	elif n <= 200:
		if d / n <= 0.06:
			return True
		else:
			return False
	elif n <= 300:
		if d / n <= 0.26:
			return True
		else:
			return False
	elif n <= 400:
		if d / n <= 0.28:
			return True
		else:
			return False
	elif n <= 500:
		if d / n <= 0.34:
			return True
		else:
			return False
	else:
		if d / n <= 0.38:
			return True
		else:
			return False






