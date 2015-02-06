import numpy as np
from numpy import linalg as LA
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import diags

def rpca_alm(M,gamma=None,tol=1e-7,maxiter=500):
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

	mu_inv = 4*one_norm/np.prod(n)

	# Kicking
	k = np.min([np.floor(mu_inv/two_norm), np.floor(gamma*mu_inv/inf_norm)])
	Y = k*M

	# Main loop
	sv = 10;
	S = np.zeros(n);
	L = S;

	print np.sum(np.abs(S) > 0)

	for k in range(maxiter):
		# Shrink entries
		S = vector_shrink(M - L + mu_inv*Y, gamma*mu_inv)
		
		# Shrink singular values 
		L,r = matrix_shrink(M - S + mu_inv*Y, mu_inv,sv)
		if r < sv:
			sv = np.min([r + 1, np.min(n)])
		else:
			sv = np.min([r + np.round(0.05*np.min(n)), np.min(n)])
		
		stopCriterion = LA.norm(M-L-S,'fro')/Frob_norm
		
		if k % 1 == 0:
			print "iter: {0}, rank(L) {1}, |S|_0: {2}, stopCriterion {3}".format(k,r,np.sum(np.abs(S) > 0),stopCriterion)
		
		# Check convergence
		if stopCriterion < tol:
			break
		
		# Update dual variable
		Y += (M-L-S)/mu_inv

	niter = k+1
	return (L,S,niter)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Auxilliary functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def matrix_shrink(X,tau,sv):

	m = np.min(X.shape)

	if use_rand_svd(m,sv):
		U,S,V = randomized_svd(X,sv)
	else:
		U,S,V = LA.svd(X,full_matrices=0)
	
	r = np.sum(S > tau);
	Y = np.zeros(X.shape,dtype=X.dtype)
	if r > 0:
		np.dot(U[:,:r]*(S[:r]-tau),V[:r,:],out=Y)

	return (Y,r)




def vector_shrink(X,tau):
	return np.sign(X)*np.maximum(np.abs(X) - tau, 0);

def use_rand_svd(n_int,d_int):
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






