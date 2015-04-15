pyrpca
======


Python implementations of RPCA

Usage
-----

```
import numpy as np
import pyrpca

n = 50 
r = 2
np.random.seed(123)
base = 100 + np.cumsum(np.random.randn(n,r),axis=0)
scales = np.abs(np.random.randn(n,r))
L = np.dot(base,scales.T)
S = np.round(0.25 * np.random.randn(n,n))
M = L + S

Lhat,Shat,niter = pyrpca.rpca_alm(M)
np.max(np.abs(S-Shat))
np.max(np.abs(L-Lhat))

_,s,_ = np.linalg.svd(L,full_matrices=False)
print s[s>1e-11]

_,s_hat,_ = np.linalg.svd(Lhat,full_matrices=False)
print s_hat[s_hat>1e-11]
```


Requirements
------------

+ Numpy
+ PyPROPACK (https://github.com/jakevdp/pypropack)


Authors
-------

`pyrpca` was written by `Alex Papanicolaou <alex.papanic@gmail.com>`_.

Reference
---------

Candes, Li, Ma, and Wright. Robust Principal Component Analysis?  Submitted for publication, December 2009.
[http://statweb.stanford.edu/~candes/papers/RobustPCA.pdf](http://statweb.stanford.edu/~candes/papers/RobustPCA.pdf)