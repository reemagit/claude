import numpy as np
from tqdm import tqdm, trange


class AverageNeighborDegree:
	@staticmethod
	def func(adj, bslice=None):
		if bslice is None:
			return adj.dot(adj).sum(axis=1) / adj.sum(axis=1)
		else:
			return adj[bslice,:].dot(adj).sum() / adj[bslice,:].sum()


class RandomWalkWithRestart:
	@staticmethod
	def func(adj,x,lambd,mode='iterative',precomputed_kernel=None):
		if precomputed_kernel is None:
			if mode == 'iterative':
				return RandomWalkWithRestart.eval_iterative(adj,x,lambd)
			elif mode == 'exact':
				return RandomWalkWithRestart.eval_exact(adj,x,lambd)
				print(tm)
			elif mode == 'slice':
				raise NotImplementedError()
				#import jax.numpy as jnp
				#I = np.eye(p.shape[0])
				#e = np.zeros(adj.shape[0])
				#e[bslice] = 1
				#yi = lambd * jnp.linalg.solve(e, I - (1 - lambd) * p)
				#return (yi + x).sum()
			else:
				raise ValueError('The value of the mode parameter can only be "iterative", or "exact"')
		else:
			tm = precomputed_kernel
			return x.dot(tm)


	@staticmethod
	def grad(adj,x,lambd,mode='iterative',precomputed_kernel=None):
		dinv = 1/adj.sum(axis=1)
		omega = RandomWalkWithRestart.func(adj=adj,x=x,lambd=lambd,mode=mode,precomputed_kernel=precomputed_kernel)
		gradval = (1 - lambd) * dinv[:, None] * omega[:, None]
		return gradval

	@staticmethod
	def eval_exact(adj=None, x=None, lambd=None):
		tm = RandomWalkWithRestart.eval_transfer_matrix(adj, lambd)
		return x.dot(tm)

	@staticmethod
	def eval_transfer_matrix(adj, lambd):
		#import jax.numpy as jnp
		p = adj / np.reshape(adj.sum(axis=0), [adj.shape[0], 1])
		I = np.eye(p.shape[0])
		return lambd*np.linalg.inv(I - (1 - lambd) * p)

	@staticmethod
	def eval_iterative(adj, x, lambd, max_iter=100, tol=1e-10):
		import scipy.sparse as sp
		ndim = max(x.shape)
		p = adj / np.reshape(adj.sum(axis=0), [adj.shape[0], 1])
		w = sp.csr_matrix(p)
		vec_sp = sp.csr_matrix(np.reshape(x, [1, ndim]))
		vec_sp /= vec_sp.sum()
		v0 = vec_sp.copy()
		vt = v0.copy()
		I_sp = sp.eye(ndim)
		err = []
		for i in tqdm(np.arange(max_iter)):
			prev_vt = vt.copy()
			vt = (1 - lambd) * vt.dot(w) + lambd * v0.dot(I_sp)
			err.append((np.linalg.norm((vt - prev_vt).toarray().flatten()).sum()))
			if err[-1] <= tol or np.mean(err[-5:-1]) == err[-1]:
				break
		return np.reshape(vt.toarray(), x.shape)

