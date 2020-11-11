import numpy as np
from tqdm import tqdm, trange

class Observable:
	f_args = None
	g_args = None

class AverageNeighborDegree(Observable):
	@staticmethod
	def func(adj, bslice=None):
		if bslice is None:
			return adj.dot(adj).sum(axis=1) / adj.sum(axis=1)
		else:
			return adj[bslice,:].dot(adj).sum() / adj[bslice,:].sum()


class Connectivity(Observable):
	def __init__(self, nodeset1=None, nodeset2=None, directed=False):
		self.f_args = [nodeset1, nodeset2, directed]
		self.g_args = [nodeset1, nodeset2]
		self.directed = directed

	@staticmethod
	def func(adj, nodeset1=None, nodeset2=None, directed=False):
		if nodeset1 is None:
			if directed:
				return adj.sum()
			else:
				return np.triu(adj).sum()
		elif nodeset2 is None:
			nodeset1 = np.asarray(nodeset1)
			if directed:
				return adj[nodeset1[:,None], nodeset1].sum()
			else:
				return np.triu(adj)[nodeset1[:,None], nodeset1].sum()
		else:
			nodeset1 = np.asarray(nodeset1)
			nodeset2 = np.asarray(nodeset2)
			if directed:
				return adj[nodeset1[:,None], nodeset2].sum()
			else:
				return adj[np.minimum(nodeset1[:, None], nodeset2), np.maximum(nodeset1[:, None], nodeset2)].sum()


	@staticmethod
	def grad(adj, nodeset1=None, nodeset2=None):
		if nodeset1 is None:
			return np.triu(np.ones(adj.shape))
		elif nodeset2 is None:
			nodeset1 = np.asarray(nodeset1)
			mask = np.zeros(adj.shape)
			mask[nodeset1[:,None],nodeset1] = 1.
			return np.triu(mask)
		else:
			nodeset1 = np.asarray(nodeset1)
			nodeset2 = np.asarray(nodeset2)
			mask = np.zeros(adj.shape)
			mask[np.minimum(nodeset1[:, None], nodeset2), np.maximum(nodeset1[:, None], nodeset2)] = 1.
			return mask

class DegreeSequence(Observable):
	obs_dim_idx = 0

	def __init__(self, nodeset=None, subgraph_nodeset=None):
		self.output_nodeset = nodeset
		self.f_args = [nodeset, subgraph_nodeset]
		self.g_args = [nodeset, subgraph_nodeset]

	@staticmethod
	def func(adj, nodeset=None, subgraph_nodeset=None):
		if nodeset is None:
			return adj.sum(axis=1)
		elif subgraph_nodeset is None:
			nodeset = np.asarray(nodeset)
			return adj[nodeset,:].sum(axis=1)
		else:
			nodeset = np.asarray(nodeset)
			subgraph_nodeset = np.asarray(subgraph_nodeset)
			return adj[nodeset[:,None], subgraph_nodeset].sum(axis=1)

	@staticmethod
	def grad(adj, nodeset=None, subgraph_nodeset=None):
		if nodeset is None:
			return np.ones(adj.shape)
		elif subgraph_nodeset is None:
			nodeset = np.asarray(nodeset)
			mask = np.zeros(adj.shape)
			mask[nodeset[:,None],nodeset] = 1.
			return mask
		else:
			nodeset = np.asarray(nodeset)
			subgraph_nodeset = np.asarray(subgraph_nodeset)
			mask = np.zeros(adj.shape)
			mask[nodeset[:, None], subgraph_nodeset] = 1.
			return mask

class RandomWalkWithRestart(Observable):
	obs_dim_idx = 1

	def __init__(self,x,lambd,mode='iterative',precomputed_kernel=None,output_nodeset=None):
		self.f_args = [x,lambd,mode,precomputed_kernel]
		self.g_args = self.f_args
		self.output_nodeset=output_nodeset

	@staticmethod
	def func(adj,x,lambd,mode='iterative',precomputed_kernel=None):
		if precomputed_kernel is None:
			if mode == 'iterative':
				return RandomWalkWithRestart.eval_iterative(adj,x,lambd)
			elif mode == 'exact':
				return RandomWalkWithRestart.eval_exact(adj,x,lambd)
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

