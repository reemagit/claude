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
	def func(adj,x,lambd,bslice=None):
		import jax.numpy as jnp
		p = adj / jnp.reshape(adj.sum(axis=0), [adj.shape[0], 1])
		I = jnp.eye(p.shape[0])
		if bslice is None:
			return lambd*x.dot(jnp.linalg.inv(I - (1-lambd)*p))
		else:
			e = np.zeros(adj.shape[0])
			e[bslice] = 1
			yi = lambd * jnp.linalg.solve(e,I - (1-lambd)*p)
			return yi

	@staticmethod
	def grad(adj,x,lambd,bslice):
		dinv = 1/adj.sum(axis=1)
		p = adj * dinv[:,None]
		I = np.eye(p.shape[0])
		omega = lambd * x[None,:].dot(np.linalg.inv(I - (1 - lambd) * p))
		inv_term = np.eye(adj.shape[0])
		gradval = (1 - lambd) * dinv[:, None, None] * omega[0,:, None, None] * inv_term[None, :, bslice]
		return gradval

'''
kappa = jnp.eye(N) - (1-lambd) * dinv[:,None] * maxent_adj

invm= jnp.linalg.inv(kappa)

maxent_k = lambd * invm

#sigma_y = np.zeros(N)
#omega = x[None,:].dot(invm)
#for n in trange(N):
#    term = (sigma_aij * (1-lambd) * lambd * dinv[:,None] * invm[None,:,n] * omega[0,:,None])**2
#    sigma_y[n] = np.sqrt(np.sum(term))

omega = x[None,:].dot(invm)
part_term = sigma_aij * (1-lambd) * lambd * dinv[:,None] * omega[0,:,None]
def wrap(n):
    #inv_term = invm
    inv_term = np.eye(N)
    term = (part_term * inv_term[None,:,n])**2
    return np.sqrt(np.sum(term))

sigma_y_neu_0 = np.asarray(list(map(wrap, trange(N))))
'''