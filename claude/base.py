
import numpy as np
import scipy.optimize as opt
from tqdm.auto import trange



class GraphEnsemble:
	def __init__(self, N_nodes, directed=False, self_loops=False):
		self.N = N_nodes
		self.self_loops = self_loops
		self.directed = directed
		self.fixed_edges = None

	def fit(self, constraints, method='anderson', opt_kwargs=None):
		self.N_theta = sum([len(c.c_vals) for c in constraints])
		self.constraints = constraints
		self.theta = self.eval_theta(method, opt_kwargs)
		self.adj_matrix = self.eval_adj_matrix(self.theta)
		self.sigma = self.eval_sigma(self.theta)

	def predict_mean(self, func, f_args=None):
		if f_args is None:
			f_args = []
		return func(self.adj_matrix, *f_args)

	def predict_std(self, func_grad, f_args=None, batch_size=None):
		if f_args is None:
			f_args = []
		if batch_size is None:
			grad_term = func_grad(self.adj_matrix, *f_args)
			return np.sqrt(((self.sigma * grad_term) ** 2).sum())
		else:
			std_vec = np.zeros(self.N)
			for b in trange(int(self.N / batch_size)):
				bslice = slice(b*batch_size,(b+1)*batch_size)
				grad_term = func_grad(self.adj_matrix, *f_args, bslice=bslice)
				std_vec[bslice] = np.sqrt(((self.sigma[...,None] * grad_term) ** 2).sum(axis=[0,1]))
			return std_vec

	def predict_zscore(self, value, func, func_grad, f_args=None, f_grad_args=None):
		mu = self.predict_mean(func, f_args=f_args)
		sigma = self.predict_std(func_grad, f_grad_args=f_grad_args)
		return (value-mu)/sigma

	def fix_edges_value(self, edgelist, values):
		el = np.asarray(edgelist)
		if hasattr(values, '__len__'):
			values = np.asarray(values)
		else:
			values = values * np.ones(el.shape[0])
		self.fixed_edges = np.concatenate([el, values[:,None]],axis=1)

	def sample(self):
		if self.directed:
			s = (np.random.rand(self.N, self.N) <= self.adj_matrix).astype(int)
			return s
		else:
			s = np.triu((np.random.rand(self.N, self.N) <= self.adj_matrix).astype(int), 1)
			return s + s.T

	def eval_coupling_matrix(self, theta):
		coupling = np.ones([self.N, self.N])
		for c_idx, c in enumerate(self.constraints):
			i0, i1 = self.get_multipliers_idx(c_idx)
			coupling *= c.coupling_matrix(theta[i0:i1], self)
		return coupling

	def eval_adj_matrix(self, theta): # Fermi-Dirac distribution
		coupling = self.eval_coupling_matrix(theta)
		adj_matrix = (coupling / (1 + coupling))
		if not self.self_loops:
			np.fill_diagonal(adj_matrix, 0.)
		if self.fixed_edges is not None:
			adj_matrix[self.fixed_edges[:,0].astype(int),self.fixed_edges[:,1].astype(int)] = self.fixed_edges[:,2]
			if not self.directed:
				adj_matrix[self.fixed_edges[:, 1].astype(int), self.fixed_edges[:, 0].astype(int)] = self.fixed_edges[:, 2]
		return adj_matrix

	def eval_sigma(self, theta):
		coupling = self.eval_coupling_matrix(theta)
		sigma = np.sqrt(coupling) / (1 + coupling)
		if not self.self_loops:
			np.fill_diagonal(sigma, 0)
		return sigma

	def eval_ml_eqs(self, theta):
		adj = self.eval_adj_matrix(theta)
		errors = np.array([])
		for c_idx, c in enumerate(self.constraints):
			errors = np.concatenate([errors, c.eval_ml_eqs(adj, self)])
		return errors

	def eval_theta(self, method='anderson', opt_kwargs=None):
		return opt.root(self.eval_ml_eqs,
						x0=np.ones(self.N_theta),
						method=method,
						options=opt_kwargs
						).x

	def get_multipliers_idx(self, c_idx):
		i0 = sum([len(c.c_vals) for c in self.constraints[:c_idx]])
		i1 = i0 + len(self.constraints[c_idx].c_vals)
		return (i0, i1)

	def check_constraint_deviations(self, c_idx):
		return self.constraints[c_idx].eval_ml_eqs(self.adj_matrix, self)



class MultiGraphEnsemble(GraphEnsemble):

	def eval_adj_matrix(self, theta): # Bose-Einstein condensate
		coupling = self.eval_coupling_matrix(theta)
		adj_matrix = (coupling / (1 - coupling))
		if not self.self_loops:
			np.fill_diagonal(adj_matrix, 0.)
		return adj_matrix

	def eval_sigma(self, theta):
		coupling = self.eval_coupling_matrix(theta)
		sigma = np.sqrt(coupling) / (1 - coupling)
		if not self.self_loops:
			np.fill_diagonal(sigma, 0)
		return sigma

	def sample(self):
		raise NotImplementedError
