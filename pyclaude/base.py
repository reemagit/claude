
import numpy as np
import scipy.optimize as opt
from tqdm.auto import trange
from scipy import sparse as sp

def has_method(o, name):
    return callable(getattr(o, name, None))

def save_ensemble(ge, filepath, save_space=False):
	if save_space:
		ge.adj_matrix=None
		ge.sigma=None
	ge.save(filepath)

def load_ensemble(filepath):
	import pickle
	ge = pickle.load(open(filepath, 'rb'))
	if ge.adj_matrix is None or ge.sigma is None:
		ge.adj_matrix = ge.eval_adj_matrix(ge.theta)
		ge.sigma = ge.eval_sigma(ge.theta)
	return ge

class GraphEnsemble:
	def __init__(self, N_nodes, directed=False, self_loops=False, nodenames=None):
		self.N = N_nodes
		self.N1 = N_nodes
		self.N2 = N_nodes
		self.self_loops = self_loops
		self.directed = directed
		self.fixed_edges = None
		self.nodenames = nodenames if nodenames is not None else list(range(N_nodes))

	def fit(self, constraints, method='anderson', opt_kwargs=None, theta0=None):
		self.N_theta = sum([len(c.c_vals) for c in constraints])
		self.constraints = constraints
		self.theta = self.eval_theta(method, opt_kwargs, theta0)
		self.adj_matrix = self.eval_adj_matrix(self.theta)
		self.sigma = self.eval_sigma(self.theta)

	def predict_mean(self, obs, f_args=None):
		if has_method(obs, 'func'): # obs is of class Observable
			func = obs.func
			f_args = obs.f_args
		else:
			func = obs
		if f_args is None:
			f_args = []
		return func(self.adj_matrix, *f_args)

	def predict_std(self, obs, g_args=None, obs_dim_idx=None, slice_param=None, batch_size=None):
		if has_method(obs, 'grad'): # obs is of class Observable
			if hasattr(obs, 'obs_dim_idx'):
				obs_dim_idx = obs.obs_dim_idx
			g_args = obs.g_args
			func_grad = obs.grad
		else:
			func_grad = obs
		if g_args is None:
			g_args = []
		if batch_size is None:
			grad_term = func_grad(self.adj_matrix, *g_args)
			if obs.sparse:
				sum_dims = None if obs_dim_idx is None else 1-obs_dim_idx # scipy sparse does not support tuple axes
				std_vec = np.squeeze(np.asarray(np.sqrt((grad_term.multiply(self.sigma) ** 2).sum(axis=sum_dims))))
				if obs_dim_idx is None:
					std_vec = std_vec.item() # sparse returns a numpy scalar, so convert to float
			else:
				sum_dims = tuple([dim for dim in range(len(grad_term.shape)) if dim != obs_dim_idx])
				std_vec = np.sqrt(((self.sigma * grad_term) ** 2).sum(axis=sum_dims))
		else:
			if obs.sparse:
				raise NotImplementedError('Sparse mode with batch_size calculation not implemented yet')
			slice_param = obs.slice_param
			std_vec = np.zeros(obs.obs_dim)
			for b in trange(int(obs.obs_dim / batch_size)):
				bslice = slice(b*batch_size,(b+1)*batch_size)
				grad_term = func_grad(self.adj_matrix, *g_args, {slice_param:bslice})
				std_vec[bslice] = np.sqrt(((self.sigma[...,None] * grad_term) ** 2).sum(axis=(0,1)))
		return std_vec

	def predict_zscore(self, value, obs, obs_grad=None, f_args=None, g_args=None, batch_size=None):
		if has_method(obs, 'func') and has_method(obs, 'grad'): # obs is of class Observable
			mu = self.predict_mean(obs)
			sigma = self.predict_std(obs, batch_size=batch_size)
		else:
			mu = self.predict_mean(obs, f_args=f_args)
			sigma = self.predict_std(obs_grad, g_args=g_args, batch_size=batch_size)
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

	def eval_theta(self, method='anderson', opt_kwargs=None, theta0=None):
		if theta0 is None:
			theta0 = np.ones(self.N_theta)
		return opt.root(self.eval_ml_eqs,
						x0=theta0,
						method=method,
						options=opt_kwargs
						).x

	def get_multipliers_idx(self, c_idx):
		i0 = sum([len(c.c_vals) for c in self.constraints[:c_idx]])
		i1 = i0 + len(self.constraints[c_idx].c_vals)
		return (i0, i1)

	def check_constraint_deviations(self, c_idx):
		return self.constraints[c_idx].eval_ml_eqs(self.adj_matrix, self)

	def get_graph_dims(self):
		return np.array([self.N,self.N])

	def save(self, filepath):
		import pickle
		pickle.dump(self, open(filepath,'wb'), pickle.HIGHEST_PROTOCOL)

class BipartiteGraphEnsemble(GraphEnsemble):
	def __init__(self, N1_nodes, N2_nodes):
		self.N1 = N1_nodes
		self.N2 = N2_nodes
		self.fixed_edges = None

	def eval_theta(self, method='anderson', opt_kwargs=None, theta0=None):
		if theta0 is None:
			theta0 = np.ones(self.N_theta)
		return opt.root(self.eval_ml_eqs,
						x0=theta0,
						method=method,
						options=opt_kwargs
						).x

	def eval_coupling_matrix(self, theta):
		coupling = np.ones([self.N1, self.N2])
		for c_idx, c in enumerate(self.constraints):
			i0, i1 = self.get_multipliers_idx(c_idx)
			coupling *= c.coupling_matrix(theta[i0:i1], self)
		return coupling

	def eval_adj_matrix(self, theta): # Fermi-Dirac distribution
		coupling = self.eval_coupling_matrix(theta)
		adj_matrix = (coupling / (1 + coupling))
		if self.fixed_edges is not None:
			adj_matrix[self.fixed_edges[:,0].astype(int),self.fixed_edges[:,1].astype(int)] = self.fixed_edges[:,2]
		return adj_matrix

	def eval_sigma(self, theta):
		coupling = self.eval_coupling_matrix(theta)
		sigma = np.sqrt(coupling) / (1 + coupling)
		return sigma

	def predict_std(self, obs, g_args=None, obs_dim_idx=None, batch_size=None, obs_dim=None):
		if has_method(obs, 'grad'): # obs is of class Observable
			if hasattr(obs, 'obs_dim_idx'):
				obs_dim_idx = obs.obs_dim_idx
			g_args = obs.g_args
			func_grad = obs.grad
		else:
			func_grad = obs
		if g_args is None:
			g_args = []
		if batch_size is None:
			grad_term = func_grad(self.adj_matrix, *g_args)
			sum_dims = tuple([dim for dim in range(len(grad_term.shape)) if dim != obs_dim_idx])
			std_vec = np.sqrt(((self.sigma * grad_term) ** 2).sum(axis=sum_dims))
		else:
			std_vec = np.zeros(obs_dim)
			for b in trange(int(obs_dim / batch_size)):
				bslice = slice(b*batch_size,(b+1)*batch_size)
				grad_term = func_grad(self.adj_matrix, *g_args, bslice=bslice)
				std_vec[bslice] = np.sqrt(((self.sigma[...,None] * grad_term) ** 2).sum(axis=(0,1)))
		if hasattr(obs, 'output_nodeset') and obs.output_nodeset is not None:
			std_vec = std_vec[obs.output_nodeset]
		return std_vec

	def sample(self):
		s = (np.random.rand(self.N1, self.N2) <= self.adj_matrix).astype(int)
		return s

	def get_graph_dims(self):
		return np.array([self.N1,self.N2])

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

class GraphEnsembleSet:
	def __init__(self, ge_list, nid_pair_list):
		if not self.check_consistency(ge_list, nid_pair_list):
			raise ValueError('Node ids are not associated to consistent GraphEnsemble dimensions')
		self.ge_list = ge_list
		self.nid_pair_list = nid_pair_list
		self.nidpair2geid = {(nid_pair_list[i][0],nid_pair_list[i][1]):i for i in range(len(ge_list))}

	def __getitem__(self, nid_pair):
		nid1,nid2 = nid_pair
		return self.ge_list[self.nidpair2geid[(nid1,nid2)]]

	def __len__(self):
		return len(self.ge_list)

	def fit(self, constraints_list, method='anderson', opt_kwargs=None, theta0=None):
		if isinstance(method, str):
			method = [method for i in range(len(self.ge_list))]
		if isinstance(opt_kwargs, dict):
			opt_kwargs = [opt_kwargs for i in range(len(self.ge_list))]
		if not isinstance(theta0, list):
			theta0 = [theta0 for i in range(len(self.ge_list))]

		for i in range(len(self.ge_list)):
			self.ge_list[i].fit(constraints_list[i], method=method[i], opt_kwargs=opt_kwargs[i], theta0=theta0[i])

	def predict_mean(self, obs):
		adj_list = [self[obs.nid_pair_list[i]].adj_matrix for i in range(len(obs))]
		return obs.func(adj_list)

	def predict_std(self, obs):
		adj_list = [self[obs.nid_pair_list[i]].adj_matrix for i in range(len(obs))]
		std = 0
		for i in range(len(obs)):
			grad_term = obs.grad(adj_list, term_num=i)
			std = std + ((self[obs.nid_pair_list[i]].sigma * grad_term) ** 2).sum()
		return np.sqrt(std)


	def predict_mean_list(self, obs_list, f_args_list):
		return [self.ge_list[i].predict_mean(obs_list[i], f_args_list[i]) for i in range(len(self.ge_list))]

	def check_consistency(self, ge_list, nid_pair_list):
		nid2dim = {}
		for i,(nid1,nid2) in enumerate(nid_pair_list):
			ge = ge_list[i]
			for j,nid in enumerate([nid1,nid2]):
				if nid not in nid2dim:
					nid2dim[nid] = ge.get_graph_dims()[j]
				else:
					if nid2dim[nid] != ge.get_graph_dims()[j]:
						return False
		return True

