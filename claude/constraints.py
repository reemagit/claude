import numpy as np
import scipy.optimize as opt
from tqdm.auto import trange
from . import observables as obs



class Constraint:
	def __init__(self, c_vals):
		self.c_vals = c_vals

	def coupling_matrix(self, c_matrix, gens=None):
		return np.exp(-c_matrix)

	def __add__(self, other):
		new_c = Constraint(np.concatenate([self.c_vals, other.c_vals]))
		new_c.ml_eqs = lambda theta: np.concatenate([self.ml_eqs(theta), other.ml_eqs(theta)])
		return new_c

class DegreeSequence(Constraint):
	def __init__(self, degrees, nodeset=None, subgraph_nodeset=None):
		super().__init__(degrees)
		self.nodeset = nodeset
		if self.nodeset is not None:
			self.nodeset = np.asarray(self.nodeset).astype(int)
		self.bg_nodeset = subgraph_nodeset
		if self.bg_nodeset is not None:
			#if not np.all(np.in1d(self.nodeset,self.bg_nodeset)):
			#	raise ValueError('Selected nodeset is not fully contained in subgraph nodeset')
			self.bg_nodeset = np.asarray(self.bg_nodeset).astype(int)

	def coupling_matrix(self, theta, gens):
		if self.nodeset is None:
			c = theta[:, None] * theta[None, :]
		elif self.bg_nodeset is None:
			c = np.zeros([gens.N, gens.N])
			c[self.nodeset, :] = theta[:,None]
			c[:, self.nodeset] = theta[None,:]
			c[self.nodeset[:,None],self.nodeset] = theta[:,None] * theta[None,:]
		else:
			c = np.zeros([gens.N, gens.N])
			c[self.nodeset[:,None], self.bg_nodeset] = theta[:, None]
			c[self.bg_nodeset[:,None], self.nodeset] = theta[None, :]
			c[self.nodeset[:,None], self.nodeset] = theta[:, None] * theta[None,:]
		return np.exp(-c)

	def eval_ml_eqs(self, adj_matrix, gens):
		if gens.directed:
			raise ValueError('Cannot impose a DegreeSequence constraint on an undirected GraphEnsemble object. '
							 'Use OutDegreeSequence and InDegreeSequence constraints instead.')
		return obs.DegreeSequence.func(adj_matrix, self.nodeset, self.bg_nodeset) - self.c_vals

class OutDegreeSequence(DegreeSequence):

	def coupling_matrix(self, theta, gens):
		if self.nodeset is None:
			c = theta[:, None] * np.ones([gens.N,gens.N])
		elif self.bg_nodeset is None:
			c = np.zeros([gens.N, gens.N])
			c[self.nodeset, :] = theta[:,None]
		else:
			c = np.zeros([gens.N, gens.N])
			c[self.nodeset[:,None], self.bg_nodeset] = theta[:, None]
		return np.exp(-c)

	def eval_ml_eqs(self, adj_matrix, gens):
		return obs.OutDegreeSequence.func(adj_matrix, self.nodeset, self.bg_nodeset) - self.c_vals
		#if self.nodeset is None:
		#	return adj_matrix.sum(axis=1) - self.c_vals
		#elif self.bg_nodeset is None:
		#	return adj_matrix[self.nodeset, :].sum(axis=1) - self.c_vals
		#else:
		#	return adj_matrix[self.nodeset[:, None], self.bg_nodeset].sum(axis=1) - self.c_vals

class InDegreeSequence(DegreeSequence):

	def coupling_matrix(self, theta, gens):
		if self.nodeset is None:
			c = theta[None, :] * np.ones([gens.N,gens.N])
		elif self.bg_nodeset is None:
			c = np.zeros([gens.N, gens.N])
			c[:, self.nodeset] = theta[None,:]
		else:
			c = np.zeros([gens.N, gens.N])
			c[self.bg_nodeset[:,None], self.nodeset] = theta[None, :]
		return np.exp(-c)

	def eval_ml_eqs(self, adj_matrix, gens):
		return obs.InDegreeSequence.func(adj_matrix, self.nodeset, self.bg_nodeset) - self.c_vals
		#if self.nodeset is None:
		#	return adj_matrix.sum(axis=0) - self.c_vals
		#elif self.bg_nodeset is None:
		#	return adj_matrix[:, self.nodeset].sum(axis=0) - self.c_vals
		#else:
		#	return adj_matrix[self.bg_nodeset[:, None], self.nodeset].sum(axis=0) - self.c_vals

class BipartiteOutDegreeSequence(OutDegreeSequence):

	def coupling_matrix(self, theta, gens):
		if self.nodeset is None:
			c = theta[:, None] * np.ones([gens.N1,gens.N2])
		elif self.bg_nodeset is None:
			c = np.zeros([gens.N1, gens.N2])
			c[self.nodeset, :] = theta[:,None]
		else:
			c = np.zeros([gens.N1, gens.N2])
			c[self.nodeset[:,None], self.bg_nodeset] = theta[:, None]
		return np.exp(-c)

class BipartiteInDegreeSequence(InDegreeSequence):

	def coupling_matrix(self, theta, gens):
		if self.nodeset is None:
			c = theta[None, :] * np.ones([gens.N1,gens.N2])
		elif self.bg_nodeset is None:
			c = np.zeros([gens.N1, gens.N2])
			c[:, self.nodeset] = theta[None,:]
		else:
			c = np.zeros([gens.N1, gens.N2])
			c[self.bg_nodeset[:,None], self.nodeset] = theta[None, :]
		return np.exp(-c)

class Connectivity(Constraint):
	def __init__(self, n_edges, nodeset1=None, nodeset2=None):
		super().__init__(np.asarray([n_edges,]))
		self.nodeset1 = nodeset1
		self.nodeset2 = nodeset2
		if self.nodeset1 is not None:
			self.nodeset1 = np.asarray(self.nodeset1).astype(int)
		if self.nodeset2 is not None:
			self.nodeset2 = np.asarray(self.nodeset2).astype(int)

	def coupling_matrix(self, theta, gens):
		if self.nodeset1 is None:
			c = theta * np.ones([gens.N,gens.N])
		elif self.nodeset2 is None:
			c = np.zeros([gens.N,gens.N])
			c[self.nodeset1[:,None],self.nodeset1] = theta
		else:
			c = np.zeros([gens.N, gens.N])
			c[self.nodeset1[:, None], self.nodeset2] = theta
			if not gens.directed:
				c[self.nodeset2[:, None], self.nodeset1] = theta
		return np.exp(-c)

	def eval_ml_eqs(self, adj_matrix, gens):
		return obs.Connectivity.func(adj_matrix, self.nodeset1, self.nodeset2, gens.directed) - self.c_vals
