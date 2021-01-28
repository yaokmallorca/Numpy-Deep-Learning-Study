from abc import ABC, abstractmethod
import numpy as np
from ..wrappers import init_wrappers, Dropout

from ..initializers import (
	WeightInitializer,
	OptimizerInitializer,
	ActivationInitializer,
	)

from ..utils.utils import (
	pad1D,
	pad2D,
	conv1D,
	conv2D,
	im2col,
	col2im,
	dilate,
	deconv2D_naive,
	calc_pad_dims_2D,
	)

class LayerBase(ABC):
	def __init__(self, optimizer=None):
		self.X = []
		self.act_fn = None
		self.trainable = True
		self.optimizer = OptimizerInitializer(optimizer)()

		self.gradients = {}
		self.parameters = {}
		self.derived_variables = {}
		super().__init__()

	@abstractmethod
	def _init_params(self, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def forward(self, z, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def backward(self, out, **kwargs):
		raise NotImplementedError

	def freeze(self):
		self.trainable = False

	def flush_gradients(self):
		assert self.trainable, 'layer is frozen'
		self.X = []
		for k,v in self.derived_variables.items():
			self.derived_variables[k] = []

		for k,v in self.gradients.items():
			self.gradients[k] = np.zeros_like(v)

	def set_params(self, summary_dict):
		"""
		set layer parameters
		parameters:
		summary_dict:
			a dictary of layer parameters and hyperparameters. 
		"""
		layer, sd = self, summary_dict

		# collapse parameters and hyperparameters nested dicts
		# into a single merged dictionary
		flatten_keys = ['parameters', 'hyperparameters']
		for k in flatten_keys:
			for k in sd:
				entry = sd[k]
				sd.update(entry)
				del sd[k]

		for k,v in sd.items():
			if k in self.parameters:
				layer.parameters[k] = v
			if k in self.hyperparameters:
				if k == 'act_fn':
					layer.act_fn = ActivationInitializer(v)()
				if k == 'optimizer':
					layer.optimizer = OptimizerInitializer(sd[k])()
				if k not in ['wrappers', 'optimizer']:
					setattr(layer, k, v)
				if k == 'wrappers':
					layer = init_wrappers(layer, sd[k])
		return layer

	def summary(self):
		return {
			'layer': self.hyperparameters['layer'],
			'parameters': self.parameters,
			'hyperparameters': self.hyperparameters,
		}

class  DotProductAttention(LayerBase):
	"""docstring for  DotProductAttention"""
	def __init__(self, scale=True, dropout_p=0, init='glorot_uniform',
		optimizer=None):
		"""
		A single "attention head" layer using a dot-product for the scoring function.
		params:
		scale: bool
			whether to scale the key-query dot product by the square root of key-query
			vector dimentionality before applying the softmax.
		dropout_p: float in [0, 1)
			the dropout probability during training.
		init: 
			the weight initializaion strategy
		optimizer:
			the optimization algorithm
		"""
		super(DotProductAttention, self).__init__(optimizer)
		self.init = init
		self.scale = scale
		self.dropout_p = dropout_p
		self._init_params()

	def _init_params(self):
		self.softmax = Dropout(Softmax(), self.dropout_p)
		smdv = self.softmax.derived_variables
		self.derived_variables = {
			'attention_weights': [].
			'dropout_mask': smdv['wrappers'][0]['dropout_mask'],
		}


	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "DotProductAttention",
			"init": self.init,
			"scale": self.scale,
			"dropout_p": self.dropout_p,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def freeze(self):
		self.trainable = False
		self.softmax.freeze()

	def unfreeze(self):
		self.trainable = True
		self.softmax.unfreeze()

	def forward(self, Q, K, V, retain_derived=True):
		"""
		compute the attention-weighted output of a collection of keys,
		values, and queries

		Notes
		-----
		In the most abstract (ie., hand-wave-y) sense:
			- Query vectors ask questions
			- Key vectors advertise their relevancy to questions
			- Value vectors give possible answers to questions
			- The dot product between Key and Query vectors provides scores for
			  each of the the `n_ex` different Value vectors

		 For a single query and `n` key-value pairs, dot-product attention 
		 (with scaling) is::
			w0 = dropout(softmax( (query @ key[0]) / sqrt(d_k) ))
			w1 = dropout(softmax( (query @ key[1]) / sqrt(d_k) ))
			...
			wn = dropout(softmax( (query @ key[n]) / sqrt(d_k) ))
			y = np.array([w0, ..., wn]) @ values
				(1 x n_ex)				(n_ex x d_v)

		In words, keys and queries are combined via dot-product to produce a
		score, which is then passed through a softmax to produce a weight on
		each value vector in Values. We elementwise multiply each value vector
		by its weight, and then take the elementwise sum of each weighted value
		vector to get the :math:`1 \times d_v` output for the current example.
		In vectorized form,
			Y = dropout(softmax(K @ Q)/sqrt(d_k)) @ V

		parameters:
		Q : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *, d_k)`
			A set of `n_ex` query vectors packed into a single matrix.
			Optional middle dimensions can be used to specify, e.g., 
			the number of parallel attention heads.
		K : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *, d_k)`
			A set of `n_ex` key vectors packed into a single matrix. 
			Optional middle dimensions can be used to specify, e.g., 
			the number of parallel attention heads.
		V : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *, d_v)`
			A set of `n_ex` value vectors packed into a single matrix. 
			Optional middle dimensions can be used to specify, e.g., 
			the number of parallel attention heads.
		retain_derived : bool
			Whether to retain the variables calculated during the forward 
			pass for use later during backprop. If False, this suggests the 
			layer will not be expected to backprop through wrt. \
			this input. Default is True.

		Returns
		-------
		Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *, d_v)`
			The attention-weighted output values
		"""
		Y, weights = self._fwd(Q, K, V)

		if retain_derived:
			self.X.append((Q, K, C))
			self.derived_variables['attention_weights'].append(weights)

	def _fwd(self, Q, K, V):
		scale = 1 / np.sqrt(Q.shape[-1]) if self.scale else 1
		scores = Q @ K.swapaxis(-2, -1) * scale # attention scores
		weights = self.softmax.forward(scores) # attention weights
		Y = weights @ V
		return Y, weights

	def backward(self, dLdY, retain_grads=True):
		"""
		backprop from layer output to inputs

		parameters:
		dLdY: (n_ex, *, d_v)
		retain_grads : bool
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.

		return:
		dQ : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *, d_k)` 
			or list of arrays
			The gradient of the loss wrt. the layer query matrix/matrices `Q`.
		dK : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *, d_k)` 
			or list of arrays
			The gradient of the loss wrt. the layer key matrix/matrices `K`.
		dV : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *, d_v)` 
			or list of arrays
			The gradient of the loss wrt. the layer value matrix/matrices `V`.
		"""
		assert self.trainable, 'layer is frozen'
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		dQ, dK, dV = [], [], []
		weights = self.derived_variables['attention_weights']
		for dy, (q, k, v), w in zip(dLdY, self.X, weights):
			dq, dk, dv = self._bwd(dy, q, k, v, w)
			dQ.append(dq)
			dK.append(dk)
			dV.append(dv)

		if len(self.X) == 1:
			dQ, dK, dV = dQ[0], dK[0], dV[0]
		return dQ, dK, dV

	def _bwd(self, dy, q, k, v, weights):
		"""
		actual computation of the gradient of the loss
		"""
		d_k = k.shape[-1]
		scale = 1 / np.sqrt(d_k) if self.scale else 1

		dV = weights.swapaxis(-2, -1) @ dy
		dWeights = dy @ v.swapaxis(-2, -1)
		dScores = self.softmax.backward(dWeights)
		dQ = dScores @ k * scale
		dK = dScores.swapaxis(-2, -1) @ q * scale
		return dQ, dK, dV

class RBM(LayerBase):
	def __init__(self, n_out, K=1, init='glorot_uniform', optimizer=None)；
	"""
	A Restricted Boltzmann machine with Bernoulli visible and hidden units.
	parameters:
	n_out: int
		the number of output dimensions
	K: int
		the number of contrastive divergence steps to run before computing 
		a single gradient update. Default is 1.
	init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
		The weight initialization strategy. Default is `'glorot_uniform'`.
	optimizer: str
	"""
		super().__init__(optimizer)

		self.K = K
		self.init = init
		self.n_in = None
		self.n_out = n_out
		self.is_initialized = False
		self.act_fn_V = ActivationInitializer("Sigmoid")()
		self.act_fn_H = ActivationInitializer("Sigmoid")()
		self._init_params()

	def _init_params(self):
		init_weights = WeightInitializer(str(self.act_fn_V), mode=self.init)
		b_in = np.zeros((1, self.n_in))
		b_out = np.zerors((1, self.n_out))
		W = init_weights((self.n_in, self.n_out))

		self.parameters = {'W':W, 'b_in':b_in, 'b_out':b_out}

		self.gradients = {
			"W": np.zeros_like(W),
			"b_in": np.zeros_like(b_in),
			"b_out": np.zeros_like(b_out),
		}

		self.derived_variables = {
			'V': None,
			'p_H': None,
			'p_V_prime': None,
			'p_H_prime': None,
			'positive_grad': None,
			'negative_grad': None,
		}
		self.is_initialized = True

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "RBM",
			"K": self.K,
			"n_in": self.n_in,
			"n_out": self.n_out,
			"init": self.init,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameterse,
			},
		}

	def CD_update(self, X):
		"""
		Perform a single contrastive divergence 'K' training update using 
		the visible inputs 'X' as a strarting point for the Gibbs sampler.

		Parameters
		 ----------
		 X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
			Layer input, representing the `n_in`-dimensional features for a
			minibatch of `n_ex` examples. Each feature in X should ideally be
			binary-valued, although it is possible to also train on real-valued
			features ranging between (0, 1) (e.g., grayscale images).
		"""
		self.forward(X)
		self.backward()

	def forward(self, V, K=None, retain_derived=True):
		"""
		Perform the CD-`k` "forward pass" of visible inputs into hidden units
		and back.
		Notes
		-----
		This implementation follows [1]_'s recommendations for the RBM forward
		pass:
			- Use real-valued probabilities for both the data and the visible
			  unit reconstructions.
			- Only the final update of the hidden units should use the actual
			  probabilities -- all others should be sampled binary states.
			- When collecting the pairwise statistics for learning weights or
			  the individual statistics for learning biases, use the
			  probabilities, not the binary states.
		References
		----------
		.. [1] Hinton, G. (2010). "A practical guide to training restricted
		   Boltzmann machines". *UTML TR 2010-003*
		Parameters
		----------
		V : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
			Visible input, representing the `n_in`-dimensional features for a
			minibatch of `n_ex` examples. Each feature in V should ideally be
			binary-valued, although it is possible to also train on real-valued
			features ranging between (0, 1) (e.g., grayscale images).
		K : int
			The number of steps of contrastive divergence steps to run before
			computing the gradient update. If None, use ``self.K``. Default is
			None.
		retain_derived : bool
			Whether to retain the variables calculated during the forward pass
			for use later during backprop. If False, this suggests the layer
			will not be expected to backprop through wrt. this input. Default
			is True.
		"""
		if not self.is_initialized:
			self.n_in = V.shape[1]
			self._init_params()

		# override self.K if nessary
		K = self.K if K is None else K
		W = self.parameters['W']
		b_in = self.parameters['b_in']
		b_out = self.parameters['b_out']

		# compute hidden unit probabilities
		Z_H = V @ H + b_out
		p_H = self.act_fn_H.fn(Z_H)

		# sample hidden states (stochastic binary values)
		H = np.randim.rand(*p_H.shape) <= p_H
		H = H.astype(float)

		# always use probabilities when computing gradients
		potitive_grad = V.T @ p_H

		# perform CD-k
		# TODO: use persistent CD-k
		# https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf
		H_prime = H.copy()
		for k in range(K):
			# resample 'v' given h (H_prime is binary for all but final step)
			Z_V_prime = H_prime @ W.T + b_in
			p_V_prime = self.act_fn_V.fn(Z_V_prime)

			# dont resample visual uints - always use raw probabilities
			V_prime = p_V_prime

			# compute p(h' | v')
			Z_H_prime = V_prime @ W + b_out
			p_H_prime = self.act_fn_H.fn(Z_H_prime)

			# if this is the final iteration of CD, keep hidden state 
			# probabilities (dont sample)
			H_prime = p_H_prime
			if k != self.K - 1:
				H_prime = np.random.rand(*p_H_prime.shape) <= p_H_prime
				H_prime = H_prime.astype(float)

		negative_grad = p_V_prime.T @ p_H_prime

		if retain_derived:
			self.derived_variables['V'] = V
			self.derived_variables['p_H'] = p_H
			self.derived_variables['p_V_prime'] = p_V_prime
			self.derived_variables['p_H_prime'] = p_H_prime
			self.derived_variables['positive_grad'] = positive_grad
			self.derived_variables['negative_grad'] = negative_grad

	def backward(self, retain_grads=True, *args):
		"""
		Perform a gradient update on the layer parameters via the contrastive
		divergence equations.
		Parameters
		----------
		retain_grads : bool
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.
		"""
		V = self.derived_variables["V"]
		p_H = self.derived_variables["p_H"]
		p_V_prime = self.derived_variables["p_V_prime"]
		p_H_prime = self.derived_variables["p_H_prime"]
		positive_grad = self.derived_variables["positive_grad"]
		negative_grad = self.derived_variables["negative_grad"]

		if retain_grads:
			self.gradients['b_in'] = V - p_V_prime
			self.gradients['b_out'] = p_H - p_H_prime
			self.gradients['W'] = positive_grad - negative_grad

	def reconstruct(self, X, n_steps=10, return_probs=False):
		"""
		Reconstruct an input `X` by running the trained Gibbs sampler for
		`n_steps`-worth of CD-`k`.
		Parameters
		----------
		X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
			Layer input, representing the `n_in`-dimensional features for a
			minibatch of `n_ex` examples. Each feature in `X` should ideally be
			binary-valued, although it is possible to also train on real-valued
			features ranging between (0, 1) (e.g., grayscale images). If `X` has
			missing values, it may be sufficient to mark them with random
			entries and allow the reconstruction to impute them.
		n_steps : int
			The number of Gibbs sampling steps to perform when generating the
			reconstruction. Default is 10.
		return_prob : bool
			Whether to return the real-valued feature probabilities for the
			reconstruction or the binary samples. Default is False.
		Returns
		-------
		V : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_ch)`
			The reconstruction (or feature probabilities if `return_prob` is
			true) of the visual input `X` after running the Gibbs sampler for
			`n_steps`.
		"""
		self.forward(X, K=n_steps)
		p_V_prime = self.derived_variables['p_V_prime']

		# ignore the gradients produced during this reconstruction
		self.flush_gradients()

		# sample V_prime reconstruction if return_prob is False
		V = p_V_prime
		if not return_prob:
			V = (np.random.rand(*p_V_prime.shape) <= p_V_prime).astype(float)
		return V

#######################################################################
#                              Layer Ops                              #
#######################################################################
class Add(LayerBase):
	def __init__(self, act_fn=None, optimizer=None):
		"""
		an additon layer that returns the sum of its inputs

		Parameters
		----------
		act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
			The element-wise output nonlinearity used in computing the final
			output. If None, use the identity function :math:`f(x) = x`.
			Default is None.
		optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
			The optimization strategy to use when performing gradient updates
			within the :meth:`update` method.  If None, use the :class:`SGD
			<numpy_ml.neural_nets.optimizers.SGD>` optimizer with
			default parameters. Default is None.
		"""
		super().__init__(optimizer)
		self.act_fn = ActivationInitializer(act_fn)()
		self._init_params()

	def _init_params(self):
		self.derived_variables = {'sum': []}

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "Sum",
			"act_fn": str(self.act_fn),
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def forward(self, X, retain_derived=True):
		"""
		Compute the layer output on a single minibatch.
		Parameters
		----------
		X : list of length `n_inputs`
			A list of tensors, all of the same shape.
		retain_derived : bool
			Whether to retain the variables calculated during the forward pass
			for use later during backprop. If False, this suggests the layer
			will not be expected to backprop through wrt. this input. Default
			is True.
		Returns
		-------
		Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *)`
			The sum over the `n_ex` examples.
		"""
		out = X[0].copy()
		for i in range(i, len(X)):
			out += X[i]
		if retain_derived:
			self.X.append(X)
			self.derived_variables['sum'].appen(out)
		return act_fn(out)

	def backward(self, dLdY, retain_grads=True):
		"""
		Backprop from layer outputs to inputs.
		Parameters
		----------
		dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *)`
			The gradient of the loss wrt. the layer output `Y`.
		retain_grads : bool
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.
		Returns
		-------
		dX : list of length `n_inputs`
			The gradient of the loss wrt. each input in `X`.
		"""
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		X = self.X
		_sum = self.derived_variables['sum']
		grads = [self._bwd(dy, x, ss) for dy, x, ss in zip(dLdY, X, _sum)]
		return grads[0] if len(X) == 1 else grads

	def _bwd(self, dLdY, X, _sum):
		grads = [dLdY * self.act_fn.grad(_sum) for _ in X]
		return grads

class  Multiply(LayerBase):
	"""docstring for  Multiply"""
	def __init__(self, act_fn=None, optimizer=None):
		super().__init__(optimizer)
		self.act_fn = ActivationInitializer(act_fn)()
		self._init_params()

	def _init_params(self):
		self.derived_variables = {'product': []}

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "Multiply",
			"act_fn": str(self.act_fn),
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def forward(self, X, retain_derived=True):
		out = X[0].copy()
		for i in range(1, len(X)):
			out *= X[i]
		if retain_derived:
			self.X.append(X)
			self.derived_variables['product'].append(out)
		return self.act_fn(out)

	def backward(self, dLdY, retain_grads=True):
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		X = self.X
		_prod = self.derived_variables['product']
		grads = [self._bwd(dy, x, pr) for dy, x, pr in zip(dLdY, X, _prod)]
		return grads[0] if len(X) == 1 else grads

	def _bwd(self, dLdY, X, prod):
		grads = [dLdY * self.act_fn.grad(prod)] * len(X)
		for i, x in enumerate(X):
			grads = [g * x if j!=i else g for j, g in enumerate(grads)]
		return grads

class Flatten(LayerBase):
	def __init__(self, keep_dim = 'first', optimizer=None):
		"""
		Flatten a multidimensional input into a 2D matrix.

		parameters:
		keep_dim: {'first', 'last', -1}		
			The dimension of the original input to retain. Typically used for		
			retaining the minibatch dimension.. If -1, flatten all dimensions.		
			Default is 'first'.
		optimizer: str
		"""
		super().__init__()
		self.keep_dims = keep_dim
		self._init_params()

	def _init_params(self):
		self.X = []
		self.gradients = {}
		self.parameters = {}
		self.derived_variables = {'in_dims': []}

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "Flatten",
			"keep_dim": self.keep_dim,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def forward(self, X, retain_derived=True):
		"""
		compute the layer output on a single minibatch

		parameters:
		X : :py:class:`ndarray <numpy.ndarray>`
			Input volume to flatten.
		retain_derived : bool
			Whether to retain the variables calculated during the forward pass
			for use later during backprop. If False, this suggests the layer
			will not be expected to backprop through wrt. this input. Default
			is True.

		return:
			Y: array (*out_dim)
			keep_dim -> first (X.shape[0], -1)
			keep_dim -> others (-1, X.shape[0])
		"""
		if retain_derived: 
			self.derived_variables['in_dims'].append(X.shape)
		if self.keep_dim == -1:
			return X.flatten().reshape(1, -1)
		rs = (X.shape[0], -1) if self.keep_dim == 'first' else (-1, X.shape[-1])
		return X.reshape(*rs)

	def backward(self, dLdY, retain_grads=True):
		"""
		Backprop from layer outputs to inputs.

		Parameters:
		dLdY : (*out_dim)
			The gradient of the loss wrt. the layer output `Y`.
		retain_grads:
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.

		return:
		dX: (*in_dim)
			the gradients of the loss, 
		"""
		if not isinstance(dLdY, list):
			dLdY = [dLdY]
		in_dims = self.derived_variables['in_dims']
		out = [dy.reshape(*dims) for dy, dims in zip(dLdY, in_dims)]
		return out[0] if len(dLdY) == 1 else out


#######################################################################
#                        Normalization Layers                         #
#######################################################################
class BatchNorm2D(object):
	"""docstring for BatchNorm2D"""
	def __init__(self, momentum=0.9, epsilon=1e-5, optimizer=None):
		"""
		A batch normalizaton layer for two-dimensional inputs with an 
		additional channel dimension.

		math
		Y = scaler * norm(X) + intercept
		norm(X) = (X - mean(X)) / sqrt(var(X) + eps)

		In contrast to :class:`LayerNorm2D`, the BatchNorm layer calculates
		the mean and var across the *batch* rather than the output features.
		This has two disadvantages:
			1. It is highly affected by batch size: smaller mini-batch sizes
			increase the variance of the estimates for the global mean and
			variance.
			2. It is difficult to apply in RNNs -- one must fit a separate
			BatchNorm layer for *each* time-step.


		"""
		super(self).__init__(optimizer)
		self.in_ch = None
		self.out_ch = None
		self.epsilon = epsilon
		self.momentum = momentum
		self.parameters = {
			'scaler': None,
			'intercept': None,
			'running_var': None,
			'running_mean': None,
		}

	def _init_params(self):
		scaler = np.random.rand(self.in_ch)
		intercept = np.zeros(self.in_ch)

		# init running mean and std at 0 and 1, respectively
		running_mean = self.zeros(self.in_ch)
		running_var = self.ones(self.in_ch)

		self.parameters = {
			"scaler": scaler,
			"intercept": intercept,
			"running_var": running_var,
			"running_mean": running_mean,
		}

		self.gradients = {
			"scaler": np.zeros_like(scaler),
			"intercept": np.zeros_like(intercept),
		}
		self.is_initialized = True

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "BatchNorm2D",
			"act_fn": None,
			"in_ch": self.in_ch,
			"out_ch": self.out_ch,
			"epsilon": self.epsilon,
			"momentum": self.momentum,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def reset_running_stats(self):
		assert self.trainable, 'layer is frozen'
		self.parameters['running_mean'] = np.zeros(self.in_ch)
		self.parameters['running_var'] = np.ones(self.in_ch)

	def forward(self, X, retain_derived=True):
		"""
		Compute the layer output on a single minibatch.
		Math:
			train:
			Y = scaler * norm(X) + intercept
			norm(X) = (X - mean(X)) / sqrt(var(X) + epsilon)
			test:
			Y = scaler * running_norm(X) + intercept
			running_norm(X) = (X - running_mean) / sqrt(running_var + epsilon)

		Parameters
		----------
		X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
			Input volume containing the `in_rows` x `in_cols`-dimensional
			features for a minibatch of `n_ex` examples.
		retain_derived : bool
			Whether to use the current intput to adjust the running mean and
			running_var computations. Setting this to True is the same as
			freezing the layer for the current input. Default is True.
		Returns
		-------
		Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
			Layer output for each of the `n_ex` examples.
		"""
		if not self.is_initialized:
			self.in_ch = self.out_ch = X.shape[3]
			self._init_params()

		ep = self.hyperparameters['epsilon']
		mm = self.hyperparameters['momentum']
		rm = self.hyperparameters['running_mean']
		rv = self.hyperparameters['running_var']

		scaler = self.parameters['scaler']
		intercept = self.parameters['intercept']

		# if the layer is frozen, use the running mean/std values rather 
		# than the mean/std values from the minibatch
		X_mean = self.parameters['runnning_mean']
		X_var  = self.parameters['running_var']

		if self.trainable and retain_grads:
			X_mean, X_var = X.mean(axis=(0, 1, 2)), X.var(axis=(0, 1, 2))
			self.parameters['running_mean'] = mm * rm + (1.0 - mm) * X_mean
			self.parameters['running_var'] = mm * rv + (1.0 - mm) * X_var

		if retain_derived:
			self.X.append(X)

		N = (X - X_mean) / np.sqrt(X_var + ep)
		y = scaler * N + intercept
		return y

	def backward(self, dLdY, retain_grads=True):
		"""
		Backprop from layer outputs to inputs.

		Parameters
		----------
		dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
			The gradient of the loss wrt. the layer output `Y`.
		retain_grads : bool
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.

		Returns
		-------
		dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
			The gradient of the loss wrt. the layer input `X`.
		"""
		assert self.trainable, 'layer is frozen'
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		dX = []
		X = self.X
		for dy, x in zip(dLdY, X):
			dx, dscaler, dintercept = self._bwd(dy, x)
			dX.append(dx)

			if retain_grads:
				self.gradients['scaler'] += dscaler
				self.gradients['intercpt'] += dintercept
		return dX[0] if len(X) == 1 else dX

	def _bwd(self, dLdY, X):
		"""
			partial(L/x) = partial(L/hat(x))*frac{1}{sqrt(var+eps)} +
						   partial(L/mean)*paitial(mean/x) +
						   partial(L/var)*paitial(var/x)
		"""
		scaler = self.parameters['scaler']
		ep = self.hyperparameters['epsilon']

		# reshape to 2D, retaning channel dim
		X_shape = X.shape
		X = np.reshape(X, (-1, X.shape[3]))
		dLdY = np.reshape(dLdY, (-1, dLdY.shape[3]))

		# apply 1D batchnorm backward pass on reshape array
		n_ex, in_ch = X.shape
		X_mean, X_var = X.mean(axis=0), X.var(axis=0)

		N = (X - X_mean) / np.sqrt(X_var + ep)
		dintercept = dLdY.sum(axis=0)
		dscaler = np.sum(dLdY*N, axis=0)

		dN = dLdY * scaler
		dX = (n_ex * dN - dN.sum(axis=0) - N * (dN * N).sum(axis=0)) / 
			(n_ex * np.sqrt(X_var + ep))
		return np.reshape(dX, X_shape), dScaler, dIntercept

class BatchNorm1D(LayerBase):
	def __init__(self, momentum=0.9, epsilon=1e-5, optimizer=None):
		"""
		a batch normalizaton layer for 1D inputs

		math train:
			Y = scaler * norm(X) + intercept
			norm(X) = (X - mean(X)) / sqrt(var(X) + eps)

		math test:
			Y = scaler * norm(X) + intercept
			norm(X) = (X - running_mean) / sqrt(running_var + eps)
		"""
		super(self).__init__(optimizer)
		self.n_in = None
		self.n_out = None
		self.epsilon = epsilon
		self.momentum = momentum
		self.parameters = {
			'scaler': None,
			'intercept': None,
			'running_var': None,
			'running_mean': None,
		}

	def _init_params(self):
		scaler = np.random.rand(self.n_in)
		intercept = np.zeros(self.n_in)

		# init running mean and std at 0 and 1, respectively
		running_mean = self.zeros(self.n_in)
		running_var = self.ones(self.n_in)

		self.parameters = {
			"scaler": scaler,
			"intercept": intercept,
			"running_var": running_var,
			"running_mean": running_mean,
		}

		self.gradients = {
			"scaler": np.zeros_like(scaler),
			"intercept": np.zeros_like(intercept),
		}
		self.is_initialized = True

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "BatchNorm1D",
			"act_fn": None,
			"in_ch": self.n_in,
			"out_ch": self.n_out,
			"epsilon": self.epsilon,
			"momentum": self.momentum,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def reset_running_stats(self):
		assert self.trainable, 'layer is frozen'
		self.parameters['running_mean'] = np.zeros(self.n_in)
		self.parameters['running_var'] = np.ones(self.n_in)

	def forward(self, X, retain_derived=True):
		"""
		compute the layer output on a single minibatch

		Parameters
		----------
		X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
			Layer input, representing the `n_in`-dimensional features for a
			minibatch of `n_ex` examples.
		retain_derived : bool
			Whether to use the current intput to adjust the running mean and
			running_var computations. Setting this to True is the same as
			freezing the layer for the current input. Default is True.
		Returns
		-------
		Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
			Layer output for each of the `n_ex` examples
		"""
		if not self.is_initialized:
			self.n_in = self.n_out = X.shape[1]
			self._init_params()

		ep = self.hyperparameters["epsilon"]
		mm = self.hyperparameters["momentum"]
		rm = self.parameters["running_mean"]
		rv = self.parameters["running_var"]

		scaler = self.parameters["scaler"]
		intercept = self.parameters["intercept"]

		if self.trainable and retain_derived:
			X_mean, X_var = X.mean(axis=0), X.var(axis=0)  # , ddof=1)
			self.parameters["running_mean"] = mm * rm + (1.0 - mm) * X_mean
			self.parameters["running_var"] = mm * rv + (1.0 - mm) * X_var

		if retain_derived:
			self.X.append(X)

		N = (X - X_mean) / np.sqrt(X_var + ep)
		y = scaler * N + intercept
		return y

	def backward(self, dLdY, retain_grads=True):
		"""
		Backprop from layer outputs to inputs.

		Parameters
		----------
		dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
			The gradient of the loss wrt. the layer output `Y`.
		retain_grads : bool
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.

		Returns
		-------
		dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
			The gradient of the loss wrt. the layer input `X`.
		"""
		assert self.trainable, 'layer is frozen'
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		dX = []
		X = self.X
		for dy, x in zip(dLdY, X):
			dx, dscaler, dintercept = self._bwd(dy, x)
			dX.append(dx)

			if retain_grads:
				self.gradients['scaler'] += dscaler
				self.gradients['intercpt'] += dintercept
		return dX[0] if len(X) == 1 else dX

	def _bwd(self, dLdY, X):
		"""
			partial(L/x) = partial(L/hat(x))*frac{1}{sqrt(var+eps)} +
						   partial(L/mean)*paitial(mean/x) +
						   partial(L/var)*paitial(var/x)
		"""
		scaler = self.parameters['scaler']
		ep = self.hyperparameters['epsilon']

		n_ex, n_in = X.shape
		X_mean, X_var = X.mean(axis=0), X.var(axis=0)

		N = (X - X_mean) / np.sqrt(X_var + ep)
		dintercept = dLdY.sum(axis=0)
		dscaler = np.sum(dLdY * N, axis=0)

		dN = dLdY * scaler
		dX = (n_ex * dN - dN.sum(axis=0)) - N * (dN * N).sum(axis=0) / (
			n_ex * np.sqrt(X_var + ep) )
		return np.reshape(dX, X_shape), dScaler, dIntercept

class LayerNorm2D(LayerBase):
	def __init__(self, epsilon = 1e-5, optimizer=None):
		"""
		A layer normalization layer for 2D inputs with an additional channel
		dimension.

		Note:
		LayerNorm2D computes mean and var in features, rather than BatchNorm2D
		computes in minibatch.
		math:
			Y = scaler * norm(X) + intercept
			norm(X) = (X - mean(X)) / sqrt(var(X) + eps)
		""" 
		super().__init__(optimizer)
		self.in_ch = None
		self.out_ch = None
		self.epsilon = epsilon
		self.parameters = {"scaler": None, "intercept": None}
		self.is_initialized = False

	def _init_params(self, X_shape):
		n_ex, in_rows, in_cols, in_ch = X_shape
		scaler = np.random.rand(in_rows, in_cols, in_ch)
		intercept = np.zeros((in_rows, in_cols, in_ch))

		self.parameters = {"scaler": scaler, "intercept": intercept}

		self.gradients = {
			"scaler": np.zeros_like(scaler),
			"intercept": np.zeros_like(intercept),
		}

		self.is_initialized = True

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "LayerNorm2D",
			"act_fn": None,
			"in_ch": self.in_ch,
			"out_ch": self.out_ch,
			"epsilon": self.epsilon,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def forward(self, X, retain_derived=True):
		"""
		compute the layer output on a single minibatch
		Parameters
		----------
		X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
			Input volume containing the `in_rows` by `in_cols`-dimensional
			features for a minibatch of `n_ex` examples.
		retain_derived : bool
			Whether to retain the variables calculated during the forward pass
			for use later during backprop. If False, this suggests the layer
			will not be expected to backprop through wrt. this input. Default
			is True.
		Returns
		-------
		Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
			Layer output for each of the `n_ex` examples.
		"""
		if not self.is_initialized:
			self.in_ch = self.out_ch = X.shape[3]
			self._init_params(X.shape)

		scaler = self.parameters["scaler"]
		ep = self.hyperparameters["epsilon"]
		intercept = self.parameters["intercept"]

		if retain_derived:
			self.X.append(X)

		X_var = X.var(axis=(1, 2, 3), keepdims=True)
		X_mean = X.mean(axis=(1, 2, 3), keepdims=True)
		lnorm = (X - X_mean) / np.sqrt(X_var + ep)
		y = scaler * lnorm + intercept
		return y

	def backward(self, dLdY, retain_grads=True):
		"""
		Backprop from layer outputs to inputs.

		Parameters
		----------
		dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
			The gradient of the loss wrt. the layer output `Y`.
		retain_grads : bool
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.

		Returns
		-------
		dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
			The gradient of the loss wrt. the layer input `X`.
		"""
		assert self.trainable, 'layer is frozen'
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		dX = []
		X = self.X
		for dy, x in zip(dLdY, X):
			dx, dscaler, dintercept = self._bwd(dy, x)
			dX.append(dx)

			if retain_grads:
				self.gradients['scaler'] += dscaler
				self.gradients['intercpt'] += dintercept
		return dX[0] if len(X) == 1 else dX

	def _bwd(self, dLdY, X):
		scaler = self.parameters['scaler']
		ep = self.hyperparameters['epsilon']

		X_mean = X.mean(axis=(1,2,3), keepdims=True)
		X_var = X.var(axis=(1,2,3), keepdims=True)
		lnorm = (X - x_mean) / np.sqrt(X_var + ep)

		dLnorm = dLdY * scaler
		dIntercept = dy.sum(axis=0)
		dScaler = np.sum(dLdY * lnorm, axis=0)

		n_in = np.prod(X.shape[1:])
		lnorm = lnorm.reshape(-1, n_in)
		dLnorm = dLnorm.reshape(lnorm.shape)
		X_var = X_var.reshape(X_var.shape[:2])

		dX = (
				n_in * dLnorm 
				- dLnorm.sum(axis=1, keepdims=True)
				- lnorm * (dLnorm * lnorm).sum(axis=1, keepdims=True)
			) / (n_in * np.sqrt(X_var + ep))

		# reshape X gradients back to proper dimension
		return np.reshape(dX, X.shape), dScaler, dIntercept

class LayerNorm1D(LayerBase):
	def __init__(self, epsilon=1e-5, optimizer=None):
		"""
		A layer normalization layer for 1D inputs.
		Notes
		-----
		In contrast to :class:`BatchNorm1D`, the LayerNorm layer calculates the
		mean and variance across *features* rather than examples in the batch
		ensuring that the mean and variance estimates are independent of batch
		size and permitting straightforward application in RNNs.
		Equations [train & test]::
		    Y = scaler * norm(X) + intercept
		    norm(X) = (X - mean(X)) / sqrt(var(X) + epsilon)
		Also in contrast to :class:`BatchNorm1D`, `scaler` and `intercept` are applied
		*elementwise* to ``norm(X)``.
		Parameters
		----------
		epsilon : float
			A small smoothing constant to use during computation of ``norm(X)``
			to avoid divide-by-zero errors. Default is 1e-5.
		optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
			The optimization strategy to use when performing gradient updates
			within the :meth:`update` method.  If None, use the :class:`SGD
			<numpy_ml.neural_nets.optimizers.SGD>` optimizer with
			default parameters. Default is None.
		"""  # noqa: E501
		super().__init__(optimizer)
		self.n_in = None
		self.n_out = None
		self.epsilon = epsilon
		self.parameters = {"scaler": None, "intercept": None}
		self.is_initialized = False

	def _init_params(self):
		scaler = np.random.rand(self.n_in)
		intercept = np.zeros(self.n_in)

		self.parameters = {"scaler": scaler, "intercept": intercept}
		self.gradients = {
			"scaler": np.zeros_like(scaler),
			"intercept": np.zeros_like(intercept),
		}
		self.is_initialized = True

	@property
	def hyperparameters(self):
		 """Return a dictionary containing the layer hyperparameters."""
		 return {
			"layer": "LayerNorm1D",
			"act_fn": None,
			"n_in": self.n_in,
			"n_out": self.n_out,
			"epsilon": self.epsilon,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		 }

	def forward(self, X, retain_derived=True):
		"""
		Compute the layer output on a single minibatch.
		Parameters
		----------
		X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
			Layer input, representing the `n_in`-dimensional features for a
			minibatch of `n_ex` examples.
		retain_derived : bool
			Whether to retain the variables calculated during the forward pass
			for use later during backprop. If False, this suggests the layer
			will not be expected to backprop through wrt. this input. Default
			is True.
		Returns
		-------
		Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
			Layer output for each of the `n_ex` examples.
		"""
		if not self.is_initialized:
			self.n_in = self.n_out = X.shape[1]
			self._init_params()

		scaler = self.parameters["scaler"]
		ep = self.hyperparameters["epsilon"]
		intercept = self.parameters["intercept"]

		if retain_derived:
			self.X.append(X)

		X_mean, X_var = X.mean(axis=1, keepdims=True), X.var(axis=1, keepdims=True)
		lnorm = (X - X_mean) / np.sqrt(X_var + ep)
		y = scaler * lnorm + intercept
		return y

	def backward(self, dLdy, retain_grads=True):
		"""
		Backprop from layer outputs to inputs.
		Parameters
		----------
		dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
		    The gradient of the loss wrt. the layer output `Y`.
		retain_grads : bool
		    Whether to include the intermediate parameter gradients computed
		    during the backward pass in the final parameter update. Default is
		    True.
		Returns
		-------
		dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
		    The gradient of the loss wrt. the layer input `X`.
		"""
		assert self.trainable, "Layer is frozen"
		if not isinstance(dLdy, list):
			dLdy = [dLdy]

		dX = []
		X = self.X
		for dy, x in zip(dLdy, X):
			dx, dScaler, dIntercept = self._bwd(dy, x)
			dX.append(dx)
			if retain_grads:
				self.gradients["scaler"] += dScaler
				self.gradients["intercept"] += dIntercept
		return dX[0] if len(X) == 1 else dX

	def _bwd(self, dLdy, X):
		"""Computation of gradient of the loss wrt X, scaler, intercept"""
		scaler = self.parameters["scaler"]
		ep = self.hyperparameters["epsilon"]

		n_ex, n_in = X.shape
		X_mean, X_var = X.mean(axis=1, keepdims=True), X.var(axis=1, keepdims=True)

		lnorm = (X - X_mean) / np.sqrt(X_var + ep)
		dIntercept = dLdy.sum(axis=0)
		dScaler = np.sum(dLdy * lnorm, axis=0)

		dLnorm = dLdy * scaler
		dX = (
			n_in * dLnorm
			- dLnorm.sum(axis=1, keepdims=True)
			- lnorm * (dLnorm * lnorm).sum(axis=1, keepdims=True)
		) / (n_in * np.sqrt(X_var + ep))
		return dX, dScaler, dIntercept

#######################################################################
#                             MLP Layers                              #
#######################################################################
class Embedding(LayerBase):
	def __init__(self, n_out, vocal_size, pool=None, init='glorot_uniform', optimizer=None):
		"""
		embedding layer

		math:
			Y = W[X]
		NB. This layer must be the first in a neural network as the gradients
		do not get passed back through to the inputs.

		Parameters
		----------
		n_out : int
			The dimensionality of the embeddings
		vocab_size : int
			The total number of items in the vocabulary. All integer indices
			are expected to range between 0 and `vocab_size - 1`.
		pool : {'sum', 'mean', None}
			If not None, apply this function to the collection of `n_in`
			encodings in each example to produce a single, pooled embedding.
			Default is None.
		init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
			The weight initialization strategy. Default is `'glorot_uniform'`.
		optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
			The optimization strategy to use when performing gradient updates
			within the :meth:`update` method.  If None, use the :class:`SGD
			<numpy_ml.neural_nets.optimizers.SGD>` optimizer with
			default parameters. Default is None.
		"""
		super().__init__()
		fstr = "'pool' must be either 'sum', 'mean', or None but got '{}'"
		assert pool in ["sum", "mean", None], fstr.format(pool)

		self.init = init
		self.pool = pool
		self.n_out = n_out
		self.vocab_size = vocab_size
		self.parameters = {'W': None}
		self.is_initialized = False
		self._init_params()

	def _init_params(self):
		init_weights = WeightInitializer("Affine(slope=1, intercept=0", mode=self.init)
		W = init_weights((self.vocab_size, self.n_out))

		self.parameters = {'W': W}
		self.derived_variables = {}
		self.gradients = {'W': np.zeros_like(W)}
		self.is_initialized = True

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "Embedding",
			"init": self.init,
			"pool": self.pool,
			"n_out": self.n_out,
			"vocab_size": self.vocab_size,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def lookup(self, ids):
		"""
		return the embeddings associated with the IDs in 'ids'

		Parameters:
		word_ids : :py:class:`ndarray <numpy.ndarray>` of shape (`M`,)
			An array of `M` IDs to retrieve embeddings for.

		Returns:
		embeddings: size (M, n_out)
			the embedding vectors for each of the M IDs
		"""
		return self.parameters['W']['id']

	def forward(self, X, retain_derive=True):
		"""
		compute the layer output on a single minibatch

		Math:
			Y = W[x]

		Parameters:
		X: array shape (n_ex, n_in), list of length 'n_ex' layer input, 
			representing a minibatch of n_ex examples. If
			``self.pool`` is None, each example must consist of exactly `n_in`
			integer token IDs. Otherwise, `X` can be a ragged array, with each
			example consisting of a variable number of token IDs.
		retain_derived : bool
			Whether to retain the variables calculated during the forward pass
			for use later during backprop. If False, this suggests the layer
			will not be expected to backprop through with regard to this input.
			Default is True.
		Returns
		-------
		Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_out)`
			Embeddings for each coordinate of each of the `n_ex` examples
		"""
		# if X is a ragged array
		if isinstance(X, List) and not issubclass(X[0].dtype.type, np.integer):
			fstr = "Input to embedding layer must be an array of integers, got '{}'"
			raise TypeError(fstr.format(X[0].dtype.type))

		# Otherwise
		if isinstance(X, np.ndarray) and not issubclass(X.dtype.type, np.integer):
			fstr = "Input to Embedding layer must be an array of integers, got '{}'"
			raise TypeError(fstr.format(X.dtype.type))

		Y = self._fwd(X)
		if retain_derived:
			self.X.append(X)
		return Y

	def _fwd(self, X):
		W = self.parameters['W']
		if self.pool is None:
			emb = W[X]
		elif self.pool == 'sum':
			emb = np.array([W[x].sum(axis=0) for x in X])[:, None, :]
		elif self.pool == 'mean':
			emb = np.array([W[x].mean(axis=0) for x in X])[:, None, :]
		return emb

	def backward(self, dLdY, retain_grads=True):
		"""
		Backprop from layer outputs to embedding weights.

		Notes
		-----
		Because the items in `X` are interpreted as indices, we cannot 
		compute the gradient of the layer output wrt. `X`.

		Parameters
		----------
		dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_out)` or list of arrays
			The gradient(s) of the loss wrt. the layer output(s)
		retain_grads : bool
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.
		"""
		assert self.trainable, 'Layer is frozen'
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		for dy, x in zip(dLdY, self.X):
			dw = self._bwd(dy, x)

			if retain_grads:
				self.gradients['W'] += dw

	def _bwd(self, dLdY, X):
		dW = np.zeros_like(-1, self.n_out)
		dLdY = dLdY.reshape(-1, self.n_out)

		if self.pool is None:
			for ix, v_id in enumerate(X.flatten()):
				dW[v_id] += dLdY[ix]
		elif self.pool == 'sum':
			for ix, v_id in enumerate(X):
				dW[v_id] += dLdY[ix]
		elif self.pool == 'mean':
			for ix, v_id in enumerate(X):
				dW[v_id] += dLdY[ix] / len(v_ids)
		return dW

class FullyConnected(LayerBase):
	def __init__(self, n_out, act_fn=None, init='glorot_uniform', optimizer=None):
		"""
		A fully connected (dense) layer

		math:
			Y = f(wx + b)
			f() is the activation function

		parameters:
		n_out: int
			the dimensionality of the layer output
		act_fn:
			activation function
		init: 
			weights init funciton
		optimizer:
		"""
		super().__init__(optimizer)
		self.init = init
		self.n_in = None
		self.n_out = n_out
		self.act_fn = ActivationInitializer(act_fn)()
		self.parameters = {'W': None, 'b': None}
		self.is_initialized = False

	def _init_parameters(self):
		init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
		b = np.zeros((1, self.n_out))
		W = init_weights((self.n_in, self.n_out))
		self.parameters = {'W':W, 'b':b}
		self.derived_variables = {'Z': []}
		self.gradients = {'W': np.zeros_like(W), 'b':np.zeros_like(b)}
		self.is_initialized = True

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "FullyConnected",
			"init": self.init,
			"n_in": self.n_in,
			"n_out": self.n_out,
			"act_fn": str(self.act_fn),
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def forward(self, X, retain_derived=True):
		"""
		Compute the layer output on a single minibatch.
		Parameters
		----------
		X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
			Layer input, representing the `n_in`-dimensional features for a
			minibatch of `n_ex` examples.
		retain_derived : bool
			Whether to retain the variables calculated during the forward pass
			for use later during backprop. If False, this suggests the layer
			will not be expected to backprop through wrt. this input. Default
			is True.
		Returns
		-------
		Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
			Layer output for each of the `n_ex` examples.
		"""
		if not self.is_initialized:
			self.n_in = X.shape[1]
			self._init_params()

		Y, Z = self._fwd(X)

		if retain_derived:
			self.X.append(X)
			self.derived_variables['Z'].append('Z')
		return Y

	def _fwd(self, X):
		W = self.parameters['W']
		b = self.parameters['b']

		Z = X @ W + b
		Y = self.act_fn(Z)
		return Y, Z

	def backward(self, dLdY, retain_grads=True):
		"""
		Backprop from layer outputs to inputs.
		Parameters
		----------
		dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)` or list of arrays
			The gradient(s) of the loss wrt. the layer output(s).
		retain_grads : bool
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.
		Returns
		-------
		dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)` or list of arrays
			The gradient of the loss wrt. the layer input(s) `X`.
		"""  # noqa: E501
		assert self.trainable, 'layer is frozen'
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		dX = []
		X = self.X
		for dy, x in zip(dLdY, X):
			dx, dw, db = self._bwd(dy, x)
			dX.append(dx)

			if retain_grads:
				self.gradients['W'] += dw
				self.gradients['b'] += db

		return dX[0] if len(X) == 1 else dX

	def _bwd(self, dLdY, X):
		W = self.parameters['W']
		b = self.parameters['b']

		Z = X @ W + b
		dZ = dLdy * self.act_fn.grad(Z)

		dX = dZ @ W.T
		dW = X.T @ dZ
		dB = dZ.sum(axis=0, keepdims=True)
		return dX, dW, dB

class Softmax(LayerBase):
	def __init__(self, dim=1, optimizer=None):
		"""
		A softmax nonlinearity layer.

		math:
			y_i = frac{e^{x_i}}{sum_j e^{x_j}}
		"""
		super().__init__(optimizer)
		self.dim = dim
		self.n_in = None
		self.is_initialized = False

	def _init_params(self):
		self.gradients = {}
		self.parameters = {}
		self.derived_variables = {}
		self.is_initialized = True

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "SoftmaxLayer",
			"n_in": self.n_in,
			"n_out": self.n_in,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def forward(self, X, retain_derived=True):
		"""
		compute the layer output on a single minibatch

		parameters:
		X: (n_ex, n_in)
		retain_derived: 

		return:
		Y: (n_ex, n_out)
		"""
		if not self.is_initialized:
			self.n_in = X.shape[1]
			self._init_params()

		Y = self._fwd(X)

		if retain_derived:
			self.X.append(X)
		return Y

	def _fwd(self, X):
		e_X = np.exp(X - np.max(X, axis=self.dim, keepdims=True))
		return e_X / e_X.sum(axis=self.dim, keepdims=True)

	def backward(self, dLdY, retain_grads=True):
		"""
		backprob from layer outputs to inputs

		math:
			S(i) = softmax(xi)
			J[i, j] = 
				softmax(xi) * (1 - softmax(xi)) i  = j
				-softmax(xi) * softmax(xj)		i != j
		"""
		dX = []
		for dy, x in zip(dLdY, X):
			dxi = []
			for dyi, xi in zip(*np.atleast_2d(dy, x)):
				yi = self._fwd(xi.reshape(1, -1)).reshape(-1, 1)
				dyidxi = np.diagflat(yi) - yi @ yi.T
				dxi.append(dyi @ dyidxi)
			dX.append(dxi)
		return np.array(dX).reshape(*X.shape)

class Sigmoid(LayerBase):
	def __init__(self, optimizer=None):
		"""
		A softmax nonlinearity layer.

		math:
			y_i = frac{e^{x_i}}{sum_j e^{x_j}}
		"""
		super().__init__(optimizer)
		self.dim = dim
		self.n_in = None
		self.is_initialized = False

	def _init_params(self):
		self.gradients = {}
		self.parameters = {}
		self.derived_variables = {}
		self.is_initialized = True

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "SigmoidLayer",
			"n_in": self.n_in,
			"n_out": self.n_in,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def forward(self, X, retain_derived=True):
		"""
		compute the layer output on a single minibatch

		parameters:
		X: (n_ex, n_in)
		retain_derived: 

		return:
		Y: (n_ex, n_out)
		"""
		if not self.is_initialized:
			self.n_in = X.shape[1]
			self._init_params()

		Y = self._fwd(X)

		if retain_derived:
			self.X.append(X)
		return Y

	def _fwd(self, X):
		e_X = np.exp(-X)
		return 1. / (1. + e_X)

	def backward(self, dLdY, retain_grads=True):
		"""
		backprob from layer outputs to inputs

		math:
			S(i) = softmax(xi)
			J[i, j] = 
				softmax(xi) * (1 - softmax(xi)) i  = j
				-softmax(xi) * softmax(xj)		i != j
		"""
		dX = []
		for dy, x in zip(dLdY, X):
			dxi = []
			for dyi, xi in zip(*np.atleast_2d(dy, x)):
				yi = self._fwd(xi.reshape(1, -1)).reshape(-1, 1)
				dyidxi = yi * (1 - yi)
				dxi.append(dyi @ dyidxi)
			dX.append(dxi)
		return np.array(dX).reshape(*X.shape)

#######################################################################
#                        Convolutional Layers                         #
#######################################################################
class Conv1D(LayerBase):
	def __init__(self, out_ch, kernel_width, pad=0, stride=1, dilation=0,
			act_fn=None, init='glorot_uniform', optimizer=None):
	"""
	apply a one-dimentional Convolutional kernel over an input

	Math:
		out = act_fn(pad(x) * W + b)
		out_dims = floor(1 + (n_row_in + pad_left + pad_right - kernel_width_dilation) / stride)
		kernel_width_dilation = 1 + d(kernel_width - 1)

	parameters:
	out_ch: int
		the number of filters/kernels to compute in the current layer
	kernel_width: int
		the width of a single 1D filter/kernel in the current layer
	act_fn: 
		activation function
	pad: int, tuple,
	stride: int
	dilation: int
		the number of pixels inserted between kernel elements. 
		the size of receipt filed: 1 + d * (kernel_width -1)
	init: {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
		the weight initialization strategy
	optimizer:
	"""
		super().__init__()
		self.pad = pad
		self.init = init
		self.in_ch = None
		self.out_ch = out_ch
		self.stride = stride
		self.dilation = dilation
		self.kernel_width = kernel_width
		self.act_fn = ActivationInitializer(act_fn)()
		self.parameters = {'W':None, 'b':None}
		self.is_initialized = False

	def _init_params(self):
		init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
		W = init_weights((self.kernel_width, self.in_ch, self.out_ch))
		b = np.zeros((1, 1, self.out_ch))

		self.parameters = {"W": W, "b": b}
		self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
		self.derived_variables = {"Z": [], "out_rows": [], "out_cols": []}
		self.is_initialized = True

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "Conv1D",
			"pad": self.pad,
			"init": self.init,
			"in_ch": self.in_ch,
			"out_ch": self.out_ch,
			"stride": self.stride,
			"dilation": self.dilation,
			"act_fn": str(self.act_fn),
			"kernel_width": self.kernel_width,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def forward(self, X, retain_derived=True):
		"""
		compute the layer output given input volume

		parameters:
		X: input array, (n_ex, l_in, in_ch)
			n_ex: number of examples; each of length 'l_in' and with 'in_ch' input channels
		retain_derived : bool
			Whether to retain the variables calculated during the forward pass
			for use later during backprop. If False, this suggests the layer
			will not be expected to backprop through wrt. this input. Default
			is True.

		returns:
		Y: (n_ex, l_out, out_ch)
		"""
		if not self.is_initialized:
			self.in_ch = X.shape[2]
			self._init_params()

		W = self.parameters['W']
		b = self.parameters['b']

		n_ex, l_in, in_ch = X.shape
		s, p, d = self.stride, self.pad, self.dilation

		# pad the input and perform the forward Convolution
		Z = conv1D(X, W, s, p, d) + b
		Y = self.act_fn(Z)

		if retain_derived:
			self.X.append(X)
			self.derived_variables['Z'].append(Z)
			self.derived_variables["out_rows"].append(Z.shape[1])
			self.derived_variables["out_cols"].append(Z.shape[2])

	def backward(self, dLdY, retain_grads=True):
		"""
		compute the gradients of the loss with respect to the layer parameters

		Parameters
		----------
		dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_out, out_ch)` or list of arrays
			The gradient(s) of the loss with respect to the layer output(s).
		retain_grads : bool
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.
		Returns
		-------
		dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_in, in_ch)`
			The gradient of the loss with respect to the layer input volume.
		"""
		assert self.trainable, 'Layer is frozen'
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		X = self.X
		Z = self.derived_variables['Z']

		dX = []
		for dy, x, z in zip(dLdY, X, Z):
			dx, dw, db = self._bwd(dy, x, z)
			dX.append(dx)

			if retain_grads:
				self.gradients['W'] += dw
				self.gradients['b'] += db

		return dX[0] if len

	self _bwd(self, dLdY, X, Z):
		W = self.parameters['W']

		# add a row dimension to X, W, and dZ to permit us to use im2col/col2im
		X2D = np.expand_dims(X, axis=1)
		W2D = np.expand_dims(W, axis=0)
		dLdZ = np.expand_dims(dLdY * self.act_fn.grad(Z), axis=1)

		d = self.dilation
		fr, fc, in_ch, out_ch = W2D.shape
		n_ex, l_out, out_ch = dLdY.shape
		fr, fc, s = 1, self.kernel_width, self.stride

		# use pad1D here in order to correctly handle self.pad = 'causal'
		# which is not defined for pad2D
		_, p = pad1D(X, self.pad, self.kernel_width, s, d)
		p2D = (0, 0, p[0], p[1])

		# columnize W, X, and dLdY
		dLdZ_col = dLdZ.transpose(3, 1, 2, 0).reshape(out_ch, -1)
		W_col = W2D.transpose(3, 2, 0, 1).reshape(out_ch, -1).T
		X_col, _ = im2col(X2D, W2D.shape, p2D, s, d)

		# compute gradients through matrix multiplication and reshape
		dB = dLdZ_col.sum(axis=1).reshape(1, 1, -1)
		dW = (dLdZ_col @ X_col.T).reshape(out_ch, in_ch, fr, fc).transpose(2, 3, 1, 0)

		# reshape columnized dX back into the same format as the input volume
		dX_col = W_col @ dLdZ_col
		dX = col2im(dX_col, X2D.shape, W2D.shape, p2D, s, d).transpose(0, 2, 3, 1)

		return np.sequeeze(dX, axis=1), np.sequeeze(dW, axis=0), dB

	def _backward_naive(self, dLdY, retain_grads=True):
		"""
		A slower (ie., non-vectorized) but more straightforward implementation
		of the gradient computations for a 2D conv layer.
		"""
		assert self.trainable, "Layer is frozen"
		if not isinstance(dLdy, list):
		    dLdy = [dLdy]

		W = self.parameters["W"]
		b = self.parameters["b"]
		Zs = self.derived_variables["Z"]

		Xs, d = self.X, self.dilation
		fw, s, p = self.kernel_width, self.stride, self.pad

		dXs = []
		for X, Z, dy in zip(Xs, Zs, dLdY):
			n_ex, l_out, out_ch = dy.shape
			X_pad, (pr1, pr2) = pad1D(X, p, self.kernel_width, s, d)

			dX = np.zeros_like(X_pad)
			dZ = dy * self.act_fn.grad(Z)

			dW, dB = np.zeros_like(W), np.zeros_like(b)
			for m in range(n_ex):
				for i in range(l_out):
					for j in range(out_ch):
						# compute window boundaries w. stride and dilation
						i0, i1 = i * s, (i*s) + fw * (d + 1) - d

						wc = W[:, :, c]
						kernel = dZ[m, i, c]
						window = X_pad[m, i0:i1:(d+1), :]

						dB[:, :, c] += kernel
						dW[:, :, c] += window * kernel
						dX[m, i0 : i1 : (d+1), :] += wc * kernel

			if retain_grads:
				self.gradients['W'] += dW
				self.gradients['b'] += dB

			pr2 = None if pr2 == 0 else -pr2
			dXs.append(dX[:, pr1:pr2, :])
		return dXs[0] if len(Xs) == 1 else dXs

class Conv2D(LayerBase):
	def __init__(self, out_ch, kernel_shape, pad=0, stride=1, dilation=0,
		act_fn=None, optimizer=None, init="glorot_uniform"):
		"""
		Apply a two-dimensional convolution kernel over an input volume.
		Notes
		-----
		Equations::
			out = act_fn(pad(X) * W + b)
			n_rows_out = floor(1 + (n_rows_in + pad_left + pad_right - filter_rows) / stride)
			n_cols_out = floor(1 + (n_cols_in + pad_top + pad_bottom - filter_cols) / stride)
		where `'*'` denotes the cross-correlation operation with stride `s` and
		dilation `d`.
		Parameters
		----------
		out_ch : int
			The number of filters/kernels to compute in the current layer
		kernel_shape : 2-tuple
			The dimension of a single 2D filter/kernel in the current layer
		act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
			The activation function for computing ``Y[t]``. If None, use the
			identity function :math:`f(X) = X` by default. Default is None.
		pad : int, tuple, or 'same'
			The number of rows/columns to zero-pad the input with. Default is 0.
		stride : int
			The stride/hop of the convolution kernels as they move over the
			input volume. Default is 1.
		dilation : int
			Number of pixels inserted between kernel elements. Effective kernel
			shape after dilation is: ``[kernel_rows * (d + 1) - d, kernel_cols
			* (d + 1) - d]``. Default is 0.
		init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
			The weight initialization strategy. Default is `'glorot_uniform'`.
		optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
			The optimization strategy to use when performing gradient updates
			within the :meth:`update` method.  If None, use the :class:`SGD
			<numpy_ml.neural_nets.optimizers.SGD>` optimizer with
			default parameters. Default is None.
		"""  # noqa: E501
		super().__init__(optimizer)
		self.pad = pad
		self.init = init
		self.in_ch = None
		self.out_ch = out_ch
		self.stride = stride
		self.dilation = dilation
		self.kernel_shape = kernel_shape
		self.act_fc = ActivationInitializer(act_fn)()
		self.parameters = {'W': None, 'b': None}
		self.is_initialized = False

	def _init_params(self):
		init_weights = WeightInitializer(str(self.act_fn), mode=self.init)

		fr, fc = self.kernel_shape
		w = init_weights((fr, fc, self.in_ch, self.out_ch))
		b = np.zeros((1, 1, 1, self.out_ch))

		self.parameters = {'W': W, 'b': b}
		self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
		self.derived_variables = {'Z':[], 'out_rows':[], 'out_cols':[]}
		self.is_initialized = True


	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "Conv2D",
			"pad": self.pad,
			"init": self.init,
			"in_ch": self.in_ch,
			"out_ch": self.out_ch,
			"stride": self.stride,
			"dilation": self.dilation,
			"act_fn": str(self.act_fn),
			"kernel_shape": self.kernel_shape,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def forward(self, X, retain_derived=True):
		"""
		compute the layer output given input volumn

		parameters:
			X: (n_ex, in_rows, in_cols, in_ch)
		retain_derived : bool
			Whether to retain the variables calculated during the forward pass
			for use later during backprop. If False, this suggests the layer
			will not be expected to backprop through wrt. this input. Default
			is True.
		returns:
		Y: (n_ex, out_rows, out_cols, out_ch)
		"""
		self not self.is_initialized:
			self.in_ch = X.shape[3]
			self._init_params()

		W = self.parameters['W']
		b = self.parameters['b']

		n_ex, in_rows, in_cols, in_ch = X.shape
		s, p, d = self.stride, self.pad, self.dilation

		# pad the input and perform the forward convolution
		Z = conv2D(X, W, s, p, d) + b
		Y = self.act_fn(Z)

		if retain_derived:
			self.X.append(X)
			self.derived_variables["Z"].append(Z)
			self.derived_variables["out_rows"].append(Z.shape[1])
			self.derived_variables["out_cols"].append(Z.shape[2])
		return Y

	def backward(self, dLdY, retain_grads=True):
		"""
		compute the gradient of the loss with respect to the layer parameters

		Parameters
		----------
		dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows,
		out_cols, out_ch)` or list of arrays
			The gradient(s) of the loss with respect to the layer output(s).
		retain_grads : bool
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.
		Returns
		-------
		dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
			The gradient of the loss with respect to the layer input volume.
		"""
		assert self.trainable, 'layer is forzen'
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		dX = []
		X = self.X
		Z = self.derived_variables['Z']

		for dy, x, z in zip(dLdY, X, Z):
			dx, dw, dz = self._bwd(dy, x, z)
			dX.append(dx)

			if retain_grads:
				self.gradients['W'] += dw
				self.gradients['b'] += db

		return dX[0] if len(X) == 1 else dX

	def _bwd(self, dLdY, X, Z):
		W =self.parameters['W']

		d = self.dilation
		fr, fc, in_ch, out_ch = W.shape
		n_ex, out_rows, out_cols, out_ch = dLdY.shape
		(fr, fc), s, p = self.kernel_shape, self.stride, self.pad

		# columnize W, X, and dLdY
		dLdZ = dLdY * self.act_fn.grad(Z)
		dLdZ_col = dLdZ.transpose(3, 1, 2, 0).reshape(out_ch, -1)
		W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1).T
		X_col, p = im2col(X, W.shape, p, s, d)

		# compute graidents via matrix multiplication and reshap3
		dB = dLdZ_col.sum(axis=1).reshape(1, 1, 1, -1)
		dW = (dLdZ_col @ X_col.T).reshape(out_ch, in_ch, fr, fc).transpose(2, 3, 1, 0)

		# reshape columnized dX back into the same format as the input volume
		dX_col = W_col @ dLdZ_col
		dX = col2im(dX_col, X.shape, W.shape, p, s, d).transpose(0, 2, 3, 1)
		return dX, dW, dB

	def _backward_naive(self, dLdY, retain_grads=True):
		assert self.trainable, "Layer is frozen"
		if not isinstance(dLdy, list):
			dLdy = [dLdy]

		W = self.parameters["W"]
		b = self.parameters["b"]
		Zs = self.derived_variables["Z"]

		Xs, d = self.X, self.dilation
		(fr, fc), s, p = self.kernel_shape, self.stride, self.pad

		dXs = []
		for X, Z, dy in zip(Xs, Zs, dLdY):
			n_ex, out_rows, out_cols, out_ch = dy.shape
			X_pad, (pr1, pr2, pc1, pc2) = pad2D(X, p, self.kernel_shape, s, d)

			dZ = dLdY * self.act_fn.grad(Z)
			dX = np.zeros_like(X_pad)
			dW, dB = np.zeros_like(W), np.zeros_like(b)
			for m in range(n_ex):
				for i in range(out_rows):
					for j in range(out_cols):
						for c in range(out_ch):
							# compute window boundaries w. stride and dilation
							i0, i1 = i * s, (i * s) + fr * (d + 1) - d
							j0, j1 = j * s, (j * s) + fc * (d + 1) - d

							wc = W[:, :, :, c]
							kernel = dZ[m, i, j, c]
							window = X_pad[m, i0 : i1 : (d+1), j0 : j1 : (d+1), :]

							dB[:, :, :, c] += kernel
							dW[:, :, :, c] += window * kernel
							dX[m, i0 : i1 : (d+1), j0 : j1 : (d+1), :] += (
								wc * kernel
								)
		if retain_grads:
			self.gradients['W'] += dW
			self.gradients['b'] += dB

		pr2 = None if pr2 == 0 else -pr2
		pc2 = None if pc2 == 0 else -pc2
		dXs.append(dX[:, pr1:pr2, pc1:pc2, :])
	return dXs[0] if len(Xs) == 1 else dXs

class Pool2D(LayerBase):
	def __init__(self, kernel_shape, stride=1, pad=0, mode='max', optimizer=None):
		"""
		a single two-dimensional pooling layer

		parameters:
		kernel_shape: 2-tuple
			the dimension of a single 2D filter/kernel in the current Layer
		stride: int
			the stride of the convolution kernels as tbey move over the input volumn.
		pad: int, tuple, or 'same'
		mode: {'max', 'average'}
			the pooling function to apply
		optimizer: 
		"""
		super().__init__(optimizer)
		self.pad = pad
		self.mode = mode
		self.in_ch = None
		self.out_ch = None
		self.stride = stride
		self.kernel_shape = kernel_shape
		self.is_initialized = False

	def _init_params(self):
		self.derived_variables = {'out_rows': [], 'out_cols': []}
		self.is_initialized = True

	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "Pool2D",
			"act_fn": None,
			"pad": self.pad,
			"mode": self.mode,
			"in_ch": self.in_ch,
			"out_ch": self.out_ch,
			"stride": self.stride,
			"kernel_shape": self.kernel_shape,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def forward(self, X, retain_derived=True):
		"""
		compute the layer output given input X

		parameters:
		X: array. (n_ex, in_rows, in_cols, in_ch)
			the input consisting of 'n_ex' samples, each with dimension 
			(in_rows, in_cols, in_ch)
		retain_derived: bool
		"""
		self not self.is_initialized:
			self.in_ch = X.shape[3]
			self._init_params()

		n_ex, in_rows, in_cols, in_ch = X.shape
		(fr, fc), s, p, d = self.kernel_shape, self.stride, self.pad, self.dilation
		X_pad, (pr1, pr2, pc1, pc2) = pad2D(X, p, self.kernel_shape, s)

		out_rows = np.floor(1 + (in_rows + pr1 + pr2 - fr) / s).astype(int)
		out_cols = np.floor(1 + (in_cols + pc1 + pc2 - fc) / s).astype(int)

		if self.mode == 'max':
			pool_fn = np.max
		elif self.mode == 'average':
			pool_fn = np.mean

		Y = np.zeros((n_ex, out_rows, out_cols, self.out_ch))
		for m in range(n_ex):
			for i in range(out_rows):
				for j in range(out_cols):
					for c in range(self.out_ch):
						i0, i1 = j * s, (j * s) + fr
						j0, j1 = j * s, (j * s) + fc

						x1 = X_pad[m, i0:i1, j0:j1, c]
						Y[m, i, j ,c] = pool_fn(x1)

		if retain_derived:
			self.X.append(X)
			self.derived_variables['out_rows'].append(out_rows)
			self.derived_variables['out_cols'].append(out_cols)
		return Y

	def backward(self, dLdY, retain_grads=True):
		"""
		backprop from layer outputs to inputs

		parameters:
		dLdY: (n_ex, in_rows, in_cols, in_ch)
			the gradients of the loss 
		retain_grads : bool
			Whether to include the intermediate parameter gradients computed
			during the backward pass in the final parameter update. Default is
			True.

		returns:
		dX: (n_ex, in_rows, in_cols, in_ch)
		"""
		assert self.trainable, 'layer is frozen'
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		Xs = self.X
		out_rows = self.derived_variables['out_rows']
		out_cols = self.derived_variables['out_cols']

		(fr, fc), s, p = self.kernel_shape, self.stride, self.pad
		dXs = []
		for X, dy, out_row, out_col in zip(Xs, dLdY, out_rows, out_cols):
			n_ex, in_rows, in_cols, nc_in = X.shape
			X_pad, (pr1, pr2, pc1, pc2) = pad2D(X, p, self.kernel_shape, s)

			dX = np.zeros_like(X_pad)
			for m in range(n_ex):
				for i in range(out_rows):
					for j in range(out_cols):
						for c in range(self.out_ch):
							# compute window boundaries, incorporating stride
							i0, i1 = i * s, (i * s) + fr
							j0, j1 = j * s, (j * s) + fc

							if self.mode == 'max':
								xi = X[m, i0:i1, j0:j1, c]

								# pass max position grads
								# others are 0
								mask = np.zeros_like(xi).astype(bool)
								x, y = np.argwhere(xi == np.max(xi))[0]
								mask[x, y] = True

								dX[m, i0:i1, j0:j1, c] += mask * dy[m, i, j, c]
							elif self.mode == 'average':
								frame = np.ones((fr, fc)) * dy[m, i, j, c]
								dX[m, i0:i1, j0:j1, c] += frame / np.prod((fr, fc))

			pr2 = None if pr2 == 0 else -pr2
			pc2 = None if pc2 == 0 else -pc2
			dXs.append(dX[:, pr1:pr2, pc1:pc2, :])
		return dXs[0] if len(Xs) == 1 else dXs

class Deconv2D(LayerBase):
	def __init__(self, out_ch, kernel_shape, pad=0, stride=1, 
			act_fn=None, optimizer=None, init='glorot_uniform'):
		"""
		apply a two-dimensional 'deconvolution' to an input volumn

		Notes:
		this layer is computing a transposed convolution / fractionally-stride
		convolution.
		"""
		super().__init__(optimizer)
		self.pad = pad
		self.init = init
		self.in_ch = None
		self.stride = stride
		self.out_ch = out_ch
		self.kernel_shape = kernel_shape
		self.act_fn = ActivationInitializer(act_fn)()
		self.parameters = {'W': None, 'b': None}
		self.is_initialized = False

	def _init_params(self):
		init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
		fr, fc = self.kernel_shape
		W = init_weights((fr, fc, self.in_ch, self.out_ch))
		b = np.zeros((1, 1, 1, self.out_ch))

		self.parameters = {'W': W, 'b': b}
		self.gradients = {"W": np.zeros_like(W), 'b': np.zeros_like(b)}
		self.derived_variables = {'Z': [], 'out_rows': [], 'out_cols': []}
		self.is_initialized = False


	@property
	def hyperparameters(self):
		"""Return a dictionary containing the layer hyperparameters."""
		return {
			"layer": "Deconv2D",
			"pad": self.pad,
			"init": self.init,
			"in_ch": self.in_ch,
			"out_ch": self.out_ch,
			"stride": self.stride,
			"act_fn": str(self.act_fn),
			"kernel_shape": self.kernel_shape,
			"optimizer": {
				"cache": self.optimizer.cache,
				"hyperparameters": self.optimizer.hyperparameters,
			},
		}

	def forward(self, X, retain_derived=True):
		"""
		compute the layer output

		parameters:
		X: (n_ex, in_rows, in_cols, in_ch)
		retain_derived: bool

		returns:
		Y: (n_ex, out_rows, out_cols, out_ch)
		"""
		if not self.is_initialized:
			self.in_ch = X.shape[3]
			self._init_params()

		W = self.parameters['W']
		b = self.parameters['b']
		s, p = self.stride, self.pad
		n_ex, in_rows, in_cols, in_ch = X.shape

		# pad the input and perform the forward deconvolution
		Z = deconv2D_naive(X, W, s, p ,0) + b
		Y = self.act_fn(Z)

		if retain_derived:
			self.X.append(X)
			self.derived_variables['Z'].append(Z)
			self.derived_variables['out_rows'].append(Z.shape[1])
			self.derived_variables['out_cols'].append(Z.shape[2])
		return Y

	def backward(self, dLdY, retain_grads=True):
		"""
		compute the gradient of the loss with respect to the layer parameters
		Note:
		rely on: im2col, and col2im

		parameters:
		dLdY: (n_ex, out_rows, out_cols, out_ch)
			the gradient of the loss with respect to the layer output
		retain_grads:
			whether to include the intermediate parameter gradients computed 
			during the backward pass in the final parameter update. Default is 
			True.

		returns:
		dx, (n_ex, in_rows, in_cols, in_ch)
		the gradient of the loss with respect to the layer input
		"""
		assert self.trainable, 'layer is frozen'
		if not isinstance(dLdY, list):
			dLdY = [dLdY]

		dX = []
		X, Z = self.X, self.derived_variables['Z']
		for dy, x, z in zip(dLdY, X, Z):
			dx, dw, db = self._bwd(dy, x, z)
			dX.append(dx)

			if retain_grads:
				self.gradient['W'] += dw
				self.gradient['b'] += db
		return dX[0] if len(X) == 1 else dX

	def _bwd(self, dLdY, X, Z):
		W = np.rot90(self.parameters['W'], 2)
		s = self.stride
		if self.stride > 1:
			X = dilate(X, s-1)
			s = 1

		fr, fc, in_ch, out_ch = W.shape
		(fr, fc), p = self.kernel_shape, self.pad
		n_ex, out_rows, out_cols, out_ch = dLdY.shape

		# pad X the first time
		X_pad, p = pad2D(X, p, W.shape[:2], s)
		n_ex, in_rows, in_cols, in_ch = X_pad.shape
		pr1, pr2, pc1, pc2 = p

		# compute additional padding to produce the deconvolution 
		out_rows = s * (in_rows - 1) - pr1 - pr2 + fr
		out_clos = s * (in_cols - 1) - pc1 - pc2 + fc
		out_dim = (out_rows, out_cols)

		# add additional "deconvolution" padding
		_p = calc_pad_dims_2D(X_pad.shape, out_dim, W.shape[:2], s, 0)
		X_pad, _ = pad2D(X_pad, _p, W.shape[:2], s)

		# columize W, X, and dLdY
		dLdZ = dLdY * self.act_fn.grad(Z)
		dLdZ, _ = pad2D(dLdZ, p, W.shape[:2], s)

		dLdZ_col = dLdZ.transpose(3, 1, 2, 0).reshape(out_ch, -1)
		W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1)
		X_col, _ = im2col(X_pad, W.shape, 0, s, 0)

		# compute gradients vias matrix multiplication and reshape
		dB = dLdZ_col.sum(axis=1).reshape(1, 1, 1, -1)
		dW = (dLdZ_col @ X_col.T).reshape(out_ch, in_ch, fr, fc).transpose(2, 3, 1, 0)
		dW = np.rot90(dW, 2)

		# reshape columnized dX back into the same format as the input volume
		dX_col = W_col.T @ dLdZ_col

		total_pad = tuple(i + j for i, j in zip(p, _p))
		dX = col2im(dX_col, X.shape, W.shape, total_pad, s, 0).transpose(0, 2, 3, 1)
		dX = dX[:, :: self.stride, :: self.stride, :]

		return dX, dW, dB

#######################################################################
#                          Recurrent Layers                           #
#######################################################################


class RNNCell(LayerBase):
    def __init__(self, n_out, act_fn="Tanh", init="glorot_uniform", optimizer=None):
        r"""
        A single step of a vanilla (Elman) RNN.
        Notes
        -----
        At timestep `t`, the vanilla RNN cell computes
        .. math::
            \mathbf{Z}^{(t)}  &=
                \mathbf{W}_{ax} \mathbf{X}^{(t)} + \mathbf{b}_{ax} +
                    \mathbf{W}_{aa} \mathbf{A}^{(t-1)} + \mathbf{b}_{aa} \\
            \mathbf{A}^{(t)}  &=  f(\mathbf{Z}^{(t)})
        where
        - :math:`\mathbf{X}^{(t)}` is the input at time `t`
        - :math:`\mathbf{A}^{(t)}` is the hidden state at timestep `t`
        - `f` is the layer activation function
        - :math:`\mathbf{W}_{ax}` and :math:`\mathbf{b}_{ax}` are the weights
          and bias for the input to hidden layer
        - :math:`\mathbf{W}_{aa}` and :math:`\mathbf{b}_{aa}` are the weights
          and biases for the hidden to hidden layer
        Parameters
        ----------
        n_out : int
            The dimension of a single hidden state / output on a given timestep
        act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The activation function for computing ``A[t]``. Default is `'Tanh'`.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is `'glorot_uniform'`.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with default
            parameters. Default is None.
        """  # noqa: E501
        super().__init__(optimizer)

        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.n_timesteps = None
        self.act_fn = ActivationInitializer(act_fn)()
        self.parameters = {"Waa": None, "Wax": None, "ba": None, "bx": None}
        self.is_initialized = False

    def _init_params(self):
        self.X = []
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)

        Wax = init_weights((self.n_in, self.n_out))
        Waa = init_weights((self.n_out, self.n_out))
        ba = np.zeros((self.n_out, 1))
        bx = np.zeros((self.n_out, 1))

        self.parameters = {"Waa": Waa, "Wax": Wax, "ba": ba, "bx": bx}

        self.gradients = {
            "Waa": np.zeros_like(Waa),
            "Wax": np.zeros_like(Wax),
            "ba": np.zeros_like(ba),
            "bx": np.zeros_like(bx),
        }

        self.derived_variables = {
            "A": [],
            "Z": [],
            "n_timesteps": 0,
            "current_step": 0,
            "dLdA_accumulator": None,
        }

        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "RNNCell",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, Xt):
        """
        Compute the network output for a single timestep.
        Parameters
        ----------
        Xt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            Input at timestep `t` consisting of `n_ex` examples each of
            dimensionality `n_in`.
        Returns
        -------
        At: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
            The value of the hidden state at timestep `t` for each of the
            `n_ex` examples.
        """
        if not self.is_initialized:
            self.n_in = Xt.shape[1]
            self._init_params()

        # increment timestep
        self.derived_variables["n_timesteps"] += 1
        self.derived_variables["current_step"] += 1

        # Retrieve parameters
        ba = self.parameters["ba"]
        bx = self.parameters["bx"]
        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]

        # initialize the hidden state to zero
        As = self.derived_variables["A"]
        if len(As) == 0:
            n_ex, n_in = Xt.shape
            A0 = np.zeros((n_ex, self.n_out))
            As.append(A0)

        # compute next hidden state
        Zt = As[-1] @ Waa + ba.T + Xt @ Wax + bx.T
        At = self.act_fn(Zt)

        self.derived_variables["Z"].append(Zt)
        self.derived_variables["A"].append(At)

        # store intermediate variables
        self.X.append(Xt)
        return At

    def backward(self, dLdAt):
        """
        Backprop for a single timestep.
        Parameters
        ----------
        dLdAt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
            The gradient of the loss wrt. the layer outputs (ie., hidden
            states) at timestep `t`.
        Returns
        -------
        dLdXt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            The gradient of the loss wrt. the layer inputs at timestep `t`.
        """
        assert self.trainable, "Layer is frozen"

        #  decrement current step
        self.derived_variables["current_step"] -= 1

        # extract context variables
        Zs = self.derived_variables["Z"]
        As = self.derived_variables["A"]
        t = self.derived_variables["current_step"]
        dA_acc = self.derived_variables["dLdA_accumulator"]

        # initialize accumulator
        if dA_acc is None:
            dA_acc = np.zeros_like(As[0])

        # get network weights for gradient calcs
        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]

        # compute gradient components at timestep t
        dA = dLdAt + dA_acc
        dZ = self.act_fn.grad(Zs[t]) * dA
        dXt = dZ @ Wax.T

        # update parameter gradients with signal from current step
        self.gradients["Waa"] += As[t].T @ dZ
        self.gradients["Wax"] += self.X[t].T @ dZ
        self.gradients["ba"] += dZ.sum(axis=0, keepdims=True).T
        self.gradients["bx"] += dZ.sum(axis=0, keepdims=True).T

        # update accumulator variable for hidden state
        self.derived_variables["dLdA_accumulator"] = dZ @ Waa.T
        return dXt

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        assert self.trainable, "Layer is frozen"

        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        self.derived_variables["n_timesteps"] = 0
        self.derived_variables["current_step"] = 0

        # reset parameter gradients to 0
        for k, v in self.parameters.items():
            self.gradients[k] = np.zeros_like(v)


class LSTMCell(LayerBase):
    def __init__(
        self,
        n_out,
        act_fn="Tanh",
        gate_fn="Sigmoid",
        init="glorot_uniform",
        optimizer=None,
    ):
        """
        A single step of a long short-term memory (LSTM) RNN.
        Notes
        -----
        Notation:
        - ``Z[t]``  is the input to each of the gates at timestep `t`
        - ``A[t]``  is the value of the hidden state at timestep `t`
        - ``Cc[t]`` is the value of the *candidate* cell/memory state at timestep `t`
        - ``C[t]``  is the value of the *final* cell/memory state at timestep `t`
        - ``Gf[t]`` is the output of the forget gate at timestep `t`
        - ``Gu[t]`` is the output of the update gate at timestep `t`
        - ``Go[t]`` is the output of the output gate at timestep `t`
        Equations::
            Z[t]  = stack([A[t-1], X[t]])
            Gf[t] = gate_fn(Wf @ Z[t] + bf)
            Gu[t] = gate_fn(Wu @ Z[t] + bu)
            Go[t] = gate_fn(Wo @ Z[t] + bo)
            Cc[t] = act_fn(Wc @ Z[t] + bc)
            C[t]  = Gf[t] * C[t-1] + Gu[t] * Cc[t]
            A[t]  = Go[t] * act_fn(C[t])
        where `@` indicates dot/matrix product, and '*' indicates elementwise
        multiplication.
        Parameters
        ----------
        n_out : int
            The dimension of a single hidden state / output on a given timestep.
        act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The activation function for computing ``A[t]``. Default is
            `'Tanh'`.
        gate_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The gate function for computing the update, forget, and output
            gates. Default is `'Sigmoid'`.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is `'glorot_uniform'`.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with default
            parameters. Default is None.
        """  # noqa: E501
        super().__init__(optimizer)

        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.n_timesteps = None
        self.act_fn = ActivationInitializer(act_fn)()
        self.gate_fn = ActivationInitializer(gate_fn)()
        self.parameters = {
            "Wf": None,
            "Wu": None,
            "Wc": None,
            "Wo": None,
            "bf": None,
            "bu": None,
            "bc": None,
            "bo": None,
        }
        self.is_initialized = False

    def _init_params(self):
        self.X = []
        init_weights_gate = WeightInitializer(str(self.gate_fn), mode=self.init)
        init_weights_act = WeightInitializer(str(self.act_fn), mode=self.init)

        Wf = init_weights_gate((self.n_in + self.n_out, self.n_out))
        Wu = init_weights_gate((self.n_in + self.n_out, self.n_out))
        Wc = init_weights_act((self.n_in + self.n_out, self.n_out))
        Wo = init_weights_gate((self.n_in + self.n_out, self.n_out))

        bf = np.zeros((1, self.n_out))
        bu = np.zeros((1, self.n_out))
        bc = np.zeros((1, self.n_out))
        bo = np.zeros((1, self.n_out))

        self.parameters = {
            "Wf": Wf,
            "Wu": Wu,
            "Wc": Wc,
            "Wo": Wo,
            "bf": bf,
            "bu": bu,
            "bc": bc,
            "bo": bo,
        }

        self.gradients = {
            "Wf": np.zeros_like(Wf),
            "Wu": np.zeros_like(Wu),
            "Wc": np.zeros_like(Wc),
            "Wo": np.zeros_like(Wo),
            "bf": np.zeros_like(bf),
            "bu": np.zeros_like(bu),
            "bc": np.zeros_like(bc),
            "bo": np.zeros_like(bo),
        }

        self.derived_variables = {
            "C": [],
            "A": [],
            "Gf": [],
            "Gu": [],
            "Go": [],
            "Gc": [],
            "Cc": [],
            "n_timesteps": 0,
            "current_step": 0,
            "dLdA_accumulator": None,
            "dLdC_accumulator": None,
        }

        self.is_initialized = True

    def _get_params(self):
        Wf = self.parameters["Wf"]
        Wu = self.parameters["Wu"]
        Wc = self.parameters["Wc"]
        Wo = self.parameters["Wo"]
        bf = self.parameters["bf"]
        bu = self.parameters["bu"]
        bc = self.parameters["bc"]
        bo = self.parameters["bo"]
        return Wf, Wu, Wc, Wo, bf, bu, bc, bo

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "LSTMCell",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "gate_fn": str(self.gate_fn),
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, Xt):
        """
        Compute the layer output for a single timestep.
        Parameters
        ----------
        Xt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            Input at timestep t consisting of `n_ex` examples each of
            dimensionality `n_in`.
        Returns
        -------
        At: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
            The value of the hidden state at timestep `t` for each of the `n_ex`
            examples.
        Ct: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
            The value of the cell/memory state at timestep `t` for each of the
            `n_ex` examples.
        """
        if not self.is_initialized:
            self.n_in = Xt.shape[1]
            self._init_params()

        Wf, Wu, Wc, Wo, bf, bu, bc, bo = self._get_params()

        self.derived_variables["n_timesteps"] += 1
        self.derived_variables["current_step"] += 1

        if len(self.derived_variables["A"]) == 0:
            n_ex, n_in = Xt.shape
            init = np.zeros((n_ex, self.n_out))
            self.derived_variables["A"].append(init)
            self.derived_variables["C"].append(init)

        A_prev = self.derived_variables["A"][-1]
        C_prev = self.derived_variables["C"][-1]

        # concatenate A_prev and Xt to create Zt
        Zt = np.hstack([A_prev, Xt])

        Gft = self.gate_fn(Zt @ Wf + bf)
        Gut = self.gate_fn(Zt @ Wu + bu)
        Got = self.gate_fn(Zt @ Wo + bo)
        Cct = self.act_fn(Zt @ Wc + bc)
        Ct = Gft * C_prev + Gut * Cct
        At = Got * self.act_fn(Ct)

        # bookkeeping
        self.X.append(Xt)
        self.derived_variables["A"].append(At)
        self.derived_variables["C"].append(Ct)
        self.derived_variables["Gf"].append(Gft)
        self.derived_variables["Gu"].append(Gut)
        self.derived_variables["Go"].append(Got)
        self.derived_variables["Cc"].append(Cct)
        return At, Ct

    def backward(self, dLdAt):
        """
        Backprop for a single timestep.
        Parameters
        ----------
        dLdAt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
            The gradient of the loss wrt. the layer outputs (ie., hidden
            states) at timestep `t`.
        Returns
        -------
        dLdXt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            The gradient of the loss wrt. the layer inputs at timestep `t`.
        """
        assert self.trainable, "Layer is frozen"

        Wf, Wu, Wc, Wo, bf, bu, bc, bo = self._get_params()

        self.derived_variables["current_step"] -= 1
        t = self.derived_variables["current_step"]

        Got = self.derived_variables["Go"][t]
        Gft = self.derived_variables["Gf"][t]
        Gut = self.derived_variables["Gu"][t]
        Cct = self.derived_variables["Cc"][t]
        At = self.derived_variables["A"][t + 1]
        Ct = self.derived_variables["C"][t + 1]
        C_prev = self.derived_variables["C"][t]
        A_prev = self.derived_variables["A"][t]

        Xt = self.X[t]
        Zt = np.hstack([A_prev, Xt])

        dA_acc = self.derived_variables["dLdA_accumulator"]
        dC_acc = self.derived_variables["dLdC_accumulator"]

        # initialize accumulators
        if dA_acc is None:
            dA_acc = np.zeros_like(At)

        if dC_acc is None:
            dC_acc = np.zeros_like(Ct)

        # Gradient calculations
        # ---------------------

        dA = dLdAt + dA_acc
        dC = dC_acc + dA * Got * self.act_fn.grad(Ct)

        # compute the input to the gate functions at timestep t
        _Go = Zt @ Wo + bo
        _Gf = Zt @ Wf + bf
        _Gu = Zt @ Wu + bu
        _Gc = Zt @ Wc + bc

        # compute gradients wrt the *input* to each gate
        dGot = dA * self.act_fn(Ct) * self.gate_fn.grad(_Go)
        dCct = dC * Gut * self.act_fn.grad(_Gc)
        dGut = dC * Cct * self.gate_fn.grad(_Gu)
        dGft = dC * C_prev * self.gate_fn.grad(_Gf)

        dZ = dGft @ Wf.T + dGut @ Wu.T + dCct @ Wc.T + dGot @ Wo.T
        dXt = dZ[:, self.n_out :]

        self.gradients["Wc"] += Zt.T @ dCct
        self.gradients["Wu"] += Zt.T @ dGut
        self.gradients["Wf"] += Zt.T @ dGft
        self.gradients["Wo"] += Zt.T @ dGot
        self.gradients["bo"] += dGot.sum(axis=0, keepdims=True)
        self.gradients["bu"] += dGut.sum(axis=0, keepdims=True)
        self.gradients["bf"] += dGft.sum(axis=0, keepdims=True)
        self.gradients["bc"] += dCct.sum(axis=0, keepdims=True)

        self.derived_variables["dLdA_accumulator"] = dZ[:, : self.n_out]
        self.derived_variables["dLdC_accumulator"] = Gft * dC
        return dXt

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        assert self.trainable, "Layer is frozen"

        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        self.derived_variables["n_timesteps"] = 0
        self.derived_variables["current_step"] = 0

        # reset parameter gradients to 0
        for k, v in self.parameters.items():
            self.gradients[k] = np.zeros_like(v)


class RNN(LayerBase):
    def __init__(self, n_out, act_fn="Tanh", init="glorot_uniform", optimizer=None):
        """
        A single vanilla (Elman)-RNN layer.
        Parameters
        ----------
        n_out : int
            The dimension of a single hidden state / output on a given
            timestep.
        act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The activation function for computing ``A[t]``. Default is
            `'Tanh'`.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is `'glorot_uniform'`.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with default
            parameters. Default is None.
        """  # noqa: E501
        super().__init__(optimizer)

        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.n_timesteps = None
        self.act_fn = ActivationInitializer(act_fn)()
        self.is_initialized = False

    def _init_params(self):
        self.cell = RNNCell(
            n_in=self.n_in,
            n_out=self.n_out,
            act_fn=self.act_fn,
            init=self.init,
            optimizer=self.optimizer,
        )
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "RNN",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": self.cell.hyperparameters["optimizer"],
        }

    def forward(self, X):
        """
        Run a forward pass across all timesteps in the input.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_t)`
            Input consisting of `n_ex` examples each of dimensionality `n_in`
            and extending for `n_t` timesteps.
        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps.
        """
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = []
        n_ex, n_in, n_t = X.shape
        for t in range(n_t):
            yt = self.cell.forward(X[:, :, t])
            Y.append(yt)
        return np.dstack(Y)

    def backward(self, dLdA):
        """
        Run a backward pass across all timesteps in the input.
        Parameters
        ----------
        dLdA : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
            The gradient of the loss with respect to the layer output for each
            of the `n_ex` examples across all `n_t` timesteps.
        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_t)`
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps.
        """
        assert self.cell.trainable, "Layer is frozen"
        dLdX = []
        n_ex, n_out, n_t = dLdA.shape
        for t in reversed(range(n_t)):
            dLdXt = self.cell.backward(dLdA[:, :, t])
            dLdX.insert(0, dLdXt)
        dLdX = np.dstack(dLdX)
        return dLdX

    @property
    def derived_variables(self):
        """
        Return a dictionary containing any intermediate variables computed
        during the forward / backward passes.
        """
        return self.cell.derived_variables

    @property
    def gradients(self):
        """
        Return a dictionary of the gradients computed during the backward
        pass
        """
        return self.cell.gradients

    @property
    def parameters(self):
        """Return a dictionary of the current layer parameters"""
        return self.cell.parameters

    def set_params(self, summary_dict):
        """
        Set the layer parameters from a dictionary of values.
        Parameters
        ----------
        summary_dict : dict
            A dictionary of layer parameters and hyperparameters. If a required
            parameter or hyperparameter is not included within `summary_dict`,
            this method will use the value in the current layer's
            :meth:`summary` method.
        Returns
        -------
        layer : :doc:`Layer <numpy_ml.neural_nets.layers>` object
            The newly-initialized layer.
        """
        self = super().set_params(summary_dict)
        return self.cell.set_parameters(summary_dict)

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        self.cell.freeze()

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        self.cell.unfreeze()

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        self.cell.flush_gradients()

    def update(self):
        """
        Update the layer parameters using the accrued gradients and layer
        optimizer. Flush all gradients once the update is complete.
        """
        self.cell.update()
        self.flush_gradients()


class LSTM(LayerBase):
    def __init__(
        self,
        n_out,
        act_fn="Tanh",
        gate_fn="Sigmoid",
        init="glorot_uniform",
        optimizer=None,
    ):
        """
        A single long short-term memory (LSTM) RNN layer.
        Parameters
        ----------
        n_out : int
            The dimension of a single hidden state / output on a given timestep.
        act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The activation function for computing ``A[t]``. Default is `'Tanh'`.
        gate_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The gate function for computing the update, forget, and output
            gates. Default is `'Sigmoid'`.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is `'glorot_uniform'`.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.
        """  # noqa: E501
        super().__init__(optimizer)

        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.n_timesteps = None
        self.act_fn = ActivationInitializer(act_fn)()
        self.gate_fn = ActivationInitializer(gate_fn)()
        self.is_initialized = False

    def _init_params(self):
        self.cell = LSTMCell(
            n_in=self.n_in,
            n_out=self.n_out,
            act_fn=self.act_fn,
            gate_fn=self.gate_fn,
            init=self.init,
        )
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "LSTM",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "gate_fn": str(self.gate_fn),
            "optimizer": self.cell.hyperparameters["optimizer"],
        }

    def forward(self, X):
        """
        Run a forward pass across all timesteps in the input.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_t)`
            Input consisting of `n_ex` examples each of dimensionality `n_in`
            and extending for `n_t` timesteps.
        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps.
        """
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = []
        n_ex, n_in, n_t = X.shape
        for t in range(n_t):
            yt, _ = self.cell.forward(X[:, :, t])
            Y.append(yt)
        return np.dstack(Y)

    def backward(self, dLdA):
        """
        Run a backward pass across all timesteps in the input.
        Parameters
        ----------
        dLdA : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
            The gradient of the loss with respect to the layer output for each
            of the `n_ex` examples across all `n_t` timesteps.
        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape (`n_ex`, `n_in`, `n_t`)
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps.
        """  # noqa: E501
        assert self.cell.trainable, "Layer is frozen"
        dLdX = []
        n_ex, n_out, n_t = dLdA.shape
        for t in reversed(range(n_t)):
            dLdXt, _ = self.cell.backward(dLdA[:, :, t])
            dLdX.insert(0, dLdXt)
        dLdX = np.dstack(dLdX)
        return dLdX

    @property
    def derived_variables(self):
        """
        Return a dictionary containing any intermediate variables computed
        during the forward / backward passes.
        """
        return self.cell.derived_variables

    @property
    def gradients(self):
        """
        Return a dictionary of the gradients computed during the backward
        pass
        """
        return self.cell.gradients

    @property
    def parameters(self):
        """Return a dictionary of the current layer parameters"""
        return self.cell.parameters

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        self.cell.freeze()

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        self.cell.unfreeze()

    def set_params(self, summary_dict):
        """
        Set the layer parameters from a dictionary of values.
        Parameters
        ----------
        summary_dict : dict
            A dictionary of layer parameters and hyperparameters. If a required
            parameter or hyperparameter is not included within `summary_dict`,
            this method will use the value in the current layer's
            :meth:`summary` method.
        Returns
        -------
        layer : :doc:`Layer <numpy_ml.neural_nets.layers>` object
            The newly-initialized layer.
        """
        self = super().set_params(summary_dict)
        return self.cell.set_parameters(summary_dict)

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        self.cell.flush_gradients()

    def update(self):
        """
        Update the layer parameters using the accrued gradients and layer
        optimizer. Flush all gradients once the update is complete.
        """
        self.cell.update()
        self.flush_gradients()