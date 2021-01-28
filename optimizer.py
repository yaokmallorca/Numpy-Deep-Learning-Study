from copy import deepcopy
import numpy as np

from .initializations import _zero
from .scheduler import *


def npdl_clip(grad, boundary):
	if boundary > 0:
		return np.clip(grad, -boundary, boundary)
	else:
		return grad

class OptimizerBase(ABC):
	"""docstring for Optimizer"""
	def __init__(self, lr, scheduler=None):
		from initializations import SchedulerInitializer

		self.cache = {}
		self.cur_step = 0
		self.hyperparameters = {}
		self.lr_scheduler = SchedulerInitializer(scheduler, lr = lr)()

	def __call__(self, param, param_grad, param_name, cur_loss=None):
		return self.update(param, param_grad, param_name, cur_loss)

	def step(self):
		self.cur_step += 1

	def reset_step(self):
		self.cur_step = 0

	def copy(self):
		return deepcopy(self)

	def set_params(self, hparam_dict=None, cache_dict=None):
		from ..initializations import SchedulerInitializer

		if hparam_dict in not None:
			for k, v in hparam_dict.items():
				if k in self.hyperparameters[k]:
					self.hyperparameters[k] = v
				if k == 'lr_scheduler':
					self.lr_scheduler = SchedulerInitializer(v, lr=None)()

		if cache_dict is not None:
			for k, v in cache_dict.items():
				if k in self.cache:
					self.cache[k] = v

	@abstractmethod
	def update(self, param, param_grad, param_name, cur_loss=None):
		raise NotImplementError


class SGD(OptimizerBase):
	""" SGD with Momentum 
		Parameters
		----------
		lr : float
			Learning rate for SGD. If scheduler is not None, this is used as
			the starting learning rate. Default is 0.01.
		momentum : float in range [0, 1]
			The fraction of the previous update to add to the current update.
			If 0, no momentum is applied. Default is 0.
		clip_norm : float
			If not None, all param gradients are scaled to have maximum l2 norm of
			`clip_norm` before computing update. Default is None.
		lr_scheduler : str, :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object, or None
			The learning rate scheduler. If None, use a constant learning
			rate equal to `lr`. Default is None.
	"""
	def __init__(self, lr=0.01, momentum=0.0, clip_norm=None, 
		lr_scheduler=None, **kwargs):
		super().__init__(lr, lr_scheduler)

		self.hyperparameters = {
			'id': "SGD",
			'lr': lr,
			'momentum': momentum,
			'clip_norm': clip_norm,
			'lr_scheduler': str(self.lr_scheduler),
		}

	def __str__(self):
		H = self.hyperparameters
		lr, mm, cn, sc = H["lr"], H["momentum"], H["clip_norm"], H["lr_scheduler"]
		return "SGD(lr={}, momentum={}, clip_norm={}, lr_scheduler={})".format(
			lr, mm, cn, sc
		)

	def update(self, param, param_grad, param_name, cur_loss=None):
		"""
			Generates update expressions of the form:
			velocity = momentum * velocity - learning_rate * gradient
			param = param + velocity

			Parameters
			----------
			param : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
				The value of the parameter to be updated.
			param_grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
				The gradient of the loss function with respect to `param_name`.
			param_name : str
				The name of the parameter.
			cur_loss : float
				The training or validation loss for the current minibatch. Used for
				learning rate scheduling e.g., by
				:class:`~numpy_ml.neural_nets.schedulers.KingScheduler`.
				Default is None.
			Returns
			-------
			updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
				The value of `param` after applying the momentum update.
		"""
		C = self.cache
		H = self.hyperparameters
		momentum, clip_norm = H['momentum'], H['clip_norm']
		lr = self.lr_scheduler(self.cur_step, cur_loss)

		if param_name not in C:
			C[param_name] = np.zeros_like(param_grad)

		# scaled gradient to avoid explosion
		t = np.inf if clip_norm is None else clip_norm
		if norm(param_grad) > t:
			param_grad = param_grad * t / norm(param_grad)

		update = momentum * C[param_name] + lr * param_grad
		self.cache[param_name] = update
		return param - update


class AdaGrad(Optimizer):
	""" the AdaGrad algorithm
	Generates update expressions of the form:

	accumulate squared gradient: r = r + grad (prod) grad
	update: velocity = - (learning_rate) * grad / (epsilon + root(r))
	param = param + velocity

	References
	----------
	.. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
			Adaptive subgradient methods for online learning and stochastic
			optimization. JMLR, 12:2121-2159.

	.. [2] Chris Dyer:
			Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
	"""

	def __init__(self, lr=0.01, eps=1e-7, clip_norm=None, lr_scheduler=None,
		**kwargs):
		super.cache = {}
		self.hyperparameters = {
			'id': 'AdaGrad',
			'lr': lr,
			'eps': eps,
			'clip_norm': clip_norm,
			'lr_scheduler': str(self.lr_scheduler),
		}

	def update(self, param, param_grad, param_name, cur_loss=None):
		"""
			accumulate squared gradient: r = r + grad (prod) grad
			update: velocity = - (learning_rate) * grad / (epsilon + root(r))
			param = param + velocity
		"""
		C = self.cache
		H = self.hyperparameters
		eps, clip_norm = H['eps'], H['clip_norm']
		lr = self.lr_scheduler(self.cur_step, cur_loss)

		if param_name not in C:
			C[param_name] = np.zeros_like(param_grad)

		# scale gradient to avoid explosion
		t = np.inf if clip_norm is None else clip_norm
		if norm(param_grad) > t:
			param_grad = param_grad * t / norm(param_grad)

		C[param_name] += param_grad ** 2
		update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
		self.cache = C
		return param - update


class AdaDelta(Optimizer):
	""" The AdaDelta algorithm
	rho = 0.95 epsilon = 1e-6
	using the step size eta and a decay factor rho and leanring rate
	r_t = rho*r_{t-1} + (1 - rho) * g^2
	eta_t = eta * frac{sqrt(s_{t-1} + epsilon)}{sqrt(r_t + epsilon)}
	s_t = rho * s_{t-1} + (1 - rho) * (eta_t*g)^2

	fixed learning rate, so dont update lr
	References
	----------
	.. [1] Zeiler, M. D. (2012):
			ADADELTA: An Adaptive Learning Rate Method.
			arXiv Preprint arXiv:1212.5701.
	"""
	def __init__(self, rho=0.9, epsilon=1e-6, *args, **kwargs):
		super(AdaDelta, self).__init__(*args, **kwargs)
		self.rho = rho
		self.epsilon = epsilon
		self.cache = None
		self.delta = None

	def update(self, params, grads):
		if self.cache is None:
			self.cache = [_zero(p.shape) for p in params]
		if self.delta is None:
			self.delta = [_zero(p.shape) for p in params]

		for i, (c, d, p, g) in enumerate(zip(self.cache, self.delta, params, grads)):
			c = self.rho * c + (1 - self.rho) * np.power(g, 2)
			update = g * np.sqrt(d + self.epsilon) / np.sqrt(c + self.epsilon)
			p -= self.lr * update
			d = self.rho * d + (1 - self.rho) * np.power(update, 2)

			self.cache[i] = c
			self.delta[i] = d
		# super(AdaDelta, self).update(params, grads)


class RMSProp(Optimizer):
	""" the RMSProp algorithm 
	epsilon = 1e-6, decay reate rho
	accumulate squared gradient: r = rho*r + (1-rho)grad(prod)grad
	velocity = -(learning_rate) * grad / root(epsilon + r)
	param = param + velocity

	fixed learning rate, so dont update lr
	References
	----------
	.. [1] Tieleman, T. and Hinton, G. (2012):
		Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
		Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
	"""
	def __init__(self, lr=0.001, decay=0.9, eps=1e-7, clip_norm=None,
		lr_scheduler=None, **kwargs):
		super().__init__(lr, lr_scheduler)
		self.cache = {}
		self.hyperparameters = {
			'id': "RMSProp",
			'lr': lr,
			'eps': eps,
			'decay': decay,
			'clip_norm': clip_norm,
			'lr_scheduler': str(self.lr_scheduler)
		}

	def __str__(self):
		H = self.hyperparameters
		sc = H["lr_scheduler"]
		lr, eps, dc, cn = H["lr"], H["eps"], H["decay"], H["clip_norm"]
		return "RMSProp(lr={}, eps={}, decay={}, clip_norm={}, lr_scheduler={})".format(
			lr, eps, dc, cn, sc
		)

	def update(self, param, param_grad, param_name, cur_loss=None):
		"""
			epsilon = 1e-6, decay reate rho
			accumulate squared gradient: r = rho*r + (1-rho)grad(prod)grad
			velocity = -(learning_rate) * grad / root(epsilon + r)
			param = param + velocity
		"""
		C = self.cache
		H = self.hyperparameters
		eps, decay, clip_norm = H['eps'], H['decay'], H['clip_norm']
		lr = self.lr_scheduler(self.cur_step, cur_loss)

		if param_name not in C:
			C[param_name] = np.zeros_like(param_grad)

		# scale gradient to avoid explosion
		t = np.inf if clip_norm is None else clip_norm
		if norm(param_grad) > t:
			param_grad = param_grad * t / norm(param_grad)

		C[param_name] = decay * C[param_name] + (1 - decay) * param_grad ** 2
		update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
		self.cache = C
		return param - update


class Adam(Optimizer):
	""" Adam algorithm
	Parameters:
	s: float
		exponential decay rate for the first moment estimates
	v: float
		exponential decay rate for the second moment estimates
	epsilon: float
		constant for numerical stability

	rho1 = 0.9, rho2 = 0.999
	the first estimate: 	s = rho1 * s + (1 - rho1) * g
	the second estimate: 	v = rho2 * v + (1 - rho2) * g^2
	correct bias in the first estimate: 	s_hat = frac{s}{(1 - rho1^t)}
	correct bias in the second estimate: 	v_hat = frac{v}{(1 - rho2^t)}
	velocity = - learning_rate * frac{s_hat}{sqrt(v_hat) + epsilon}
	param = param + velocity

	References
	----------
	.. [1] Kingma, Diederik, and Jimmy Ba (2014):
			Adam: A Method for Stochastic Optimization.
			arXiv preprint arXiv:1412.6980.
	"""
	def __init__(self, lr=0.001, decay1=0.9, decay2=0.999, eps=1e-7,
		clip_norm=None, lr_scheduler=None, **kwargs):
		self.cache = {}
		self.hyperparameters = {
			'id': 'Adam',
			'lr': lr,
			'decay1': decay1,
			'decay2': decay2,
			'clip_norm': clip_norm,
			'lr_scheduler': str(self.lr_scheduler),
		}

	def __str__(self):
		H = self.hyperparameters
		lr, d1, d2 = H["lr"], H["decay1"], H["decay2"]
		eps, cn, sc = H["eps"], H["clip_norm"], H["lr_scheduler"]
		return "Adam(lr={}, decay1={}, decay2={}, eps={}, clip_norm={}, lr_scheduler={})".format(
			lr, d1, d2, eps, cn, sc
		)

	def update(self, param, param_grad, param_name, cur_loss=None):
		"""
			rho1 = 0.9, rho2 = 0.999
			the first estimate: 	s = rho1 * s + (1 - rho1) * g
			the second estimate: 	v = rho2 * v + (1 - rho2) * g^2
			correct bias in the first estimate: 	s_hat = frac{s}{(1 - rho1^t)}
			correct bias in the second estimate: 	v_hat = frac{v}{(1 - rho2^t)}
			velocity = - learning_rate * frac{s_hat}{sqrt(v_hat) + epsilon}
			param = param + velocity
		"""
		C = self.cache
		H = self.hyperparameters
		d1, d2 = H['decay1'], H['decay2']
		eps, clip_norm = H['eps'], H['clip_norm']
		lr = self.lr_scheduler(self.cur_step, cur_loss)

		if param not in C:
			C[param_name] = {
				't': 0,
				'mean': np.zeros_like(param_grad),
				'std': np.zeros_like(param_grad),
			}

		# scale gradient to avoid explosion
		t = np.inf if clip_norm is None else clip_norm
		if norm(param_grad) > t:
			param_grad = param_grad * t / norm(param_grad)

		t = C[param_name]['t'] + 1
		var = C[param_name]['var']
		mean = C[param_name]['mean']

		# update cache
		C[param_name]['t'] = t
		C[param_name]['var'] = d2 * var + (1 - d2) * param_grad ** 2
		C[param_name]['mean'] = d1 * mean + (1 - d1) * param_grad
		self.cache = C

		# calc unbiasd moment estimates and Adam update
		v_hat = C[param_name]['var'] / (1 - d2 ** t)
		m_hat = C[param_name]['mean'] / (1 - d1 ** t)
		update = lr * m_hat / (np.sqrt(v_hat) + eps)
		return param - update
