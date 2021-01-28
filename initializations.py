import copy
import numpy as np

from utils.utils import *
from optimizer import OptimizerBase, SGD, AdaGrad, RMSProp, Adam
from activation import ActivationBase, Affine, ReLU, Tanh, Sigmoid, LeakyReLU

import random.random as _rng
from scheduler import (
		SchedulerBase,
		ConstantScheduler,
		ExponentialScheduler,
		NoamScheduler,
	)


_dtype = 'float32'

def _cast_dtype(res):
	return np.array(res, dtype=_dtype)

_zero = Zero()
_one = One()

class ActivationInitializer(object):
	def __init__(self, param=None):
		"""
		A class for initializing activation functions. Valid inputs are:
			(a) __str__ representations of `ActivationBase` instances
			(b) `ActivationBase` instances
		If `param` is `None`, return the identity function: f(X) = X
		"""
		self.param = param

	def __call__(self):
		param = self.param
		if param is None:
			act = Affine(slope=1, intercept=0)
		elif isinstance(param, ActivationBase):
			act = param
		elif isinstance(param, str):
			act = self.init_from_str(param)
		else:
			raise ValueError("Unknown activation: {}".format(param))
		return act

	def init_from_str(self, act_str):
		act_str = act_str.lower()
		if act_str == "relu":
			act_fn = ReLU()
		elif act_str == "tanh":
			act_fn = Tanh()
		elif act_str == "sigmoid":
			act_fn = Sigmoid()
		elif "affine" in act_str:
			r = r"affine\(slope=(.*), intercept=(.*)\)"
			slope, intercept = re.match(r, act_str).groups()
			act_fn = Affine(float(slope), float(intercept))
		elif "leaky relu" in act_str:
			r = r"leaky relu\(alpha=(.*)\)"
			alpha = re.match(r, act_str).groups()[0]
			act_fn = LeakyReLU(float(alpha))
		else:
			raise ValueError("Unknown activation: {}".format(act_str))
		return act_fn


class SchedulerInitializer(object):
	def __init__(self, param=None, lr=None):
		"""
		A class for initializing learning rate schedulers. Valid inputs are:
			(a) __str__ representations of `SchedulerBase` instances
			(b) `SchedulerBase` instances
			(c) Parameter dicts (e.g., as produced via the `summary` method in
				`LayerBase` instances)
		If `param` is `None`, return the ConstantScheduler with learning rate
		equal to `lr`.
		"""
		if all([lr is None, param is None]):
			raise ValueError("lr and param cannot both be `None`")

		self.lr = lr
		self.param = param

	def __call__(self):
		param = self.param
		if param is None:
			scheduler = ConstantScheduler(self.lr)
		elif isinstance(param, SchedulerBase):
			scheduler = param
		elif isinstance(param, str):
			scheduler = self.init_from_str()
		elif isinstance(param, dict):
			scheduler = self.init_from_dict()
		return scheduler

	def init_from_str(self):
		r = r"([a-zA-Z]*)=([^,)]*)"
		sch_str = self.param.lower()
		kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, sch_str)])

		if "constant" in sch_str:
			scheduler = ConstantScheduler(**kwargs)
		elif "exponential" in sch_str:
			scheduler = ExponentialScheduler(**kwargs)
		elif "noam" in sch_str:
			scheduler = NoamScheduler(**kwargs)
		elif "king" in sch_str:
			scheduler = KingScheduler(**kwargs)
		else:
			raise NotImplementedError("{}".format(sch_str))
		return scheduler

	def init_from_dict(self):
		S = self.param
		sc = S["hyperparameters"] if "hyperparameters" in S else None

		if sc is None:
			raise ValueError("Must have `hyperparameters` key: {}".format(S))

		if sc and sc["id"] == "ConstantScheduler":
			scheduler = ConstantScheduler().set_params(sc)
		elif sc and sc["id"] == "ExponentialScheduler":
			scheduler = ExponentialScheduler().set_params(sc)
		elif sc and sc["id"] == "NoamScheduler":
			scheduler = NoamScheduler().set_params(sc)
		elif sc:
			raise NotImplementedError("{}".format(sc["id"]))
		return scheduler


class OptimizerInitializer(object):
	def __init__(self, param=None):
		"""
		A class for initializing optimizers. Valid inputs are:
			(a) __str__ representations of `OptimizerBase` instances
			(b) `OptimizerBase` instances
			(c) Parameter dicts (e.g., as produced via the `summary` method in
				`LayerBase` instances)
		If `param` is `None`, return the SGD optimizer with default parameters.
		"""
		self.param = param

	def __call__(self):
		param = self.param
		if param is None:
			opt = SGD()
		elif isinstance(param, OptimizerBase):
			opt = param
		elif isinstance(param, str):
			opt = self.init_from_str()
		elif isinstance(param, dict):
			opt = self.init_from_dict()
		return opt

	def init_from_str(self):
		r = r"([a-zA-Z]*)=([^,)]*)"
		opt_str = self.param.lower()
		kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, opt_str)])
		if "sgd" in opt_str:
			optimizer = SGD(**kwargs)
		elif "adagrad" in opt_str:
			optimizer = AdaGrad(**kwargs)
		elif "rmsprop" in opt_str:
			optimizer = RMSProp(**kwargs)
		elif "adam" in opt_str:
			optimizer = Adam(**kwargs)
		else:
			raise NotImplementedError("{}".format(opt_str))
		return optimizer

	def init_from_dict(self):
		O = self.param
		cc = O["cache"] if "cache" in O else None
		op = O["hyperparameters"] if "hyperparameters" in O else None

		if op is None:
			raise ValueError("Must have `hyperparemeters` key: {}".format(O))

		if op and op["id"] == "SGD":
			optimizer = SGD().set_params(op, cc)
		elif op and op["id"] == "RMSProp":
			optimizer = RMSProp().set_params(op, cc)
		elif op and op["id"] == "AdaGrad":
			optimizer = AdaGrad().set_params(op, cc)
		elif op and op["id"] == "Adam":
			optimizer = Adam().set_params(op, cc)
		elif op:
			raise NotImplementedError("{}".format(op["id"]))
		return optimizer


class WeightInitializer(object):
	def __init__(self, act_fn_str, mode="glorot_uniform"):
		"""
		A factory for weight initializers.
		Parameters
		----------
		act_fn_str : str
		    The string representation for the layer activation function
		mode : str (default: 'glorot_uniform')
			The weight initialization strategy. Valid entries are {"he_normal",
			"he_uniform", "glorot_normal", glorot_uniform", "std_normal",
			"trunc_normal"}
		"""
		if mode not in [
			"he_normal",
			"he_uniform",
			"glorot_normal",
			"glorot_uniform",
			"std_normal",
			"trunc_normal",
		]:
			raise ValueError("Unrecognize initialization mode: {}".format(mode))

		self.mode = mode
		self.act_fn = act_fn_str

		if mode == "glorot_uniform":
			self._fn = glorot_uniform
		elif mode == "glorot_normal":
			self._fn = glorot_normal
		elif mode == "he_uniform":
			self._fn = he_uniform
		elif mode == "he_normal":
			self._fn = he_normal
		elif mode == "std_normal":
			self._fn = np.random.randn
		elif mode == "trunc_normal":
			self._fn = partial(truncated_normal, mean=0, std=1)

	def __call__(self, weight_shape):
		if "glorot" in self.mode:
			gain = self._calc_glorot_gain()
			W = self._fn(weight_shape, gain)
		elif self.mode == "std_normal":
			W = self._fn(*weight_shape)
		else:
			W = self._fn(weight_shape)
		return W

	def _calc_glorot_gain(self):
		"""
		Values from:
		https://pytorch.org/docs/stable/nn.html?#torch.nn.init.calculate_gain
		"""
		gain = 1.0
		act_str = self.act_fn.lower()
		if act_str == "tanh":
			gain = 5.0 / 3.0
		elif act_str == "relu":
			gain = np.sqrt(2)
		elif "leaky relu" in act_str:
			r = r"leaky relu\(alpha=(.*)\)"
			alpha = re.match(r, act_str).groups()[0]
			gain = np.sqrt(2 / 1 + float(alpha) ** 2)
		return gain


def decompose_size(size):
	if len(size) == 2:
		fan_in = size[0]
		fan_out = size[1]
	elif len(size) == 4 or len(size) == 5:
		respective_field_size = np.prod(size[2:])
		fan_in = size[1] * respective_field_size
		fan_out = size[0] * respective_field_size
	else:
		fan_in = fan_out = int(np.sqrt(np.prod(size)))
	return fan_in, fan_out

def he_uniform(weight_shape):
	"""
	Initializes network weights `W` with using the He uniform initialization
	strategy.
	Notes
	-----
	The He uniform initializations trategy initializes thew eights in `W` using
	draws from Uniform(-b, b) where
	.. math::
		b = sqrt{frac{6}{text{fan_in}}}
	Developed for deep networks with ReLU nonlinearities.
	Parameters
	----------
	weight_shape : tuple
		The dimensions of the weight matrix/volume.
	Returns
	-------
	W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
		The initialized weights.
	"""
	fan_in, fan_out = calc_fan(weight_shape)
	std = np.sqrt(6 / fan_in)
	return np.random.uniform(-b, b, size=weight_shape)

def truncated_normal(mean, std, out_shape):
	"""
	Generate draws from a truncated normal distribution via rejection sampling.
	Notes
	-----
	The rejection sampling regimen draws samples from a normal distribution
	with mean `mean` and standard deviation `std`, and resamples any values
	more than two standard deviations from `mean`.
	Parameters
	----------
	mean : float or array_like of floats
		The mean/center of the distribution
	std : float or array_like of floats
		Standard deviation (spread or "width") of the distribution.
	out_shape : int or tuple of ints
		Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
		``m * n * k`` samples are drawn.
	Returns
	-------
	samples : :py:class:`ndarray <numpy.ndarray>` of shape `out_shape`
		Samples from the truncated normal distribution parameterized by `mean`
		and `std`.
	"""
	samples = np.random.normal(loc=mean, scale=std, size=out_shape)
	reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
	while any(reject.flatten()):
		resamples = np.random.normal(loc=mean, scale=std, size=reject.sum())
		samples[reject] = resamples
		reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
	return samples

def he_normal(weight_shape):
	"""
	Initialize network weights `W` using the He normal initialization strategy.
	Notes
	-----
	The He normal initialization strategy initializes the weights in `W` using
	draws from TruncatedNormal(0, b) where the variance `b` is
	.. math::
	    b = frac{2}{text{fan_in}}
	He normal initialization was originally developed for deep networks with
	:class:`~numpy_ml.neural_nets.activations.ReLU` nonlinearities.
	Parameters
	----------
	weight_shape : tuple
	    The dimensions of the weight matrix/volume.
	Returns
	-------
	W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
	    The initialized weights.
	"""
	fan_in, fan_out = calc_fan(weight_shape)
	std = np.sqrt(2 / fan_in)
	return truncated_normal(0, std, weight_shape)

def glorot_uniform(weight_shape, gain=1.0):
	"""
	Initialize network weights `W` using the Glorot uniform initialization
	strategy.
	Notes
	-----
	The Glorot uniform initialization strategy initializes weights using draws
	from ``Uniform(-b, b)`` where:
	.. math::
		b = text{gain} sqrt{frac{6}{text{fan_in} + text{fan_out}}}
	The motivation for Glorot uniform initialization is to choose weights to
	ensure that the variance of the layer outputs are approximately equal to
	the variance of its inputs.
	This initialization strategy was primarily developed for deep networks with
	tanh and logistic sigmoid nonlinearities.
	Parameters
	----------
	weight_shape : tuple
		The dimensions of the weight matrix/volume.
	Returns
	-------
	W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
		The initialized weights.
	"""
	fan_in, fan_out = calc_fan(weight_shape)
	b = gain * np.sqrt(6 / (fan_in + fan_out))
	return np.random.uniform(-b, b, size=weight_shape)

def glorot_normal(weight_shape, gain=1.0):
	"""
	Initialize network weights `W` using the Glorot normal initialization strategy.
	Notes
	-----
	The Glorot normal initializaiton initializes weights with draws from
	TruncatedNormal(0, b) where the variance `b` is
		 math::
			b = frac{2 text{gain}^2}{text{fan_in} + text{fan_out}}
	The motivation for Glorot normal initialization is to choose weights to
	ensure that the variance of the layer outputs are approximately equal to
	the variance of its inputs.
	This initialization strategy was primarily developed for deep networks with
	:class:`~numpy_ml.neural_nets.activations.Tanh` and
	:class:`~numpy_ml.neural_nets.activations.Sigmoid` nonlinearities.
	Parameters
	----------
	weight_shape : tuple
		The dimensions of the weight matrix/volume.
	Returns
	-------
	W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
		The initialized weights.
	"""
	fan_in, fan_out = calc_fan(weight_shape)
	std = gain * np.sqrt(2 / (fan_in + fan_out))
	return truncated_normal(0, std, weight_shape)

class Initializer(object):
	def __call__(self, size):
		return self.call(size)

	def call(self, size):
		raise NotImplementError()

	def __str__(self):
		return self.__class__.__name__

class Zero(Initializer):
	def call(self, size):
		return _cast_dtype(np.zeros(size))

class One(Initializer):
	def call(self, size):
		return _cast_dtype(np.ones(size))

class Uniform(Initializer):
	def __init__(self, scale=0.5):
		self.scale = scale

	def call(self, size):
		return _cast_dtype(_rng.uniform(-self.scale, self.scale, size=size))

class Normal(Initializer):
	def __init__(self, std=0.01, mean=0.0):
		self.std = std
		self.mean = mean

	def call(self, size):
		return _cast_dtype(_rng.normal(loc=self.mean, scale=self.std, size=size))

class LecunUniform(Initializer):
	"""LeCun uniform initializer.

	It draws samples from a uniform distribution within [-limit, limit]
	where `limit` is `sqrt(3 / fan_in)` [1]_
	where `fan_in` is the number of input units in the weight matrix.
	
	References
	----------
	.. [1] LeCun 98, Efficient Backprop, http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
	"""
	def call(self, size):
		fan_in, fin_out = decompose_size(size)
		return Uniform(np.sqrt(3./ fan_in))(size)

class GlorotNormal(Initializer):
	"""Glorot uniform initializer, also called Xavier uniform initializer.

	It draws samples from a uniform distribution within [-limit, limit]
	where `limit` is `sqrt(6 / (fan_in + fan_out))` [1]_
	where `fan_in` is the number of input units in the weight matrix
	and `fan_out` is the number of output units in the weight matrix.
	
	References
	----------
	.. [1] Glorot & Bengio, AISTATS 2010. http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
	"""
	def call(self, size):
		fan_in, fan_out = decompose_size(size)
		return Uniform(np.sqrt(6./ (fan_in + fan_out)))(size)

class HeNormal(Initializer):
	"""He normal initializer.

	It draws samples from a truncated normal distribution centered on 0
	with `stddev = sqrt(2 / fan_in)` [1]_
	where `fan_in` is the number of input units in the weight matrix.
	
	References
	----------
	.. [1] He et al., http://arxiv.org/abs/1502.01852
	"""
	def call(self, size):
		fan_in, fan_out = decompose_size(size)
		return Normal(std=np.sqrt(2. / fan_in))(size)

class HeUniform(Initializer):
	"""He uniform variance scaling initializer.

	It draws samples from a uniform distribution within [-limit, limit]
	where `limit` is `sqrt(6 / fan_in)` [1]_
	where `fan_in` is the number of input units in the weight matrix.
	
	References
	----------
	.. [1] He et al., http://arxiv.org/abs/1502.01852
	"""
	def call(self, size):
		fan_in, fan_out = decompose_size(size)
		return Uniform(np.sqrt(6. / fan_in))(size)

class Orthogonal(Initializer):
	"""Intialize weights as Orthogonal matrix.

	Orthogonal matrix initialization [1]_. For n-dimensional shapes where
	n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
	corresponds to the fan-in, so this makes the initialization usable for
	both dense and convolutional layers.

	Parameters
	----------
	gain : float or 'relu'.
		Scaling factor for the weights. Set this to ``1.0`` for linear and
		sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
		to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
		leakiness ``alpha``. Other transfer functions may need different
		factors.

	References
	----------
	.. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
		   "Exact solutions to the nonlinear dynamics of learning in deep
		   linear neural networks." arXiv preprint arXiv:1312.6120 (2013).
	"""
	def __init__(self, gain=1.0):
		if gain == 'relu':
			gain = np.sqrt(2)
		self.gain = gain

	def call(self, size):
		flat_shape = (size[0], np.prod(size[1:]))
		a = _rng.normal(loc=0., scale=1., size=flat_shape)
		u, _, v = np.linalg.svd(a, full_matrices=False)
		q = u if u.shape == flat_shape else v
		q = q.reshape(size)
		q = self.gain * q
		return _cast_dtype(q)


def get(initialization):
	if initialization.__class__.__name__ == 'str':
		if initialization in ['zero', 'Zero']:
			return Zero()
		if initialization in ['one', 'One']:
			return One()
		if initialization in ['uniform', 'Uniform']:
			return Uniform()
		if initialization in ['normal', 'Normal']:
			return Normal()
		if initialization in ['lecun_uniform', 'LecunUniform']:
			return LecunUniform()
		if initialization in ['glorot_uniform', 'GlorotUniform']:
			return GlorotUniform()
		if initialization in ['glorot_normal', 'GlorotNormal']:
			return GlorotNormal()
		if initialization in ['HeNormal', 'he_normal']:
			return HeNormal()
		if initialization in ['HeUniform', 'he_uniform']:
			return HeUniform()
		if initialization in ['Orthogonal', 'orthogonal']:
			return Orthogonal()
		raise ValueError('Unknown initialization name: {}.'.format(initialization))

	elif isinstance(initialization, Initializer):
		return copy.deepcopy(initialization)

	else:
		raise ValueError("Unknown type: {}.".format(initialization.__class__.__name__))