from abc import ABC, abstractmethod
import copy
import numpy as np

class ActivationBase(ABC):
	"""docstring for Activation"""
	def __init__(self):
		super().__init__()

	def __call__(self, z):
		if z.ndim == 1:
			z = z.reshape(1, -1)
		return self.fn(z)

	@abstractmethod
	def forward(self, z):
		raise NotImplementedError

	@abstractmethod
	def grad(self, x, **kwargs):
		raise NotImplementedError


class Sigmoid(ActivationBase):
	"""docstring for Sigmoid"""
	def __init__(self,):
		super().__init__()

	def __str__(self):
		return "Sigmoid"

	def fn(self, z):
		"""
			f(x) = 1 / (1 + e^{-x})
		"""
		self.last_forward = 1.0 / (1.0 + np.exp(-z))
		return self.last_forward

	def grad(self, x):
		"""
			f(x) = Sigmoid(x)
			f'(x) = f(x)(1-f(x)) = e^{-x} / (1 + e^{-x})^2
		"""
		self.last_forward = self.fn(x) if x else self.last_forward
		return self.last_forward * (1 - self.last_forward)

	def grad2(self, x):
		r"""
		Evaluate the second derivative of the logistic sigmoid on the elements of `x`.
			 math::
				\frac{\partial^2 \sigma}{\partial x_i^2} =
					\frac{\partial \sigma}{\partial x_i} (1 - 2 \sigma(x_i))
		"""
		self.last_forward = self.fn(x) if x else self.last_forward
		return self.last_forward * (1 - self.last_forward) * (1 - 2 * self.last_forward)

class Tanh(object):
	"""docstring for TanH"""
	def __init__(self):
		super().__init__()

	def __str__(self):
		return "Tanh"

	def fn(self, z):
		"""
			f(x) = (e^{x} - e^{-x})/(e^{x} + e^{-x})
		"""
		self.last_forward = np.tanh(z)
		return self.last_forward

	def grad(self, x):
		"""
			f'(x) = 1 - (tanh(x))^2
		"""
		last_forward = self.fn(x) if x else self.last_forward
		return 1 - last_forward ** 2

	def grad2(self, x):
		"""
			f''(x) = -2 tanh(x) f'(x)
		"""
		last_forward = self.fn(x) if x else self.last_forward
		return -2 * last_forward * (1 - last_forward ** 2)

class Affine(ActivationBase):
	def __init__(self, slope=1, intercept=0):
		self.slope = slope
		self.intercept = intercept
		super().__init__()

	def __str__(self):
		return "Affine"

	def fn(self, z):
		"""
			f(z) = slope * z + intercept
		"""
		return self.slope * z + self.intercept

	def grad(self, x):
		return np.ones_like(x) * self.slope

	def grad2(self, x):
		return np.zero_like(x)

class ReLU(object):
	"""docstring for ReLU"""
	def __init__(self,):
		super().__init__()

	def __str__(self):
		return "ReLU"

	def fn(self, z):
		"""
			f(x) = max(0, x)
		"""
		self.last_forward = np.maximum(0, z)
		return self.last_forward

	def grad(self, x):
		"""
			f'(x) = 0 if x<=0
					1 if x>0
		"""
		return (x>0).astype(int)

	def grad2(self, x):
		return np.zero_like(x)

class LeakyReLU(ActivationBase):
	"""
	References
	----------
	.. [*] Mass, L. M., Hannun, A. Y, & Ng, A. Y. (2013). "Rectifier
		nonlinearities improve neural network acoustic models". *Proceedings of
		the 30th International Conference of Machine Learning, 30*.
	"""
	def __init__(self, alpha=0.3):
		self.alpha = alpha
		super().__init__()

	def __str__(self):
		return "LeakyReLU"

	def fn(self, z):
		"""
			LeakyReLU(z)
				&= z 		if z > 0
				&= alpha*z 	otherwise
		"""
		_z = z.copy()
		_z[z<0] = _z[z<0] * self.alpha
		return _z

	def grad(self, x):
		out = np.ones_like(x)
		out[x<0] *= self.alpha
		return out

	def grad2(self, x):
		return np.zero_like(x)

class Softmax(ActivationBase):
	"""docstring for Softmax"""
	def __init__(self, dim=1):
		super(Softmax, self).__init__()
		self.dim = dim

	def __str__(self):
		return "Softmax"

	def fn(self, z):
		"""
			f(x[j]) = e^{x[j]} / (\\sum_{k=1}^{K})(e^{x[k]})
			x \\in ()
		"""
		x_exp = np.exp(z)
		x_sum = np.sum(x_exp, axis=self.dim, keepdims=True)
		self.last_forward = x_exp / x_sum
		return self.last_forward

	def grad(self, x):
		"""
			f'(x) = diag(softmax(x)) - softmax(x).T * softmax(x)
		"""
		last_forward = self.fn(x) if x else self.last_forward
		res = np.diag(last_forward) - np.multiply(last_forward.T, last_forward)
		return res

class SoftPlus(ActivationBase):
	"""docstring for SoftPlus"""
	def __init__(self):
		super().__init__()

	def __str__(self):
		return "SoftPlus"

	def fn(self, z):
		"""
			f(x) = log(1+e^{x})
		"""
		self.last_forward = np.exp(z)
		return np.log(1 + self.last_forward)

	def grad(self, x):
		"""
			f'(x) = e^{x} / (1 + e^{x})
		"""
		last_forward = np.exp(x) if x else self.last_forward
		return last_forward / (1 + last_forward)

	def grad2(self, x):
		last_forward = np.exp(x) if x else self.last_forward
		return last_forward / ((last_forward + 1) ** 2)

# class SoftSign(ActivationBase):
# 	"""docstring for SoftSign"""
# 	def __init__(self):
# 		super().__init__()
# 
# 	def __str__(self):
# 		return "SoftSign"
# 
# 	def fn(self, z):
# 		"""
# 			f(x) = x/(1+abs(x))
# 		"""
# 		self.last_forward = np.abs(z)
# 		return z / (1. + self.last_forward)
# 
# 	def grad(self, x):
# 		"""
# 			f'(x) = 1/(1+abs(x))^2
# 		"""
# 		last_forward = (1 + np.abs(x)) if x else self.last_forward
# 		return 1. / np.power(last_forward, 2)


