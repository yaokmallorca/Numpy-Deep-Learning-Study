import numpy as np

class LossBase(object):
	def __init__(self):
		super().__init__()

	def __str__(self):
		return self.__class__.__name__

	def loss(self, y_true, y_pred):
		raise NotImplementError()

	def grad(self, y_true, y_pred):
		raise NotImplementError()

class MeanSquareError(LossBase):
	""" 
	y_true: ground truth
	y_pred: pred
	act_fn: activation object
	z: the squared error loss with repect to z
	"""
	def __init__(self, mean=True):
		super().__init__()
		self.mean = mean

	def __call__(self, y_true, y_pred):
		return self.loss(y_true, y_pred)

	def loss(self, y_true, y_pred):
		l2_norm = np.linalg.norm(y_pred - y_true) ** 2
		return l2_norm / y_true.size

	def grad(self, y_true, y_pred):
		return (y_pred - y_true)

class CrossEntropy(LossBase):
	def __init__(self, dim=1):
		super().__init__()
		self._dim = dim
		self.eps = 1e-12

	# gt: (batch_size, 1, gt_shape)
	def _one_hot(self, gt, n_cls, dim=1):
		if len(gt.shape) == 3:
			N, h, w = gt.shape
			one_hot = np.zeros((N, n_cls, h, w))
			for n in range(N):
				for c in range(n_cls):
					one_hot[n][c][np.where(gt[n][c] == c)] = 1
		else:
			N, s = gt.shape
			one_hot = np.zeros((N, n_cls, s))
			for n in range(N):
				for c in range(n_cls):
					one_hot[n][c][np.where(gt[n] == c)] = 1
		return one_hot


	def __call__(self, y_true, y_pred, n_cls):
		return self.loss(y_true, y_pred, n_cls)

	"""
	def loss(self, y_true, y_pred, n_cls):
		N = y_true.shape[0]
		y_pred_exp = np.exp(y_pred - np.max(y_pred, axis=-1, keepdims=True))
		y_pred_sm = y_pred_exp/np.sum(y_pred_exp, axis=-1, keepdims=True)
		y_true_one_hot = self._one_hot(y_true, n_cls)
		eps = 1e-12
		predictions = np.clip(y_pred_sm, eps, 1.-eps)
		# loss_val = -np.sum(y_true_one_hot * np.log(predictions)) / y_true.shape[0]
		ce = - np.sum(y_true_one_hot*np.log(predictions)) / N
		# loss_val = -np.sum(np.matmul(y_true_one_hot, np.log(y_pred+eps)))
		return ce
	"""

	def _softmax(self, y_pred, eps=1e-12, dim=1):
		y_pred_exp = np.exp(y_pred - np.max(y_pred, axis=dim, keepdims=True))
		y_pred_exp = y_pred_exp / (np.sum(y_pred_exp, axis=dim, keepdims=True))
		return y_pred_exp

	def loss(self, y_true, y_pred, n_cls):
		y_pred_softmax = self._softmax(y_pred, dim=self._dim)
		print(y_pred_softmax)
		y_true_one_hot = self._one_hot(y_true, n_cls)
		predictions = np.clip(y_pred_softmax, self.eps, 1.-self.eps)
		N = y_true.shape[0]
		ce = - np.sum(y_true_one_hot*np.log(predictions)) / y_true.size
		return ce

	def grad(self, y_true, y_pred):
		grad = y_pred - y_true
		return grad

class BinaryCrossEntropy(LossBase):
	def __init__(self, epsilon=1e-11):
		self.epsilon = epsilon

	def __call__(self, y_true, y_pred):
		return self.loss(y_true, y_pred)

	def loss(self, y_true, y_pred):
		y_pred = np.clip(y_pred, self.epsilon, 1-self.epsilon)
		loss = np.mean(-np.sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred), axis=1))
		return loss

	def grad(self, y_true, y_pred):
		y_pred = np.clip(y_pred, self.epsilon, 1-self.epsilon)
		divisor = np.maxmum(y_pred * (1 - y_pred), self.epsilon)
		return (y_true - y_pred) / divisor

class VAELoss(LossBase):
	"""
	The variational lower bound for a variational autoencoder with Bernouli units.
	L_{VAELoss} = CrossEntropy(y_true, y_pred) + KL[p||q]
	the prior p assumed to be a unit Gaussian
	the learned variational distribution q 
	KL : KL divergence 
	D_{kl} = sum(p*log(p/q))

	References
	----------
	.. [1] Kingma, D. P. & Welling, M. (2014). "Auto-encoding variational Bayes".
	   *arXiv preprint arXiv:1312.6114.* https://arxiv.org/pdf/1312.6114.pdf
	"""
	def __init__(self):
		super().__init__()

	def __call__(self, y_true, y_pred, t_mean, t_log_var):
		return self.loss(y_true, y_pred, t_mean, t_log_var)

	"""
	variational lower bound for a Bernouli VAE
	parameters:
	y_true: the original image (n_ex, N)
	y_pred: the VAE reconstruction of image (n_ex, N)
	t_mean: Mean of the variational distribution (n_ex, T)
	t_log_var: log of the variance vector of the variational distribution (n_ex, T)
	"""
	def loss(self, y_true, y_pred, t_mean, t_log_var):
		eps = np.finfo(float).eps
		y_pred = np.clip(y_pred, eps, 1-eps)
		# reconstruction loss: binary cross entropy
		rec_loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis = 1)

		# KL divergence
		# a unit Gaussian
		kl_loss = -0.5 * np.sum(1 + t_log_var - t_mean ** 2 - np.exp(t_log_var), axis=1)
		loss = np.mean(kl_loss + rec_loss)
		return loss

	def grad(self, y_true, y_pred, t_mean, t_log_var):
		N = y_true.shape[0]
		eps = np.finof(float).eps
		y_pred = np.clip(y_pred, eps, 1-eps)

		dY_pred = -y / (N * y_pred) - (y - 1) / (N - N * y_pred)
		dLogVar = (np.exp(t_log_var) - 1) / (2 * N)
		dMean = t_mean / N
		return dY_pred, dLogVar, dMean





