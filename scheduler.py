from copy import deepcopy
from abc import ABC, abstractmethod
import numpy as np
from math import erf

def gaussian_cdf(x, mean, var):
	eps = np.finfo(float).eps
	x_scaled = (x - mean) / np.sqrt(var + eps)
	return (1 + erf(x_scaled / np.sqrt(2))) / 2

class SchedulerBase(ABC):
	"""docstring for lr_SchedulerBase"""
	def __init__(self, lr=0.01):
		super(lr_SchedulerBase, self).__init__()
		self.lr = lr
		self.hyperparameters = {}

	def copy(self):
		return deepcopy(self)

	def learning_rate(self, step=None):
		raise NotImplementError

	def set_params(self, hyparam_dict):
		if hyparam_dict is not None:
			for k, v in hyparam_dict.dict.items:
				if k in self.hyperparameters:
					self.hyperparameters[k] = v

class ConstantScheduler(lr_SchedulerBase):
	def __init__(self, lr=0.01, **kwargs):
		super().__init__()
		self.lr = lr
		self.hyperparameters = {"id": "ConstantScheduler", "lr": self.lr}

	def __str__(self):
		return "ConstantScheduler(lr={})".format(self.lr)

	def learning_rate(self, step, **kwargs):
		return self.lr

class ExponentialScheduler(lr_SchedulerBase):
	def __init__(self, lr=0.01, stage_length=500, staircase=False, decay=0.1, **kwargs):
		"""
		An exponential scheduler decays the learning rate by 'decay' every 'stage_length' steps,
		starting from 'initial_lr'

		learning_rate = initial_lr * decay ** curr_stage
		where:
			curr_stage = step / stage_length			if staircase = False
			curr_stage = floor(step / stage_length)		if staircase = True
		"""
		super().__init__()
		self.decay = decay
		self.staircase = staircase 
		self.lr = lr
		self.stage_length = stage_length 
		self.hyperparameters = {
			'id': 'StepScheduler',
			'decay': self.decay,
			'staircase': self.staircase,
			'init_lr': self.lr,
			'stage_length': self.stage_length,
		}

	def __str__(self):
		return "ExponentialScheduler(initial_lr={}, stage_length={}, staircase={}, decay={})".format(
			self.initial_lr, self.stage_length, self.staircase, self.decay
		)

	def learning_rate(self, step, **kwargs):
		curr_stage = step / self.stage_length
		if self.staircase:
			curr_stage = np.floor(curr_stage)
		return self.lr * self.decay ** curr_stage

class NoamScheduler(lr_SchedulerBase):
	def __init__(self, model_dim=512, scale_factor=1, warmup_steps=4000, **kwargs):
		"""
		The Noam learning rate scheduler, originally used in conjunction with 
		the Adam optimizer in [1]
		Notes:
		The Noam scheduler increases the learning rate linearly for hte first "warmup_steps" 
		steps. and decreases it thereafter proportionally to the iverse square root of the 
		step number
			lr = scale_factor * ( (model_dim ** (-0.5)) * adj_step )
			adj_step = min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
		References
		----------
			[1] Vaswani et al. (2017) "Attention is all you need". *31st
			Conference on Neural Information Processing Systems*,
			https://arxiv.org/pdf/1706.03762.pdf
		"""
		super().__init__()
		self.model_dim = model_dim 
		self.scale_factor = scale_factor
		self.warmup_steps = warmup_steps
		self.hyperparameters = {
			'id': "Noamscheduler",
			'model_dim': self.model_dim,
			'scale_factor': self.scale_factor,
			'warmup_steps': self.warmup_steps,
		}

	def __str__(self):
		return "NoamScheduler(model_dim={}, scale_factor={}, warmup_steps={})".format(
			self.model_dim, self.scale_factor, self.warmup_steps
		)

	def learning_rate(self, step, **kwargs):
		warmup, d_model = self.warmup_steps, self.model_dim
		new_lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
		return self.scale_factor * new_lr
