import activation as act
import layers.loss as loss

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


"""
a = np.random.uniform(-1, size=(1,3,5,5))
a_tensor = torch.from_numpy(a)
torch_relu = F.relu(a_tensor)
print("relu result: ", torch_relu)
Relu = act.get("ReLU")
own_relu = Relu.forward(a)
print("own relu result: ", own_relu)

torch_sigmoid = F.sigmoid(a_tensor)
print("Sigmoid result: ", torch_sigmoid)
Sigmoid = act.get("Sigmoid")
own_sigmoid = Sigmoid.forward(a)
print("own Sigmoid result: ", own_sigmoid)

torch_softmax = F.softmax(a_tensor, dim=1)
print("softmax result: ", torch_softmax, torch_softmax.size())
softmax = act.get("softmax")
own_softmax = softmax.forward(a)
print("own softmax result: ", own_softmax)

torch_tanh = F.tanh(a_tensor)
print("Tanh result: ", torch_tanh)
tanh = act.get("tanh")
own_tanh = tanh.forward(a)
print("own tanh result: ", own_tanh)

torch_softsign = F.softsign(a_tensor)
print("softsign result: ", torch_softsign)
softsign = act.get("softsign")
own_softsign = softsign.forward(a)
print("own softsign result: ", own_softsign)

torch_softplus = F.softplus(a_tensor)
print("softplus result: ", torch_softplus)
softplus = act.get("softplus")
own_softplus = softplus.forward(a)
print("own softplus result: ", own_softplus)
"""

print("############## loss layer ###############")
print("MSE loss")
pred = np.random.randint(10, size=(3,3,5,5))
gt = np.random.randint(10, size=(3,3,5,5))
pred = pred.astype(np.float32)

pred_tensor = torch.from_numpy(pred)
pred_tensor.requires_grad = True
gt_tensor = torch.from_numpy(gt).type(torch.FloatTensor)

mse_loss_own = loss.MeanSquareError()
mse_loss_torch = nn.MSELoss()

own_loss_value = mse_loss_own(gt, pred)
own_loss_grad = mse_loss_own.grad(gt, pred)

print(gt_tensor.size(), pred_tensor.size())
torch_loss_value = mse_loss_torch(gt_tensor, pred_tensor)
torch_loss_value.backward()

print("own loss value: ", own_loss_value)
print("torch loss value: ", torch_loss_value)
# rint("own loss grad: ", own_loss_grad)
# rint("torch loss grad: ", torch_loss_value.grad)

print("##################################################")
print("softmax cross entropy loss")
softmax = act.get("softmax")
pred = np.random.randint(5, size=(3,5,5))
gt = np.random.randint(5, size=(3,5))
# pred = np.array([[1000000000, 2.5, 0.67],
# 				 [4.3, 0.78, -2.5],
# 				 [-2.6, 1.56, -876]])
pred = pred.astype(np.float32)
# gt = np.array([0,0,1])
n_cls = 5

pred_tensor = torch.from_numpy(pred)
pred_tensor.requires_grad = True
# pred_tensor_softmax = F.softmax(pred_tensor, dim=1)
# print(pred_tensor_softmax)
gt_tensor = torch.from_numpy(gt).type(torch.LongTensor)

torch_ce_loss = nn.CrossEntropyLoss()
# print(gt_tensor.squeeze(1))
# print(pred_tensor_softmax)
torch_ce_value = torch_ce_loss(pred_tensor, gt_tensor)

own_ce_loss = loss.CrossEntropy()
# pred_np_softmax = softmax.forward(pred)
own_ce_value = own_ce_loss(gt, pred, n_cls)
print("own loss value: ", own_ce_value)
print("torch loss value: ", torch_ce_value)


