from turtledemo.sorting_animate import partition

from sympy import false
from torchvision import transforms
import torchvision
import torch
import os
from torch.utils import data

def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(dim=1,keepdim=True)
    return X_exp/partition
def net(X,w,b):
    return softmax(torch.matmul(X.reshape(-1,w.shape[0]),w)+b)
def cross_entropy(y_hat,y):
    y_data_calc=y_hat[range(len(y_hat)),y]
    return -torch.log(y_data_calc)
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_argmax=y_hat.argmax(axis=1).type(y.dtype)
    else:
        y_argmax=y_hat
    cmp= y_argmax == y
    return cmp.sum()


batch_size=256


trans= transforms.ToTensor()

# os.mkdir(os.path.join('.','imgdata'))

mnist_train=torchvision.datasets.FashionMNIST(
    root="./img_train",
    train=True,
    transform=trans,
    download=True
)
mnist_test=torchvision.datasets.FashionMNIST(
    root="./img_test",
    train=false,
    transform=trans,
    download=True
)

num_input=784
num_output=10
w=torch.normal(0,0.01,(784,10),requires_grad=True)
b=torch.zeros((1,10),requires_grad=True)

X=torch.normal(0,1,(2,5))
X_prob=softmax(X)
print(X_prob,X_prob.sum(1))

