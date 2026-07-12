import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch import optim
from torch.utils import data

class Accumulator: #@save
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat_calc=torch.argmax(y_hat,dim=1)
    else:
        y_hat_calc=y_hat
    cmp=y_hat_calc.type(y.dtype) == y
    return float(cmp.sum())

def train_epoch_ch3(net, train_iter, loss, trainer):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric = Accumulator(3)

    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)

        if isinstance(trainer,torch.optim.Optimizer):
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
        else:
            l.sum().backward()
            trainer(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

def train_ch3(num_epoch,net,train_iter,test_iter,loss,trainer):
    for epoch in range(num_epoch):
        loss_mean,accuracy_rate=train_epoch_ch3(net,train_iter,loss,trainer)
        test_accuracy_rate=evaluate_accuracy(net,test_iter)
        print(f'epoch {epoch+1}, loss {loss_mean},\n accuracy_rate {accuracy_rate}, test_accuracy_rate {test_accuracy_rate}')
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric=Accumulator(2)
    for X,y in data_iter:
        y_hat=net(X)
        metric.add(accuracy(y_hat,y),y.numel())
    return metric[0]/metric[1]

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

trans=transforms.ToTensor()

mnist_train=torchvision.datasets.FashionMNIST(
    root="./img_train",
    train=True,
    transform=trans,
    download=True
)
mnist_test=torchvision.datasets.FashionMNIST(
    root="./img_test",
    train=False,
    transform=trans,
    download=True
)

input_size=784
output_size=10
net=nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size,output_size)
)
net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(),lr=0.1)
train_iter=data.DataLoader(mnist_train,batch_size=256,shuffle=True)
test_iter=data.DataLoader(mnist_test,batch_size=256,shuffle=True)
num_epochs=3
train_ch3(num_epochs,net,train_iter,test_iter,loss,trainer)
w=net[1].weight
print(w.shape)
b=net[1].bias
print(b.shape)