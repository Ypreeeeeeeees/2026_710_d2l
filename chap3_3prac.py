import torch
from torch.utils import data
from torch import optim
from torch import nn

def synthetic_data(w,b,data_size):
    X=torch.normal(0,1,(data_size,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape(-1,1)
def load_data(data_array,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

true_w=torch.tensor([2.0,-3.4])
true_b=torch.tensor([1.0])
X,y=synthetic_data(true_w,true_b,1000)
net=nn.Sequential(nn.Linear(2,1))
loss=nn.MSELoss(reduction='none')
trainer=optim.SGD(net.parameters(),lr=0.01)
data_iter=load_data((X,y),100)
epoch_num=10

for epoch in range(epoch_num):
    for batch_X,batch_y in data_iter:
        l=loss(net(batch_X),batch_y)/2
        l.sum().backward()
        trainer.step()
        trainer.zero_grad()
    with torch.no_grad():
        l=loss(net(X),y)/2
        print(f'epoch {epoch+1},loss {float(l.sum().mean()):6f}')
w=net[0].weight.data
print(w)
b=net[0].bias.data
print(b)
