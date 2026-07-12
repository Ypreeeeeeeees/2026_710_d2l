import torch
from torch import nn
from torch.utils import data
from torch import optim
def synthetic_data(w,b,num_examples):
    X=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))
def load_array(data_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

true_w=torch.tensor([2.0,4.0])
true_b=torch.tensor([-3.4])
X,y=synthetic_data(true_w,true_b,1000)
data_iter=load_array((X,y),64)
net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss=nn.MSELoss(reduction='none')

trainer=optim.SGD(net.parameters(),lr=0.03)

num_epochs=10
for epoch in range(num_epochs):
    for batch_X,batch_y in data_iter:
        l=loss(net(batch_X),batch_y)/2
        trainer.zero_grad()
        l.sum().backward()
        trainer.step()
    with torch.no_grad():
        train_l = loss(net(X), y) / 2
        print(f'epoch{epoch + 1} , loss {float(train_l.mean()):f}')

w=net[0].weight.data
print(w)
b=net[0].bias.data
print(b)


