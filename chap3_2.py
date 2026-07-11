import random
import torch

def synthetic_data(w,b,num_examples):
    X=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]
def linreg(X,w,b):
    return torch.matmul(X,w)+b
def squared_loss(y,y_hat):
    y=y.reshape(y_hat.shape)
    diff=y-y_hat
    return diff**2/2
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()



true_w=torch.tensor([2,-3.4])
true_b=torch.tensor([4.2])
features,labels=synthetic_data(true_w,true_b,1000)
w=torch.zeros(2,requires_grad=True)
b=torch.zeros((1,),requires_grad=True)
batch_size=300
num_epochs=10
lr=1.0

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat=linreg(X,w,b)
        L=squared_loss(y,y_hat)
        L.sum().backward()
        sgd([w,b],lr,batch_size)
        print(f'epoch{epoch+1}: ,loss {float(L.detach().mean()):f}')
print(f'{w},{b}')





