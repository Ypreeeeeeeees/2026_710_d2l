import random
import torch

def data_generate(true_w,true_b,data_size):
    X=torch.normal(0,1,(data_size,len(true_w)))
    y=torch.matmul(X,true_w)+true_b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]
def linreg(w,b,X):
    return torch.matmul(X,w)+b
def square_loss(y,y_hat):
    y.reshape(y_hat.shape)
    return (y-y_hat)**2/2
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

true_w=torch.tensor([7,-11.0])
true_b=torch.tensor([2026.0])
features,labels=data_generate(true_w,true_b,1000)
batch_size=200
num_epochs=10
lr=0.5
w=torch.normal(0,0.1,(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)
for i in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        y_hat=linreg(w,b,X)
        L=square_loss(y,y_hat)
        L.sum().backward()
        sgd([w,b],lr,batch_size)
        print(f'epoch {i+1},loss == {float(L.detach().mean()):f}')
print(w)
print(b)

