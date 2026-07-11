import os
import pandas as pd
import torch
import numpy as np

data=pd.read_csv(os.path.join('.','data','house_cond.csv'))
data['NumRooms']=data['NumRooms'].fillna(data['NumRooms'].mean())
data=pd.get_dummies(data,dummy_na=True,dtype=float)
print(data)
data=torch.tensor(data.to_numpy(dtype=float))
X=data[:,[0,2]]
y=data[:,1]
tmp=torch.ones(10,1)
X=torch.cat((X,tmp),dim=1)
# print(X)
w=(X.T@X).inverse()@X.T@y
print(w)
test=torch.tensor([3.5,1,1])
print((w*test).sum())
