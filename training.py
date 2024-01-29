import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from net1d import *

'''
Import the training data, replace the sata_path of your own
Data we use is in form of n*2401  fs=4hz  1(label)+4*60*10
'''
data_path=r'/data/resampled_train_data.npy'

class train_set(Dataset):
    def __init__(self):
        self.dataset=np.load(data_path)
    def __getitem__(self, index):
        return(self.dataset[index,1:2401],self.dataset[index,0])
    def __len__(self):
        return len(self.dataset)
    def get_all(self):
        return (self.dataset[:,1:2401],self.dataset[:,0])
    
train_dataset=train_set()
train_iter=DataLoader(train_dataset,batch_size=64,shuffle=True)

'The structure of cnn model in LARA'
net=Net1D(in_channels=1,base_filters=256,ratio=1,
          filter_list=[256,512,512,1024,1024],m_blocks_list=[2,3,3,2,2],
          kernel_size=16,stride=2,
          n_classes=1,use_bn=True,use_do=True,verbose=False,groups_width=16)##理论上n_class永远不应该是2

device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
net.to(device=device)
lossfun=nn.BCELoss()
updater=torch.optim.Adam(net.parameters(),lr=1e-3)

train_loss=[]
for epoch in range(200):
    net.train()
    epochloss=[]
    for train_x,train_y in train_iter:
        train_x=torch.tensor(np.expand_dims(train_x,1)).to('cuda',torch.float32)
        train_y=torch.tensor(np.expand_dims(np.array(train_y<=3),1)).to('cuda',torch.float32)
        y_hat=torch.sigmoid(net(train_x))
        loss=lossfun(y_hat,train_y)
        epochloss.append(loss)
        updater.zero_grad()
        loss.backward()
        updater.step()
    ls=sum(epochloss)/len(epochloss)
    print(f"epoch{epoch},loss:{ls}")
    train_loss.append(ls)
print('modle training down!')

torch.save(net.state_dict(),'model_state')
print('model state been preserved in model_state')