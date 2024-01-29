'''
LARA = cnn_model + information fusion

LARA will use the inner cnn model to scan a long-term FHR data in step of 1 minute and output a Risk Distribution 
Map (RDM) and a Risk Inex (RI) for the long-term data

'''
from information_fusion import *
from cam_utils import *
from net1d import *
from torch.utils.data import Dataset,DataLoader
from LARA_plot import *

def split_long(long_data, step=4*60, size=4*60*10):
    monitor_time=int(len(long_data)/(4*60))
    sampled_data=np.zeros(shape=(monitor_time-10+1,4*60*10))
    for idx,start in enumerate(range(0,len(long_data)-size+step, step)):
        sampled_data[idx,:]=long_data[start:start+2400]
        if idx!=0:
            assert sum(sampled_data[idx-1,-(2400-240):]==sampled_data[idx,0:2400-240])==240*9
    assert idx==monitor_time-10
    return sampled_data

def model_predict(splited_data,model,device):
    class temp_dataset(Dataset):
        def __init__(self, data):
            self.data=data
        def __getitem__(self, index):
            return self.data[index]
        def __len__(self):
            return len(self.data)
    temp_iter=DataLoader(temp_dataset(splited_data),batch_size=64,shuffle=False)
    model.eval()
    result=np.array([])
    with torch.no_grad():
        for x in temp_iter:
            input_tensor=torch.tensor(np.expand_dims(x,1)).to(device,torch.float32)
            out_y=np.array(torch.sigmoid(model(input_tensor)).to('cpu')).reshape(-1)##检验顺序是没问题的
            result=np.concatenate([result,out_y])
    return result

class LARA():
    def __init__(self,device_str) :
        'initialize the cnn kernel'
        self.net=Net1D(in_channels=1,base_filters=256,ratio=1,
                    filter_list=[256,512,512,1024,1024],m_blocks_list=[2,3,3,2,2],
                    kernel_size=16,stride=2,
                    n_classes=1,use_bn=True,use_do=True,verbose=False,groups_width=16)
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.net.load_state_dict(torch.load('model_state',map_location=self.device))
        print('cnn model loaded, you can use LARA.process to analysis your data')
    
    def split_long(self,long_data, step=4*60, size=4*60*10):
        monitor_time=int(len(long_data)/(4*60))
        splited_data=np.zeros(shape=(monitor_time-10+1,4*60*10))
        for idx,start in enumerate(range(0,len(long_data)-size+step, step)):
            splited_data[idx,:]=long_data[start:start+2400]
            if idx!=0:
                assert sum(splited_data[idx-1,-(2400-240):]==splited_data[idx,0:2400-240])==240*9
        assert idx==monitor_time-10
        return splited_data
    
    def model_predict(self,splited_data):
        class temp_dataset(Dataset):
            def __init__(self, data):
                self.data=data
            def __getitem__(self, index):
                return self.data[index]
            def __len__(self):
                return len(self.data)
        temp_iter=DataLoader(temp_dataset(splited_data),batch_size=64,shuffle=False)
        self.net.eval()
        result=np.array([])
        with torch.no_grad():
            for x in temp_iter:
                input_tensor=torch.tensor(np.expand_dims(x,1)).to(self.device,torch.float32)
                out_y=np.array(torch.sigmoid(self.net(input_tensor)).to('cpu')).reshape(-1)##检验顺序是没问题的
                result=np.concatenate([result,out_y])
        return result

    def process(self,long_FHR) :
        splited_data=split_long(long_FHR)
        y_hat=model_predict(splited_data)
        cam_value=calculate_cam(splited_data,model=self.net, device=self.device)
        self.weighted_result=information_fusion(y_hat,cam_value,use_weight=True,use_cam=True)
        self.unweighted_result=information_fusion(y_hat,cam_value,False,use_cam=False)
        self.R_S_result=information_fusion(y_hat,cam_value,use_weight=True,use_cam=False)
        self.long_cam=infusing_cam(cam_value)

    def out_put(self, operator='R_S'):
        '''
        Three operators is supplied in LARA
        Basic operator: Using average method to fuse the model prefiction
        Risk-Sensitive (R-S) operator: emphasize the point of severe risk
        Cam-Weighted (C-W) operator: weight the model prediction by the cam value, reflecting the attention of model 
        '''

        if operator == 'Basic':
            classic_bar(self.unweighted_result,x_lable='Time/minute',y_lable='risk index',title='RDM of abnormal monitor')
            print(f'Risk Index (RI):\nBasic: {self.unweighted_result.mean()}')
        if operator == 'R_S':
            classic_bar(self.R_S_result,x_lable='Time/minute',y_lable='risk index',title='RDM of abnormal monitor')
            print(f'Risk Index (RI):\nR_S: {self.unweighted_result.mean()}')
        if operator == 'C_W':
            classic_bar(self.weighted_result,x_lable='Time/minute',y_lable='risk index',title='RDM of abnormal monitor')
            print(f'Risk Index (RI):\nC_M: {self.unweighted_result.mean()}')