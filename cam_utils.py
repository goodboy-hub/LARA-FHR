'''
Since our model is 1-d CNN model, we fixed the Grad-
'''
from pytorch_grad_cam import GradCAM
import numpy as np
import torch

def calculate_cam(sampled_data,model,device):
    cam=GradCAM(model=model, target_layers=[model.stage_list[4]])
    cam_table=np.zeros_like(sampled_data)
    for idx in range(len(sampled_data)):
        input_tensor=torch.tensor((np.expand_dims(np.expand_dims(sampled_data[idx,:],0),1)),device=device,dtype=torch.float)
        cam_point=cam(input_tensor).reshape(2400,)
        cam_table[idx,:]=cam_point
    assert idx == len(sampled_data)-1
    return cam_table


#-------------------------------------------------------------------------------------------------------#
def infusing_cam(cam_value):
    infused_cam=cam_value.reshape(len(cam_value),10,240).mean(axis=2)
    monitor_time=len(cam_value)+9
    long_cam=np.zeros(shape=(monitor_time))
    for idx in range(monitor_time):
        if idx>=9 and monitor_time-1-idx>=9:
            weights=[infused_cam[idx-i,i] for i in range(9,-1,-1)]##这个是正着取的 
            assert len(weights)==10
        if idx<=8:
            weights=[infused_cam[i,idx-i] for i in range(idx+1)]
        if monitor_time-1-idx<=8:
            weights=[infused_cam[idx-i,i] for i in range(9,idx+9-monitor_time,-1)]
        long_cam[idx]=sum(weights)/len(weights)
    return long_cam
        

