#importing necessary librearies
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform,VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio



#defining detect function
def detect(frame,net,transform):
    height,width=frame.shape[:2]
    frame_t=transform(frame)[0]
    x=torch.from_numpy(frame_t).permute(2,0,1)
    x=Variable(x.unsqueeze(0))
    y=net(x)
    detection=y.data
    scale=torch.Tensor([width,height,width,height])
    #detection contains [batch,number of classes,number of occurance,(score,x0,x1,x2,x3)]
    for i in range(detection.size(1)):
        j=0
        while detection[0,i,j,0]>=0.6:
            pt=(detection[0,i,j,1:]*scale).numpy()
            cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),pt[3]),(255,0,0),2)
            cv2.putText(frame,labelmap[i-1],(int(pt[0]),int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
            j+=1
    return frame


#creating ssd neural network
net=build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location=lambda storage,loc:storage))

#creating transformation
transform=BaseTransform(net.size,(104/256.0,117/256.0,123/256.0))

#Doing object detection on video
reader=imageio.get_reader('epic_horses.mp4')
fps=reader.get_meta_data()['fps']
writer=imageio.get_writer('output_final.mp4',fps=fps)
for i,frame in enumerate(reader):
    frame=detect(frame,net.eval(),transform)
    writer.append_data(frame)
    print(i)
writer.close()
    