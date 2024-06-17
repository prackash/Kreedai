import cv2
from ultralytics import YOLO
import math
import numpy as np
yhit=[]
xhit=[]
tframe=[]
# Function to process frame and draw detections
def process_frame(frame, model):
    results = model(frame)
    for result in results:
        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, score, class_id = box
            if score > 0.05:  # Confidence threshold
                label = f"{model.names[int(class_id)]} {score:.2f}"
                color = (0, 255, 0)  # Green color for the bounding box
                xhit.append((x1+x2)/2)
                yhit.append((y1+y2)/2)
                tframe.append(frame.copy())
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# Main function to process video
def process_video(input_video_path, output_video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model)
        out.write(frame)
        cv2.imshow('YOLOv8 Object Detection', frame)
        
    i=yhit.index(max(yhit))
    cv2.imwrite("/Users/prackash/Developer/tensorflow/yolov8/outimage.jpeg",tframe[i])
    cap.release()
    out.release()
    return tframe[i],i
    
def transform_point(perspective_matrix,x,y):
    point_homogeneous = np.array([x,y,1])
    #Perform matrix multiplication to get the transformed point
    transformed_point_homogeneous = np.dot(perspective_matrix,point_homogeneous)
    #Divide by the third component (w) to obtain the transformed(x',y') coordinates
    transformed_point=transformed_point_homogeneous[:2]/transformed_point_homogeneous[2]
    return transformed_point
def getInput(img):
    hh, ww = img.shape[:2]
    print(hh,ww)
    print(type(hh),type(ww))
    tl=1
    while tl:
        tlx,tly = map(int,input("Enter x & y coordinate for the Top Left corner: ").split())
        
        if (tlx>ww) or (tly>hh):
            print(f"Out of bounds width = {ww}, height = {hh}.\n Coordinates chosen by you ({tlx},{tly})")
            continue
        ltimg=np.copy(img)
        ltimg=cv2.circle(ltimg,(int(tlx),int(tly)),5,(0,255,0),-1)
        tl=int(input("Press 1 to reenter coordinates else Press 0"))
    tl=1
    while tl:
        xtr,ytr =map(int,input("Enter x & y coordinate for the Top Right corner: ").split())
        if (xtr>hh) or (ytr>ww):
            print(f"Out of bounds height = {hh}, width = {ww}.\n Coordinates chosen by you ({xtr},{ytr})")
            continue
        rtimg=np.copy(ltimg)
        rtimg=cv2.circle(rtimg,(int(xtr),int(ytr)),5,(0,255,0),-1)
        tl=int(input("Press 1 to reenter coordinates else Press 0"))
    tl=1
    while tl:
        blx,bly = map(int,input("Enter x & y coordinate for the Bottom Left corner: ").split())
        if (blx>hh) or (bly>ww):
            print(f"Out of bounds height = {hh}, width = {ww}.\n Coordinates chosen by you ({blx},{bly})")
            continue
        lbimg=np.copy(rtimg)
        lbimg=cv2.circle(lbimg,(int(blx),int(bly)),5,(0,255,0),-1)
        tl=int(input("Press 1 to reenter coordinates else Press 0"))
    tl=1
    while tl:
        xbr,ybr = map(int,input("Enter x & y coordinate for the Bottom Right corner: ").split())
        if (xbr>hh) or (ybr>ww):
            print(f"Out of bounds height = {hh}, width = {ww}.\n Coordinates chosen by you ({xbr},{ybr})")
            continue
        rbimg=img.copy()
        rbimg=cv2.circle(rbimg,(int(xbr),int(ybr)),5,(0,255,0),-1)
        tl=int(input("Press 1 to reenter coordinates else Press 0"))
    return [[tlx,tly],[xtr,ytr],[xbr,ybr],[blx,bly]]

def warpProcess(img):
    hh, ww = img.shape[:2]
    input = np.float32(getInput(img))
    for val in input:
        img=cv2.circle(img,(int(val[0]),int(val[1])),5,(0,255,0),-1)
    cv2.imshow("img2",img)
    width_top = np.sqrt((input[1][0] - input[0][0])**2 + (input[1][1] - input[0][1])**2)
    width_bottom = np.sqrt((input[3][0] - input[2][0])**2 + (input[3][1] - input[2][1])**2)
    height_left = np.sqrt((input[2][0] - input[0][0])**2 + (input[2][1] - input[0][1])**2)
    height_right = np.sqrt((input[3][0] - input[1][0])**2 + (input[3][1] - input[1][1])**2)
    width = width_top
    height = max(int(height_left), int(height_right))
    
    x = input[0,0]
    y = input[0,1]

    output = np.float32([[x,y], [x+width-1,y], [x+width-1,y+height-1], [x,y+height-1]])

    print("width : ", width ,", height : ", height ,", X : ",x,", Y : ",y)
    print("input: ",input)
    print("output: ",output)
    matrix = cv2.getPerspectiveTransform(input,output)
    imgOutput = cv2.warpPerspective(img, matrix, (ww,hh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    cv2.imshow("Transformed",imgOutput)
    return imgOutput,matrix,input,width,height


# Paths to the model file
model_path = '/Users/prackash/Developer/tensorflow/yolov8/runs/detect/train6/weights/best.pt'  # Replace with the path to your YOLOv8 model file

# Input and output video paths
input_video_path = '/Users/prackash/Developer/tensorflow/yolov8/test.mp4'
output_video_path = '/Users/prackash/Developer/tensorflow/yolov8/outtest.mp4'

# Run the video processing
image,idx=process_video(input_video_path, output_video_path, model_path)
warpImg,matrix,input,width,height=warpProcess(tframe[idx])

TL=transform_point(matrix,input[0,0],input[0,1])
TR=transform_point(matrix,input[1,0],input[1,1])
BR=transform_point(matrix,input[2,0],input[2,1])
BL=transform_point(matrix,input[3,0],input[3,1])
min_x= int(min(TL[0],TR[0],BL[0],BR[0]))
min_y= int(min(TL[1],TR[1],BL[1],BR[1]))
max_x= int(max(TL[0],TR[0],BL[0],BR[0]))
max_y= int(max(TL[1],TR[1],BL[1],BR[1]))

crop_img=warpImg[min_y:max_y,min_x:max_x]

a=xhit[idx]
b=yhit[idx]
c=int(a-min_x)
d=int(b-min_y)
pred=cv2.circle(crop_img,(int(c),int(d)),3,(255,0,0),4)
cv2.imshow("prediction",pred)

th,tw,_=pred.shape

cv2.rectangle(pred, (int((width/2)-((width/305)*22.86)),0),(int((width/2)+((width/305)*22.86)),height),(255,0,0),1)
height,width,_=pred.shape
t=0
f=int((height/20.2)*2)
b=f
length=""
if (d<b and d>t):
    length="Full Toss"
    
ft=cv2.rectangle(pred,(0,t),(width,b),(100,100,100),1) #full toss
t=b
b+=f
if (d<b and d>t):
    length="Yorker"
y=cv2.rectangle(ft,(0,t),(width,b),(100,200,100),1)# yorker
t=b
b+=f
if (d<b and d>t):
    length="The Slot"
ts=cv2.rectangle(y,(0,t),(width,b),(100,100,200),1)# the slot
t=b
b+=f
if (d<b and d>t):
    length="Length"
l=cv2.rectangle(ts,(0,t),(width,b),(200,200,100),1)# length
t=b
b=height
if (d<b and d>t):
    length="Short"
s=cv2.rectangle(l,(0,t),(width,b),(200,200,200),1) # short
line=0
if (c<int((width/2)-((width/305)*22.86))):
    line=-1
elif (c>int((width/2)+((width/305)*22.86))):
    line=1
print(line)
print(length)

cv2.imshow('Length',s)
cv2.imwrite("/Users/prackash/Developer/tensorflow/yolov8/length.jpeg",s)
cv2.waitKey(0)

cv2.destroyAllWindows()