import numpy as np
import cv2
def transform_point(perspective_matrix,x,y):
    point_homogeneous = np.array([x,y,1])
    #Perform matrix multiplication to get the transformed point
    transformed_point_homogeneous = np.dot(perspective_matrix,point_homogeneous)
    #Divide by the third component (w) to obtain the transformed(x',y') coordinates
    transformed_point=transformed_point_homogeneous[:2]/transformed_point_homogeneous[2]
    return transformed_point


# pts = [[],[],[],[]]

def warpProcess(img,pts):
    hh, ww = img.shape[:2]
    input = np.float32(pts)
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