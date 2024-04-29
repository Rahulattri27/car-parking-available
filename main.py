import pickle
import numpy as np
import cv2

cap=cv2.VideoCapture("/Users/rahulkumarair/Documents/rahul_vsCode/litter_moodel.ipynb/car parking/carPark.mp4")

with open('CarParkPos','rb') as file:
        posList=pickle.load(file)
saved_model=pickle.load(open('/Users/rahulkumarair/Documents/rahul_vsCode/litter_moodel.ipynb/car parking/finalised_model.sav','rb'))

width,height=107,48

def checkParkingSpace(imgProcessed):
    space_counter=0
    free_pos=[]
    for pos in posList:
        x,y=pos
        imgCrop=imgProcessed[y:y+height,x:x+width]
        time_length = 30.0
        fps=25
        frame_seq = 749
        frame_no = (frame_seq /(time_length*fps))
        count=saved_model.predict(imgCrop) 
        cv2.putText(img,str(posList.index(pos)),(x,y+height-10 ),1,2,(255,0,0),1)

        if count==1:
             color=(0,255,0)
             thickness=5
             space_counter+=1
             free_pos.append(posList.index(pos))
        else:
             color=(0,0,255)
             thickness=2
        cv2.rectangle(img,pos,(pos[0]+width,pos[1]+height),color,thickness)
        

    cv2.imshow(str(x*y),imgCrop)
    
         
    cv2.putText(img,str(f'Free :{space_counter}/{len(posList)}'),(100,50),2,1,(0,255,0),2)
    cv2.putText(img,"Free spaces:",(70,90),2,1,(0,255,255),2)
    cv2.putText(img,str(free_pos[:len(free_pos)]),(300,90),2,1,(0,255,255),2)
    
    

     


while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES)==cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    sucess,img=cap.read()
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(3,3),1)
    imgThreshold=cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
    imgMedian=cv2.medianBlur(imgThreshold,5)
    kernel=np.ones((3,3),np.uint8)
    imgDilate=cv2.dilate(imgMedian,kernel,iterations=1)


    checkParkingSpace(imgDilate)
    # for pos in posList:
    #     cv2.rectangle(img,pos,(pos[0]+width,pos[1]+height),(255,0,255),1)
 
    cv2.imshow("image",img)
    # cv2.imshow("imagegray",imgGray)
    # cv2.imshow("imageblur",imgBlur)
    # cv2.imshow("imgthreshold",imgMedian)

    if cv2.waitKey(10)==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()