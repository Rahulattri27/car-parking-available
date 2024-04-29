import pickle
import cv2
try:
    with open('CarParkPos','rb') as file:
        posList=pickle.load(file)
except:
    posList=[]
width,height=107,48


def mouseClick(events,x,y,flags,params):
    if events==cv2.EVENT_LBUTTONDOWN:
        posList.append((x,y))
    if events==cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1,y1=pos
            if x1<x<x1+width and y1<y<y1+height:
                posList.pop(i)
    with open('CarParkPos','wb') as file:
        pickle.dump(posList,file)
    

while True:
    cap=cv2.imread("/Users/rahulkumarair/Documents/rahul_vsCode/litter_moodel.ipynb/car parking/carParkImg.png")


    # cv2.rectangle(cap,(50,192),(157,240),(255,0,255),1)
    for pos in posList:
        cv2.rectangle(cap,pos,(pos[0]+width,pos[1]+height),(255,0,255),1)

    cv2.imshow("image",cap)
    cv2.setMouseCallback("image",mouseClick)
    if cv2.waitKey(1)==ord("q"):
        break


