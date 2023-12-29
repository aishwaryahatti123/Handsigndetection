
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np


cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
Classifier=Classifier("model/keras_model.h5","model/labels.txt")
offset=20
imgsize=300

folder="Data/C"
counter=0
label=["A","B","C"]
while True:
    success, img = cap.read()
    imgOutput=img.copy()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']


        imgwhite=np.ones((imgsize,imgsize,3),np.uint8)*255
        imgcrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgcropshape=imgcrop.shape

        aspectRatio=h/w
        if aspectRatio>1:
            k=imgsize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgcrop,(wCal,imgsize))
            imgResizeshape=imgResize.shape
            wGap=math.ceil((imgsize-wCal)/2)
            imgwhite[:,wGap:wGap+wCal] = imgResize
            prediction,index=Classifier.getPrediction(imgwhite)
            print(prediction,index)


        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgcrop, (imgsize,hCal))
            imgResizeshape = imgResize.shape
            hGap = math.ceil((imgsize - hCal) / 2)
            imgwhite[hGap:hGap + hCal,:] = imgResize
            prediction, index = Classifier.getPrediction(imgwhite,draw=False)

        cv2.putText(imgOutput,label[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)

        cv2.imshow("imagecrop", imgcrop)
        cv2.imshow("imagewhite", imgwhite)
    cv2.imshow("image", imgOutput)
    cv2.waitKey(1)

