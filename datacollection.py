import math
import time

import cap as cap

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
offset=20
imgsize=300

folder="Data/C"
counter=0
while True:
    success, img = cap.read()
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
        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgcrop, (imgsize,hCal))
            imgResizeshape = imgResize.shape
            hGap = math.ceil((imgsize - hCal) / 2)
            imgwhite[hGap:hGap + hCal,:] = imgResize

        cv2.imshow("imagecrop", imgcrop)
        cv2.imshow("imagewhite", imgwhite)
    cv2.imshow("image", img)
    key=cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(counter)


