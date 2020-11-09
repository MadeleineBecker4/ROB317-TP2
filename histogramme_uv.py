#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:14:50 2020
"""


import numpy as np
import cv2
import time

filename = 'Extrait1-Cosmos_Laundromat1(340p).m4v'
#filename = 'Rotation_OY(Pan).m4v'
directory = '../TP2_Videos_Exemples/'
#directory = './TP2_Videos/
nVal = 256
#nValU = nVal
#nValV = nVal
histogrammeUV = np.zeros((nVal,nVal))
cap = cv2.VideoCapture(directory+filename)

ret,frame = cap.read()
index = 0
ret = True
while(ret):
    frame_Yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #frame_uv = frame_Yuv[:,:,1:]
    
    histogrammeUV = cv2.calcHist([frame_Yuv],[1,2], None, [256,256], [0,256,0,256])
    
    cv2.imshow('test affichage',frame_Yuv)
    cv2.imshow('histogramme',histogrammeUV/(histogrammeUV.max()-histogrammeUV.min()))
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('image%04d.png'%index,frame)
        cv2.imwrite('hist%04d.png'%index,histogrammeUV)
    elif k == ord('p'):
        while k != ord('g'):
            time.sleep(1)
            k = cv2.waitKey(30) & 0xff
    index += 1
    ret,frame = cap.read()


cv2.destroyAllWindows()
cap.release()