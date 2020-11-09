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
#directory = '../TP2_Videos_Exemples/'
directory = './TP2_Videos/'
nVal = 256
#nValU = nVal
#nValV = nVal
histUV = np.zeros((nVal,nVal))
histUV_old = np.zeros((nVal,nVal))
cap = cv2.VideoCapture(directory+filename)

ret,frame = cap.read()
index = 0
ret = True
while(ret):
    frame_Yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #frame_uv = frame_Yuv[:,:,1:]
    
    histUV = cv2.calcHist([frame_Yuv],[1,2], None, [256,256], [0,256,0,256])
    histShape = np.shape(histUV)
    HumanVisibleHistogrammeUV = np.zeros((histShape[0]*3,histShape[1]*3))
    HumanVisibleHistogrammeUV[0::3,0::3] = histUV
    HumanVisibleHistogrammeUV[0::3,1::3] = histUV
    HumanVisibleHistogrammeUV[0::3,2::3] = histUV
    HumanVisibleHistogrammeUV[1::3,0::3] = histUV
    HumanVisibleHistogrammeUV[1::3,1::3] = histUV
    HumanVisibleHistogrammeUV[1::3,2::3] = histUV
    HumanVisibleHistogrammeUV[2::3,0::3] = histUV
    HumanVisibleHistogrammeUV[2::3,1::3] = histUV
    HumanVisibleHistogrammeUV[2::3,2::3] = histUV
    
    cv2.imshow('test affichage',frame)
    #cv2.imshow('histogramme',histUV/(histUV.max()-histUV.min()))
    cv2.imshow('histogramme visible by human',HumanVisibleHistogrammeUV/(HumanVisibleHistogrammeUV.max()-HumanVisibleHistogrammeUV.min()))
    
    if index>0:
        distance = cv2.compareHist(histUV,histUV_old,cv2.HISTCMP_CORREL)
        if distance <0.3:
            print(distance)
            cv2.imwrite('image%04d.png'%index,frame)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('image%04d.png'%index,frame)
        cv2.imwrite('hist%04d.png'%index,histUV)
    elif k == ord('p'):
        while k != ord('g'):
            time.sleep(1)
            k = cv2.waitKey(30) & 0xff
    index += 1
    histUV_old = histUV
    ret,frame = cap.read()
    if (frame==[]):
        print("The end")


cv2.destroyAllWindows()
cap.release()