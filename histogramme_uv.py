#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:14:50 2020
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

filename = 'Extrait1-Cosmos_Laundromat1(340p).m4v'
#filename = 'Rotation_OY(Pan).m4v'
#directory = '../TP2_Videos_Exemples/'
directory = './TP2_Videos/'
nVal = 256
nbImages = 3168
#nValU = nVal
#nValV = nVal
histUV = np.zeros((nVal,nVal))
histUV_old = np.zeros((nVal,nVal))
cap = cv2.VideoCapture(directory+filename)

dist_Correl = np.zeros(nbImages)
dist_ChiSquare = np.zeros(nbImages)
dist_Intersection = np.zeros(nbImages)
dist_Bhattacharyya = np.zeros(nbImages)
dist_Hellinger = np.zeros(nbImages)
dist_ChiSquareAlt = np.zeros(nbImages)
dist_KLDiv = np.zeros(nbImages)
X = np.arange(nbImages)

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
        dist_Correl[index] = cv2.compareHist(histUV,histUV_old,cv2.HISTCMP_CORREL)
        dist_ChiSquare[index] = cv2.compareHist(histUV,histUV_old,cv2.HISTCMP_CHISQR)
        dist_Intersection[index] = cv2.compareHist(histUV,histUV_old,cv2.HISTCMP_INTERSECT)
        dist_Bhattacharyya[index] = cv2.compareHist(histUV,histUV_old,cv2.HISTCMP_BHATTACHARYYA)
        dist_Hellinger[index] = cv2.compareHist(histUV,histUV_old,cv2.HISTCMP_HELLINGER)
        dist_ChiSquareAlt[index] = cv2.compareHist(histUV,histUV_old,cv2.HISTCMP_CHISQR_ALT)
        dist_KLDiv[index] = cv2.compareHist(histUV,histUV_old,cv2.HISTCMP_KL_DIV)

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

plt.plot(X,dist_Correl, label = "Correlation")
plt.plot(X,dist_ChiSquare, label = "ChiSquare")
plt.plot(X,dist_Intersection, label = "Intersection")
plt.plot(X,dist_Bhattacharyya, label = "Bhattacharyya")
plt.plot(X,dist_Hellinger, label = "Hellinger")
plt.plot(X,dist_ChiSquareAlt, label = "ChiSquareAlt")
plt.plot(X,dist_KLDiv, label = "KLDiv")
plt.legend()
plt.title("All possible distances in compareHist")
plt.show()


cv2.destroyAllWindows()
cap.release()