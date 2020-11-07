#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:14:50 2020

@author: iad
"""


import numpy as np
import cv2

filename = 'Extrait1-Cosmos_Laundromat1(340p).m4v'
#filename = 'Rotation_OY(Pan).m4v'
directory = '../TP2_Videos_Exemples/'
nVal = 256
#nValU = nVal
#nValV = nVal
histogrammeUV = np.zeros((nVal,nVal))
cap = cv2.VideoCapture(directory+filename)

index = 0
ret = True
while(ret):
    ret,frame = cap.read()
    frame_Yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #frame_uv = frame_Yuv[:,:,1:]
    index += 1
    for i in range (0,nVal):
        for j in range (0,nVal):
            goodU = (frame_Yuv[:,:,1] == i)
            goodUgoodV = (frame_Yuv[:,:,1]
            
            histogrammeUV[i][j] = np.count_nonzero(frame_uv[:][:] == [i,j])
    
    
    cv2.imshow('test affichage',frame_Yuv)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('OF_PyrLk%04d.png'%index,img)


cv2.destroyAllWindows()
cap.release()