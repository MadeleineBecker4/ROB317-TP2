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
directory = '../TP2_Videos_Exemples/'
#directory = './TP2_Videos/'
nVal = 256
nbImages = 3168

histVxVy = np.zeros((nVal,nVal))
#histVxVy_old = np.zeros((nVal,nVal))
cap = cv2.VideoCapture(directory+filename)

dist_Correl = np.zeros(nbImages)
dist_ChiSquare = np.zeros(nbImages)
dist_Intersection = np.zeros(nbImages)
dist_Bhattacharyya = np.zeros(nbImages)
dist_Hellinger = np.zeros(nbImages)
dist_ChiSquareAlt = np.zeros(nbImages)
dist_KLDiv = np.zeros(nbImages)
X = np.arange(nbImages)

VxMax = np.zeros(nbImages)
VxMin = np.zeros(nbImages)
VyMax = np.zeros(nbImages)
VyMin = np.zeros(nbImages)



ret, frame1 = cap.read() # Passe à l'image suivante
prvs_frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
hsv = np.zeros_like(frame1) # Image nulle de même taille que frame1 (affichage OF)
hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

while(ret):
    index += 1
    flow = cv2.calcOpticalFlowFarneback(prvs_frame,next_frame,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)	
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartésien vers polaire
    hsv[:,:,0] = (ang*180)/(2*np.pi) # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:,:,2] = (mag*255)/np.amax(mag) # Valeur <--> Norme 

    VxMax[index] = flow[:,:,0].max()
    VxMin[index] = flow[:,:,0].min()
    VyMax[index] = flow[:,:,1].max()
    VyMin[index] = flow[:,:,1].min()
    
    histVxVy = cv2.calcHist([flow],[0,1], None, [256,256], [-20,20,-20,20])
    cv2.imshow('histogramme',histVxVy/(histVxVy.max()-histVxVy.min()))


    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    result = np.vstack((frame2,bgr))
    cv2.imshow('Image et Champ de vitesses (Farneback)',result)
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
    elif k == ord('p'):
        while k != ord('g'):
            time.sleep(1)
            k = cv2.waitKey(30) & 0xff
    prvs_frame = next_frame
    ret, frame2 = cap.read()
    if (ret):
        next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 


cv2.destroyAllWindows()
cap.release()

plt.plot(X,VxMax,label = "VxMax")
plt.plot(X,VxMin,label = "VxMin")
plt.plot(X,VyMax,label = "VyMax")
plt.plot(X,VyMin,label = "VyMin")
plt.legend()
plt.show()































'''
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
    if not(ret):
        print("The end")

renorm_dist_Correl = 1-dist_Correl/dist_Correl.max()
renorm_dist_ChiSquare = dist_ChiSquare/dist_ChiSquare.max()
renorm_dist_Intersection = 1-dist_Intersection/dist_Intersection.max()
renorm_dist_Bhattacharyya = dist_Bhattacharyya/dist_Bhattacharyya.max()
renorm_dist_Hellinger = dist_Hellinger/dist_Hellinger.max()
renorm_dist_ChiSquareAlt = dist_ChiSquareAlt/dist_ChiSquareAlt.max()
renorm_dist_KLDiv = dist_Correl/dist_KLDiv.max()
'''

'''
renorm_moy_dist = (renorm_dist_Correl + \
                   renorm_dist_ChiSquare + \
                   renorm_dist_Intersection + \ 
                   renorm_dist_Bhattacharyya + \
                   renorm_dist_Hellinger + \
                   renorm_dist_ChiSquareAlt + \
                   renorm_dist_KLDiv)/7
'''

'''
renorm_moy_dist = (renorm_dist_Correl +renorm_dist_ChiSquare +  renorm_dist_Intersection + renorm_dist_Bhattacharyya +  renorm_dist_Hellinger + renorm_dist_ChiSquareAlt + renorm_dist_KLDiv)/7

plt.plot(X,renorm_dist_Correl, label = "Correlation")
plt.plot(X,renorm_dist_ChiSquare, label = "ChiSquare")
#plt.plot(X,renorm_dist_Intersection, label = "Intersection")
#plt.plot(X,renorm_dist_Bhattacharyya, label = "Bhattacharyya")
#plt.plot(X,renorm_dist_Hellinger, label = "Hellinger")
#plt.plot(X,renorm_dist_ChiSquareAlt, label = "ChiSquareAlt")
plt.plot(X,renorm_dist_KLDiv, label = "KLDiv")
plt.legend()
plt.title("All possible distances in compareHist")
plt.show()

plt.plot(X,renorm_moy_dist)
plt.title("moy dist")
plt.show()


cv2.destroyAllWindows()
cap.release()
'''