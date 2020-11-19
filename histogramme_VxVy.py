#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:14:50 2020
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import auxFunctions as af

filename = 'Rotation_OZ(Roll).m4v'
#filename = 'Rotation_OY(Pan).m4v'
directory = '../TP2_Videos_Exemples/'
#directory = './TP2_Videos/'
imgDir = '../figure/newfig/'
#imgDir = './Images/'
# indice de la frame juste apres une coupure
cutGroundTruth = af.getCutGroundTruth(filename)
nbImages = af.getNbFrame(filename,100000) # nombre de frame de la video

histSize = 61 # impaire
vxMinHist = -20
vxMaxHist = 20
vyMinHist = -20
vyMaxHist = 20
vxRange = (np.arange(histSize) +1/2) * (vxMaxHist - vxMinHist) / histSize + vxMinHist
vyRange = (np.arange(histSize) +1/2) * (vyMaxHist - vyMinHist) / histSize + vyMinHist
vxGrid, vyGrid = np.meshgrid(vxRange, vyRange, indexing = 'ij')
V_unrolled = np.array([vxGrid.ravel(),vyGrid.ravel()])
meanVx = np.zeros(nbImages)
meanVy = np.zeros(nbImages)
covV = np.zeros((nbImages,2,2))


histVxVy = np.zeros((histSize,histSize))
histShape = np.shape(histVxVy)
zoomHistVxVy = np.zeros((histShape[0]*3, histShape[1]*3))
# histogramme obtenue pour une scene statique
histStatic = np.zeros((histSize, histSize), dtype=np.float32)
histStatic[histSize//2,histSize//2] = 1.

cap = cv2.VideoCapture(directory+filename)

dist_Correl = np.zeros(nbImages)
dist_ChiSquare = np.zeros(nbImages)
dist_Intersection = np.zeros(nbImages)
dist_Bhattacharyya = np.zeros(nbImages)
dist_Hellinger = np.zeros(nbImages)
dist_ChiSquareAlt = np.zeros(nbImages)
dist_KLDiv = np.zeros(nbImages)

dist_Correl_static = np.zeros(nbImages)
dist_ChiSquare_static = np.zeros(nbImages)
dist_Intersection_static = np.zeros(nbImages)
dist_Bhattacharyya_static = np.zeros(nbImages)
dist_Hellinger_static = np.zeros(nbImages)
dist_ChiSquareAlt_static = np.zeros(nbImages)
dist_KLDiv_static = np.zeros(nbImages)

X = np.arange(nbImages)

VxMax = np.zeros(nbImages)
VxMin = np.zeros(nbImages)
VyMax = np.zeros(nbImages)
VyMin = np.zeros(nbImages)


ret, frame1 = cap.read() # Passe à l'image suivante
prvs_frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
hsv = np.zeros_like(frame1) # Image nulle de même taille que frame1 (affichage OF)
hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum

index = 0
ret, frame2 = cap.read()
next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

while(ret):
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
    
    histVxVy = cv2.calcHist([flow],[0,1], None, [histSize,histSize], [vxMinHist,vxMaxHist,vyMinHist,vyMaxHist])
    histVx = np.sum(histVxVy, axis = 1)
    histVy = np.sum(histVxVy, axis = 0)
    meanVx[index] = sum(vxRange*histVx) /sum(histVx)
    meanVy[index] = sum(vyRange*histVy) /sum(histVy)
    unrolled_histVxVy = histVxVy.ravel()
    covV[index,:,:] = np.cov(V_unrolled, fweights = unrolled_histVxVy)
    
    zoomHistVxVy[0::3, 0::3] = histVxVy
    zoomHistVxVy[0::3, 1::3] = histVxVy
    zoomHistVxVy[0::3, 2::3] = histVxVy
    zoomHistVxVy[1::3, 0::3] = histVxVy
    zoomHistVxVy[1::3, 1::3] = histVxVy
    zoomHistVxVy[1::3, 2::3] = histVxVy
    zoomHistVxVy[2::3, 0::3] = histVxVy
    zoomHistVxVy[2::3, 1::3] = histVxVy
    zoomHistVxVy[2::3, 2::3] = histVxVy
    
    normalizedHistVxVy = zoomHistVxVy/(zoomHistVxVy.max()-zoomHistVxVy.min())
    sqrtHistVxVy = np.sqrt(normalizedHistVxVy)
    cv2.imshow('histogramme',normalizedHistVxVy)
    cv2.imshow('sqrt(histogramme)',sqrtHistVxVy)

    # distance entre deux histogramme de vistesse consecutif
    if index>0:
        dist_Correl[index] = cv2.compareHist(histVxVy,histVxVy_old,cv2.HISTCMP_CORREL)
        dist_ChiSquare[index] = cv2.compareHist(histVxVy,histVxVy_old,cv2.HISTCMP_CHISQR)
        dist_Intersection[index] = cv2.compareHist(histVxVy,histVxVy_old,cv2.HISTCMP_INTERSECT)
        dist_Bhattacharyya[index] = cv2.compareHist(histVxVy,histVxVy_old,cv2.HISTCMP_BHATTACHARYYA)
        dist_Hellinger[index] = cv2.compareHist(histVxVy,histVxVy_old,cv2.HISTCMP_HELLINGER)
        dist_ChiSquareAlt[index] = cv2.compareHist(histVxVy,histVxVy_old,cv2.HISTCMP_CHISQR_ALT)
        dist_KLDiv[index] = cv2.compareHist(histVxVy,histVxVy_old,cv2.HISTCMP_KL_DIV)
    # distance entre l'hisogramme de vitesse actuelle et l'histogramme de
    # vitesse d'une scene completement statique
    dist_Correl_static[index] = cv2.compareHist(histVxVy,histStatic,cv2.HISTCMP_CORREL)
    dist_ChiSquare_static[index] = cv2.compareHist(histVxVy,histStatic,cv2.HISTCMP_CHISQR)
    dist_Intersection_static[index] = cv2.compareHist(histVxVy,histStatic,cv2.HISTCMP_INTERSECT)
    dist_Bhattacharyya_static[index] = cv2.compareHist(histVxVy,histStatic,cv2.HISTCMP_BHATTACHARYYA)
    dist_Hellinger_static[index] = cv2.compareHist(histVxVy,histStatic,cv2.HISTCMP_HELLINGER)
    dist_ChiSquareAlt_static[index] = cv2.compareHist(histVxVy,histStatic,cv2.HISTCMP_CHISQR_ALT)
    dist_KLDiv_static[index] = cv2.compareHist(histVxVy,histStatic,cv2.HISTCMP_KL_DIV)

    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    result = np.vstack((frame2,bgr))
    cv2.imshow('Image et Champ de vitesses (Farneback)',result)
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite(imgDir+'Frame_%04d.png'%index,frame2)
        cv2.imwrite(imgDir+'OF_hsv_%04d.png'%index,bgr)
        cv2.imwrite(imgDir+'zoom_normalizedHistVxVy%04d.png' % index, normalizedHistVxVy*255)
        cv2.imwrite(imgDir+'zoom_sqrtHistVxVy%04d.png' % index, sqrtHistVxVy*255)
    elif k == ord('p'):
        while k != ord('g'):
            time.sleep(1)
            k = cv2.waitKey(30) & 0xff
    prvs_frame = next_frame
    histVxVy_old = histVxVy
    ret, frame2 = cap.read()
    index += 1
    if (ret):
        next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 


cv2.destroyAllWindows()
cap.release()

if cutGroundTruth != None:
    for idx in cutGroundTruth:
        plt.axvline(x=idx, color='k')
plt.plot(X,VxMax,label = "VxMax")
plt.plot(X,VxMin,label = "VxMin")
plt.plot(X,VyMax,label = "VyMax")
plt.plot(X,VyMin,label = "VyMin")
plt.legend()
plt.show()

renorm_dist_Correl = 1-dist_Correl/dist_Correl.max()
renorm_dist_ChiSquare = dist_ChiSquare/dist_ChiSquare.max()
renorm_dist_Intersection = 1-dist_Intersection/dist_Intersection.max()
renorm_dist_Bhattacharyya = dist_Bhattacharyya/dist_Bhattacharyya.max()
renorm_dist_Hellinger = dist_Hellinger/dist_Hellinger.max()
renorm_dist_ChiSquareAlt = dist_ChiSquareAlt/dist_ChiSquareAlt.max()
renorm_dist_KLDiv = dist_KLDiv/dist_KLDiv.max()

renorm_dist_Correl_static = 1-dist_Correl_static/dist_Correl_static.max()
renorm_dist_ChiSquare_static = dist_ChiSquare_static/dist_ChiSquare_static.max()
renorm_dist_Intersection_static = 1-dist_Intersection_static/dist_Intersection_static.max()
renorm_dist_Bhattacharyya_static = dist_Bhattacharyya_static/dist_Bhattacharyya_static.max()
renorm_dist_Hellinger_static = dist_Hellinger_static/dist_Hellinger_static.max()
renorm_dist_ChiSquareAlt_static = dist_ChiSquareAlt_static/dist_ChiSquareAlt_static.max()
renorm_dist_KLDiv_static = dist_KLDiv_static/dist_KLDiv_static.max()

if cutGroundTruth != None:
    for idx in cutGroundTruth:
        plt.axvline(x=idx, color='k')
plt.plot(X,renorm_dist_Correl, label = "Correlation")
plt.plot(X,renorm_dist_ChiSquare, label = "ChiSquare")
plt.plot(X,renorm_dist_Intersection, label = "Intersection")
plt.plot(X,renorm_dist_Bhattacharyya, label = "Bhattacharyya")
plt.plot(X,renorm_dist_Hellinger, label = "Hellinger")
plt.plot(X,renorm_dist_ChiSquareAlt, label = "ChiSquareAlt")
plt.plot(X,renorm_dist_KLDiv, label = "KLDiv")
plt.legend()
plt.title("All possible distances in compareHist")
plt.show()

if cutGroundTruth != None:
    for idx in cutGroundTruth:
        plt.axvline(x=idx, color='k')
plt.plot(X,renorm_dist_Correl_static, label = "Correlation")
plt.plot(X,renorm_dist_ChiSquare_static, label = "ChiSquare")
plt.plot(X,renorm_dist_Intersection_static, label = "Intersection")
plt.plot(X,renorm_dist_Bhattacharyya_static, label = "Bhattacharyya")
plt.plot(X,renorm_dist_Hellinger_static, label = "Hellinger")
plt.plot(X,renorm_dist_ChiSquareAlt_static, label = "ChiSquareAlt")
plt.plot(X,renorm_dist_KLDiv_static, label = "KLDiv")
plt.legend()
plt.title("All possible distances in compareHist")
plt.show()


valp = np.linalg.eigvalsh(covV)

if cutGroundTruth != None:
    for idx in cutGroundTruth:
        plt.axvline(x=idx, color='k')
plt.plot(X, meanVx, label = "meanVx")
plt.plot(X, meanVy, label = "meanVy")
plt.legend()
plt.show()
if cutGroundTruth != None:
    for idx in cutGroundTruth:
        plt.axvline(x=idx, color='k')
plt.plot(X,covV[:,0,0], label = "covVxVx")
plt.plot(X,covV[:,0,1], label = "covVxVy")
plt.plot(X,covV[:,1,0], label = "covVyVx")
plt.plot(X,covV[:,1,1], label = "covVyVy")
plt.legend()
plt.show()
if cutGroundTruth != None:
    for idx in cutGroundTruth:
        plt.axvline(x=idx, color='k')
plt.plot(X, valp[:,0], 'b')
plt.plot(X, valp[:,1], 'r')
plt.title("valeur propre de la covariance de V pour chaque frame")
plt.show()





