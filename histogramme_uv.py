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
#imgDir = '../figure/'
imgDir = './Images/'


nVal = 256
nbImages = 3168
nbFrames = 50
seuil=0.65

histUV = np.zeros((nVal, nVal))
#Sauvegarde de l'histogramme precedent pour pouvoir calculer les distances
#entre histogrammes sans moyenne glissante
histUV_old = np.zeros((nVal, nVal))
#Sauvegarde de plusieurs histogrammes precedents pour pouvoir calculer la moyenne glissante
buffHist = np.zeros((nbFrames, nVal, nVal), dtype=np.float32)
cap = cv2.VideoCapture(directory+filename)

dist_Correl = np.zeros(nbImages)
# dist_ChiSquare = np.zeros(nbImages)
# dist_Intersection = np.zeros(nbImages)
# dist_Bhattacharyya = np.zeros(nbImages)
# dist_Hellinger = np.zeros(nbImages)
# dist_ChiSquareAlt = np.zeros(nbImages)
# dist_KLDiv = np.zeros(nbImages)
X = np.arange(nbImages)

ret, frame = cap.read()

index = 0
ret = True
while(ret):

    #Transformation de l'image pour la lire en Yuv.
    frame_Yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #Affichage en BGR de l'image traitee
    cv2.imshow('Images', frame)

    #Calcul de l'histogramme de l'image Yuv
    histUV = cv2.calcHist([frame_Yuv], [1, 2], None, [
                          256, 256], [0, 256, 0, 256])

    #Sauvegarde de l'histogramme pour pouvoir faire la moyenne glissante
    buffHist[index % nbFrames] = histUV

    #Modification de l'histogramme pour pouvoir l'agrandir avant de l'afficher,
    #pour qu'il soit plus grand et donc plus lisible a l'oeil nu.
    histShape = np.shape(histUV)
    HumanVisibleHistogrammeUV = np.zeros((histShape[0]*3, histShape[1]*3))
    HumanVisibleHistogrammeUV[0::3, 0::3] = histUV
    HumanVisibleHistogrammeUV[0::3, 1::3] = histUV
    HumanVisibleHistogrammeUV[0::3, 2::3] = histUV
    HumanVisibleHistogrammeUV[1::3, 0::3] = histUV
    HumanVisibleHistogrammeUV[1::3, 1::3] = histUV
    HumanVisibleHistogrammeUV[1::3, 2::3] = histUV
    HumanVisibleHistogrammeUV[2::3, 0::3] = histUV
    HumanVisibleHistogrammeUV[2::3, 1::3] = histUV
    HumanVisibleHistogrammeUV[2::3, 2::3] = histUV
    # cv2.imshow('histogramme',histUV/(histUV.max()-histUV.min()))
    cv2.imshow('Histogram visible by human', HumanVisibleHistogrammeUV /
               (HumanVisibleHistogrammeUV.max()-HumanVisibleHistogrammeUV.min()))

    #Pour pouvoir calculer la moyenne glissante, il faut que suffisamment 
    #d'histogrammes aient ete calcules
    if index > nbFrames:

        #Calcul de la moyenne des histogrammes precedents et suivants la transition
        #etudiee. On etudie la transition au milieu des histogrammes sauvegardes.
        #Pour eviter de devoir decaler les histogrammes a chaque pas de temps, 
        #on les sauvegarde a la position index%nbFrames. Cela necessite une petite
        #manipulation au moment du calcul des moyennes.
        histPrev = np.zeros_like(histUV)
        histNext = np.zeros_like(histUV)
        N = nbFrames//2
        offset = index%nbFrames +1
        for j in range(N):
            histPrev = histPrev + buffHist[(offset+j)%nbFrames]
            histNext = histNext + buffHist[(offset+N+j)%nbFrames]

        #Calcul de la distance entre les deux moyennes
        dist_Correl[index] = cv2.compareHist(histPrev, histNext, cv2.HISTCMP_CORREL)

        #Calcul de la distance entre deux histogrammes successifs
        #dist_Correl[index] = cv2.compareHist(histUV, histUV_old, cv2.HISTCMP_CORREL)

        #Si les images sont considerees comme des transitions, elles sont enregistrees,
        #et les distances sont affichees dans la console
        if dist_Correl[index]<seuil:
            cv2.imwrite(imgDir+'image%04d.png' % index, frame)
            print(dist_Correl[index])

    # Calul des distances entre deux histogrammes successifs pour toutes les distances 
    # disponibles dans la fonction cv2.compareHist
    # if index > 0:
    #     dist_Correl[index] = cv2.compareHist(histUV, histUV_old, cv2.HISTCMP_CORREL)
    #     dist_ChiSquare[index] = cv2.compareHist(
    #         histUV, histUV_old, cv2.HISTCMP_CHISQR)
    #     dist_Intersection[index] = cv2.compareHist(
    #         histUV, histUV_old, cv2.HISTCMP_INTERSECT)
    #     dist_Bhattacharyya[index] = cv2.compareHist(
    #         histUV, histUV_old, cv2.HISTCMP_BHATTACHARYYA)
    #     dist_Hellinger[index] = cv2.compareHist(
    #         histUV, histUV_old, cv2.HISTCMP_HELLINGER)
    #     dist_ChiSquareAlt[index] = cv2.compareHist(
    #         histUV, histUV_old, cv2.HISTCMP_CHISQR_ALT)
    #     dist_KLDiv[index] = cv2.compareHist(
    #         histUV, histUV_old, cv2.HISTCMP_KL_DIV)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('image%04d.png' % index, frame)
        cv2.imwrite('hist%04d.png' % index, histUV)
    elif k == ord('p'):    #Pause de la video
        while k != ord('g'): #Reprise de la video
            time.sleep(1)
            k = cv2.waitKey(30) & 0xff

    #Mise a jour des parametres. On n'a pas besoin de mettre a jour buffHist
    #de part la facon dont les histogrammes sont sauvegardes
    index += 1
    histUV_old = histUV

    #Lecture de l'image suivante
    ret, frame = cap.read()
    if not(ret):
        print("The end")


# Les distances ne sont pas du tout normalisees, donc pour les comparer on les normalise 
# entre 0 histgrammes identiques et 1 distance maximale entre deux histogrammes de la video

# renorm_dist_Correl = 1-dist_Correl/dist_Correl.max()
# renorm_dist_ChiSquare = dist_ChiSquare/dist_ChiSquare.max()
# renorm_dist_Intersection = 1-dist_Intersection/dist_Intersection.max()
# renorm_dist_Bhattacharyya = dist_Bhattacharyya/dist_Bhattacharyya.max()
# renorm_dist_Hellinger = dist_Hellinger/dist_Hellinger.max()
# renorm_dist_ChiSquareAlt = dist_ChiSquareAlt/dist_ChiSquareAlt.max()
# renorm_dist_KLDiv = dist_KLDiv/dist_KLDiv.max()

# '''
# renorm_moy_dist = (renorm_dist_Correl + \
#                    renorm_dist_ChiSquare + \
#                    renorm_dist_Intersection + \ 
#                    renorm_dist_Bhattacharyya + \
#                    renorm_dist_Hellinger + \
#                    renorm_dist_ChiSquareAlt + \
#                    renorm_dist_KLDiv)/7
# '''
# renorm_moy_dist = (renorm_dist_Correl + renorm_dist_ChiSquare + renorm_dist_Intersection +
#                    renorm_dist_Bhattacharyya + renorm_dist_Hellinger + renorm_dist_ChiSquareAlt + renorm_dist_KLDiv)/7


# Affichage des distances entre histogrammes pour toutes les distances disponibles 
# dans la fonction cv2.compareHist

# plt.plot(X, renorm_dist_Correl, label="Correlation")
# plt.plot(X, renorm_dist_ChiSquare, label="ChiSquare")
# #plt.plot(X,renorm_dist_Intersection, label = "Intersection")
# #plt.plot(X,renorm_dist_Bhattacharyya, label = "Bhattacharyya")
# #plt.plot(X,renorm_dist_Hellinger, label = "Hellinger")
# #plt.plot(X,renorm_dist_ChiSquareAlt, label = "ChiSquareAlt")
# plt.plot(X, renorm_dist_KLDiv, label="KLDiv")
# plt.legend()
# plt.title("All possible distances in compareHist")
# plt.show()

# plt.plot(X, renorm_moy_dist)
# plt.title("moy dist")
# plt.show()

plt.plot(X,dist_Correl)
plt.title("Correlation des histogrammes")
plt.show()

cv2.destroyAllWindows()
cap.release()
