##############################################################################
#                       Méthode 2 : distance maximale
##############################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import auxFunctions as af

filename = 'Extrait1-Cosmos_Laundromat1(340p).m4v'
#filename = 'Rotation_OY(Pan).m4v'
#directory = '../TP2_Videos_Exemples/'
directory = './TP2_Videos/'
#imgDir = '../figure/'
imgDir = './Images/'
cutGroundTruth = af.getCutGroundTruth(filename) 
nbImages = af.getNbFrame(filename) # nombre de frame de la video

cap = cv2.VideoCapture(directory+filename)
ret, frame = cap.read()

nVal = 256
type_dist = cv2.HISTCMP_CORREL
calcul_dist = 0 # 0 si on cherche la distance minimale, 1 si on cherche la distance maximale

histUV = np.zeros((nVal, nVal))
#Sauvegarde de l'histogramme precedent pour pouvoir calculer les distances
#entre histogrammes sans moyenne glissante
histUV_old = np.zeros((nVal, nVal))

dists = np.zeros(nbImages -1)


# 1. Calcul des distances entre deux images

index = 0
while(ret):
    #Transformation de l'image pour la lire en Yuv.
    frame_Yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #Affichage en BGR de l'image traitee
    #cv2.imshow('Images', frame)

    #Calcul de l'histogramme de l'image Yuv
    histUV = cv2.calcHist([frame_Yuv], [1, 2], None, [256, 256], [0, 256, 0, 256])

    if index>0:
        #Calcul de la distance entre deux histogrammes successifs
        dists[index-1] = cv2.compareHist(histUV, histUV_old, type_dist)
    
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
    
    index += 1
    histUV_old = histUV

    #Lecture de l'image suivante
    ret, frame = cap.read()
    if not(ret):
        print("The end")


# 2. Ensuite, on calcule l'index de l'image ayant la distance maximale (ou minimale
# selon la distance utilisee).

n = len(cutGroundTruth)
key_img=np.zeros(n)
for i in range(n):
    if i==0:
        start = 0
    else:
        start = cutGroundTruth[i-1]
    stop = cutGroundTruth[i]
    
    if calcul_dist ==0: # si on cherche la plus petite distance possible, on 
        #initialise dist_max a la distance maximale atteinte
        dist_max = max(dists)
    elif calcul_dist == 1: # si on cherche la plus grande distance possible,
        #on initialise dist_max a la distance minimale
        dist_max = 0
    index_max = start
    for j in range(start, stop-1):
        if calcul_dist == 0:
            if min(dist_max,dists[j]) == dists[j]:
                dist_max = dists[j]
                index_max = j
        elif calcul_dist == 1:
            if max(dist_max,dists[j]) == dists[j]:
                dist_max = dists[j]
                index_max = j
    key_img[i] = index_max
print(key_img)


#3. Affichage et sauvegarde des images clefs correspondantes

cap.release()
cap = cv2.VideoCapture(directory+filename)
ret, frame = cap.read()
n = len(key_img)

index = 0
key_index = 0
while ret :
    if key_index <n and index == key_img[key_index]:
        cv2.imshow('Images clefs', frame)
        cv2.imwrite(imgDir+'imgclef_distmax/image%04d.png' % index, frame)
        key_index+=1
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('image%04d.png' % index, frame)
    elif k == ord('p'):    #Pause de la video
        while k != ord('g'): #Reprise de la video
            time.sleep(1)
            k = cv2.waitKey(30) & 0xff

    index += 1
    #Lecture de l'image suivante
    ret, frame = cap.read()
    if not(ret):
        print("The end")

cap.release()
cv2.destroyAllWindows()