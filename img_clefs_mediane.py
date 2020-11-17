##############################################################################
#                         Méthode 1 : image médiane
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

#Calcul des index des images clefs
n = len(cutGroundTruth)
key_img=np.zeros(n)
for i in range (n):
    if i==0:
        start = 0
    else:
        start = cutGroundTruth[i-1]
    stop = cutGroundTruth[i]
    key_img[i] = start + (stop-start)//2
print(key_img)

#Affichage et sauvegarde des images clefs correspondantes
index = 0
key_index = 0
n = len(key_img)
while ret :
    if key_index < n and index == key_img[key_index]:
        cv2.imshow('Images clefs', frame)
        cv2.imwrite(imgDir+'imgclef_mediane/image%04d.png' % index, frame)
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