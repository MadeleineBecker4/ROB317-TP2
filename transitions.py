#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:14:50 2020

Ce programme a pour but de nous aider a determiner les vraies positions des
transitions dans la video
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

filename = 'Extrait1-Cosmos_Laundromat1(340p).m4v'
#filename = 'Rotation_OY(Pan).m4v'
directory = '../TP2_Videos_Exemples/'
#directory = './TP2_Videos/'
#imgDir = '../figure/'
imgDir = './Images/'



nbFrames = 50

cap = cv2.VideoCapture(directory+filename)

ret, frame = cap.read()
buffFrames = np.zeros((nbFrames, frame.shape[0], frame.shape[1], frame.shape[2]), dtype=np.uint8)

index = 0
while(ret):
    #Affichage en BGR de l'image courante
    cv2.imshow('Images', frame)
    print(index)

    #Sauvegarde de l'image courante pour pouvoir revenir en arriere si necessaire
    buffFrames[index % nbFrames] = frame

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('image%04d.png' % index, frame)
    elif k == ord('p'):    #Pause de la video
        i_courant = index
        while k != ord('g'): #Reprise de la video
            if k==ord('b') and i_courant>index-nbFrames: #Affichage de l'image precedente
                i_courant -= 1
                cv2.imshow('Images', buffFrames[i_courant%nbFrames])
                print(i_courant)
            if k==ord('n') and i_courant<index: #Affichage de l'image suivante
                i_courant+=1
                cv2.imshow('Images', buffFrames[i_courant%nbFrames])
                print(i_courant)
            time.sleep(0.1)
            k = cv2.waitKey(30) & 0xff

    time.sleep(0.05)
    #Mise a jour de l'index
    index += 1

    #Lecture de l'image suivante
    ret, frame = cap.read()
    if not(ret):
        print("The end")

cv2.destroyAllWindows()
cap.release()
