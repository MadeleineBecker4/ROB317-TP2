#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 18:37:40 2020

@author: iad
"""

def getCutGroundTruth (videoFilename):
    '''
    video file name est le nome du fichier de la video
    Parameters
    ----------
    videoFilename : str
                    The file name (including extension) without the path of the
                    video
    Returns
    -------
    CutGroundTruth : int array
                     indexes of cut. each index correspond to the first frame
                     just after the cut. The index of the first frame of the
                     video is not included 
    '''
    if videoFilename == 'Extrait1-Cosmos_Laundromat1(340p).m4v':
        CutGroundTruth = [250,479,511,600,653,691, 1114, 1181, 1310, 1415, 1517, 1565, 1712, 1781,1864, 1989, 2047,2166,2216,2278,2442,2512,2559,2637,2714,2765,2838,3020,3094,3131,3162]
        return CutGroundTruth
    return None

def getNbFrame (videoFilename):
    '''
    video file name est le nome du fichier de la video
    Parameters
    ----------
    videoFilename : str
                    The file name (including extension) without the path of the
                    video
    default : int 
                default value if filename not present
    Returns
    -------
    nbFrame : int
                number of frame of the video
    '''
    if videoFilename == 'Extrait1-Cosmos_Laundromat1(340p).m4v':
        nbFrame = 3168
        return nbFrame
    if videoFilename == 'Rotation_OX(Tilt).m4v':
        nbFrame = 135
        return nbFrame
    if videoFilename == 'Rotation_OY(Pan).m4v':
        nbFrame = 145
        return nbFrame
    if videoFilename == 'Rotation_OZ(Pan).m4v':
        nbFrame = 111
        return nbFrame
    if videoFilename == 'Travelling_OX.m4v':
        nbFrame = 198
        return nbFrame
    if videoFilename == 'Travelling_OZ.m4v':
        nbFrame = 157
        return nbFrame
    if videoFilename == 'ZOOM_O_TRAVELLING.m4v':
        nbFrame = 379
        return nbFrame
    return None