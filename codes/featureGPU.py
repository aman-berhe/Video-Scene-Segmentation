import ExtractingFrameFeatures as eff
import cv2
import numpy as np
import math
import pickle
from sklearn.preprocessing import normalize
import sys
import pandas as pd
import os

videoFiles='/vol/work3/maurice/dvd_extracted/GameOfThrones/'
saveFiles='/vol/work3/berhe/ExtractedFeatures/GoT/'
episodeFile=[3,4,9,10,6,9,9,10]
seasonList=[2,2,2,2,3,3,4,4]
"""
videofilesFiles=[]
for fl in os.listdir(videoFiles):
    if '.mkv' in fl:
        videofilesFiles.append(fl)
videofilesFiles.sort()        
for file in videofilesFiles:
    framesFeatures,timeStamp,frameIds=eff.getframes(1,videoFiles+file)
    framesFeatures=np.array(framesFeatures)
    np.save(saveFiles+file+'_features.npy',framesFeatures)
    with open(saveFiles+file+'_frameTimeStamp.pkl', 'wb') as f:
        pickle.dump(timeStamp,f)
"""
for i in range(len(episodeFile)):
    if episodeFile[i]<10:
        file='GameOfThrones.Season0'+str(seasonList[i])+'.Episode0'+str(episodeFile[i])+'.mkv'
    else:
        file='GameOfThrones.Season0'+str(seasonList[i])+'.Episode'+str(episodeFile[i])+'.mkv'
    print(file)
    framesFeatures,timeStamp,frameIds=eff.getframes(1,videoFiles+file)
    framesFeatures=np.array(framesFeatures)
    np.save(saveFiles+file+'_features.npy',framesFeatures)
    with open(saveFiles+file+'_frameTimeStamp.pkl', 'wb') as f:
        pickle.dump(timeStamp,f)