
import numpy as np
import cv2
import sys
import math
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,euclidean
from sklearn import cluster
import time
import utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())
import tensorflow
import Segmantation_lib as sg
from tqdm import tqdm_notebook as tqdm
import random
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import applications
import pandas as pd
import Segmantation_lib as sg
from tqdm import tqdm_notebook as tqdm
import random


"""
We can choose diffrent models pretrained networks in Keras using the GetModel(name)
"""
videoFile='/people/berhe/Bureau/video/GameOfThrones.Season01.Episode01.mkv'

def getModel(modelName='VGG16'):
    if modelName=='VGG16':
        return applications.vgg16.VGG16(weights='imagenet',include_top=False,pooling='avg')
    if modelName=='VGG19':
        return applications.vgg19.VGG19(weights='imagenet',include_top=False,pooling='avg')
    if modelName=='Xception':
        return applications.xception.Xception(weights='imagenet',include_top=False,pooling='avg')
    if modelName=='InceptionV3':
        return applications.inception_v3.InceptionV3(weights='imagenet',include_top=False,pooling='avg')
    if modelName=='MobileNet':
        return applications.mobilenet.MobileNet(weights='imagenet',include_top=False,pooling='avg')
    if modelName=='ResNet50':
        return applications.resnet50.ResNet50(weights='imagenet',include_top=False,pooling='avg')



"""
get the VGG16 model weights to learn features for the time being
"""
model=VGG16(weights='imagenet',include_top=False)
"""
uncomment the line below to see summary of the model
"""
#print(model.summary())

"""
to save the frames of a video as images in a folder
"""
def saveFrames(videoFile):
    frameRate=cap.get(5)
    featuresList=[]
    while (cap.isOpened()):
        frameId=cap.get(1)
        ret,frame=cap.read()
        if ret!=True:
            break
        if (frameId % math.floor(frameRate)==0):
            fileName='/home/berhe/Desktop/TV-series/video/Frames_01/'+str(int(frameId))+'.jpg'
            cv2.imwrite(fileName,frame)
            sys.stdout.write('Extracting frame number %i'%frameId)
            sys.stdout.flush()
    cap.release()

def getMean(tempFeatures):
    print(len(tempFeatures))
    meanFeature=np.mean(tempFeatures,axis=0)
    #print(tempFeatures[0].shape)
    return meanFeature

def getframes(frameNumber,videoFile='/people/berhe/Bureau/video/GameOfThrones.Season01.Episode01.mkv'):
    framesFeatures=[]
    cap=cv2.VideoCapture(videoFile)
    frameRate=cap.get(5)
    fps=cap.get(cv2.CAP_PROP_FPS)
    timeStamp=[cap.get(cv2.CAP_PROP_POS_MSEC)]
    frameIds=[]
    while(cap.isOpened()):
        frameId=cap.get(1)
        ret,frame=cap.read()
        if ret!=True:
            break
        #if(frameId%math.floor(frameNumber)==0):
        timeStamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        framesFeatures.append(getFeatures(frame,model))
        frameIds.append(frameId)
        sys.stdout.write('\r%i'%frameId)
        sys.stdout.flush()
    cap.release()
    print('\n Done!')
    return framesFeatures,timeStamp,frameIds

def getFrames_byTimeStamp(shotStart,shotEnd,videoFile):
    frames=[]
    cap=cv2.VideoCapture(videoFile)
    #frameRate=cap.get(5)
    #fps=cap.get(cv2.CAP_PROP_FPS)
    timeStamp=cap.get(cv2.CAP_PROP_POS_MSEC)
    frameIds=[]
    shotStart=shotStart*1000
    shotEnd=shotEnd*1000
    while(cap.isOpened()):
        frameId=cap.get(1)
        ret,frame=cap.read()
        #print(frame.shape)
        if ret!=True:
            break
        timeStamp=cap.get(cv2.CAP_PROP_POS_MSEC)
        if(timeStamp>=shotStart):
            if (timeStamp<shotEnd):
                frames.append(frame)
                sys.stdout.write('\r%i'%timeStamp)
                sys.stdout.flush()
            else:
                break
        #frameIds.append(frameId)  
    cap.release()
    return frames



def getFeatures(frame,model):
    #change image or frame to array representation
    """
    if the first argument is a file of image add the following code and continue
    frame=image.load_img(path+frame,target_size(224,224))
    """
    #print(frame.shape)
    frame_img=cv2.resize(frame,(224,224))
    #frame_img=frame_img.reshape(3,224,224)
    frame_img=image.img_to_array(frame_img)
    frame_data=np.expand_dims(frame_img,axis=0)
    frame_data=preprocess_input(frame_data)
    #sys.stdout.write('.')
    #sys.stdout.flush()
    frame_features=model.predict(frame_data)[0]
    frame_features=np.array(frame_features)
    frame_features=frame_features.flatten()

    return frame_features
"""
Extract feature directly from the video frames with out saving them and it returns the features extracted usign
the VGG16 mode of imagenet data. we have a list of frame features.

Frame featuress can be used to construct similarity matrix example using cosine similairy.
"""
def getFrameFeatures(videoFile,model):
    cap=cv2.VideoCapture(videoFile)
    fps=cap.get(cv2.CAP_PROP_FPS)
    frameRate=cap.get(5)
    print(frameRate)
    timeStamp=[cap.get(cv2.CAP_PROP_POS_MSEC)]
    featuresList=[]
    while (cap.isOpened()):
        frameId=cap.get(1)
        ret,frame=cap.read()
        timeStamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        if ret!=True:
            break
        else:
            sys.stdout.write('\rExtracting frame number %i'%frameId)
            sys.stdout.flush()
            frame_features=getFeatures(frame,model)
            featuresList.append(frame_features)
    cap.release()
    print()
    print('Extracting Features of frames Finshed!!')
    return featuresList,timeStamp
"""
Get the shot frame features
"""
def shotFeatures(videoFile,shotsFile):
    shotStart,shotEnd=sg.getShots(shotFile=shotsFile)
    shotFeaturesList=[]
    tempFrames=[]
    with tqdm(total=len(shotEnd),file=sys.stdout) as pbar:
        for i in range(len(shotEnd)):
            tempFrames=getFrames_byTimeStamp(shotStart[i],shotEnd[i],videoFile)
            #print(shotStart[i],shotEnd[i],len(tempFrames))
            if len(tempFrames)>3:
                random.shuffle(tempFrames)
                tempFrames=tempFrames[0:3]#random.sample(range(len(tempFrames)),3)
            for f in range(len(tempFrames)):
                shotFeaturesList.append(getFeatures(tempFrames[f],model))
            #print(len(shotFeaturesList))
            pbar.update(1)
        print('\n Done!')
    return shotFeaturesList

def frameFeatures_Shots(videoFile,shotsFile):
    cap=cv2.VideoCapture(videoFile)
    fps=cap.get(cv2.CAP_PROP_FPS)
    timeStamp=[cap.get(cv2.CAP_PROP_POS_MSEC)]
    frameRate=cap.get(5)
    avgdFeatures=[]
    tempFeatures=[]
    shotStart,shotEnd=sg.getShots(shotFile=shotsFile)
    idx=0
    while (cap.isOpened()):
        frameId=cap.get(1)
        ret,frame=cap.read()
        timeSt=cap.get(cv2.CAP_PROP_POS_MSEC)
        timeStamp.append(timeSt)
        if ret!=True:
            break
        else:
            if (timeSt/1000)<shotEnd[idx]:
                tempFeatures.append(getFeatures(frame,model))
            else:
                if len(tempFeatures)!=0:
                    meanFeature=np.mean(tempFeatures,axis=0)
                    avgdFeatures.append(meanFeature)
                    idx=idx+1
                if idx == len(shotEnd):
                    tempFeatures=[]
                    idx=idx+1
                    break
        sys.stdout.write('\r Extracting frame number %i'%frameId)
        sys.stdout.flush()
    cap.release()
    print('\nDone!')
    return avgdFeatures, shotEnd,timeStamp

"""
similairy matrix: takes the array of features and compute pairwise similairt of the features
the features of the same frame has similarity score 1. So the diagonal of the return array is always 1
"""
def getSimilarityMatrix(featuresArr,distance):
        return np.matrix((1-pairwise_distances(featuresArr,metric=distance)))


"""
Clustering the frames using the similarity matrix
inputs: n_clusters-->number of clusters
        clusterAlgo-->clustering algorithm to try
        simMatrix-->The similairt matrix computed
output: Cluster labels of each frame
"""
def getClusters(n_Clusters,clusterAlgo,simMatrix):
    if clusterAlgo=="specteral" or clusterAlgo=="sp":
        return cluster.SpectralClustering(n_Clusters,affinity='precomputed').fit_predict(simMatrix)
    if clusterAlgo=="affinity" or clusterAlgo=="af":
        return cluster.AffinityPropagation(affinity='precomputed').fit_predict(simMatrix)
    if clusterAlgo=='KMeans' or clusterAlgo=='km':
        return cluster.KMeans(n_clusters=n_Clusters, init='k-means++', max_iter=100, n_init=1).fit_predict(simMatrix)

    return "No cluster Labels"

"""
Takes the time stamp of each frame and the manually segmented scene boundries of an episode
It returns the cluster labels as sequence of frame cluster labels.
For evaluation puposes of the frame segmentation based on clustering neighbouring frames
"""

def frameGroundClusters(episodeNumber,frameTimeStamp):
    referencecluster=[]
    idx=0
    Df=pd.read_csv('/people/berhe/Bureau/TLP_thesis/Scenes/all_scenes.csv')
    episodeTime=Df.query('Episode==1')['end_time']
    episodeTime=[i for i in episodeTime]
    for i in frameTimeStamp:
        try:
            if (i*1000)<=episodeTime[idx]:
                referencecluster.append(idx)
            else:
                idx=idx+1
                referencecluster.append(idx)
        except IndexError:
            idx=idx+1
            leng=len(frameTimeStamp)-len(referencecluster)
            for j in range(1,leng):
                referencecluster.append(idx)
            break
    return referencecluster

#if __name__=='__main__':
#    
#    framesFeatures,timeStamp,frameIds=getframes(1000,videoFile)
#    framesFeatures=np.array(framesFeatures)
#    SM=getSimilarityMatrix(framesFeatures,'cosine')
      