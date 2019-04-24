# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 16:17:12 2019

@author: berhe
"""
import Segmantation_lib as sg
import evaluationMetrics as em
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import Text_representations as tr
import Subtitle as sb
import nlpPreprocessing as nlp
import pickle
import random
from scipy.spatial.distance import cosine,euclidean
from sklearn import cluster
from sklearn.metrics import pairwise_distances
import evaluationMetrics as em
import os
from gensim.models import Word2Vec

class TVSSegmentation:
    
    dirFeaturesBB='/vol/work3/berhe/ExtractedFeatures/BB/'
    dirFeaturesGoT='/vol/work3/berhe/ExtractedFeatures/GoT/'
    subtlDir='/people/berhe/Bureau/TLP_thesis/subtitles/'
    GoT=False
    BB=False
    saveClstResult='resultsFile_'
    tvs=""
    season=1
    episode=1
    Season="Season0"+str(season)
    Episode=("Episode"+str(episode)) if int(episode) >= 10 else ("Episode0"+str(episode))
    fileName=Season+'.'+Episode
    subDir=dirFeaturesGoT
    sub='/people/berhe/Bureau/TLP_thesis/subtitles/GoT/English/GameOfThrones.Season01.Episode01.en.srt'
    #letter=input('Enter  tv-series first letter \n Game of Thrones(G)\n Breaking bad (b)\n')
    #season,episode=input('chose the season and Episode\n').split(' ')
    
        #episode=input('chose the season and Episode\n').split(' ')
    #season,episode,letter= intVaribales()
    #print(letter,season,episode)
    def __init__ (self,tvs,season,episode):
        self.tvs=tvs
        self.season=season
        self.episode=episode
        if tvs == 'G' or tvs == 'g':
            self.subDir=self.dirFeaturesGoT
            self.subtlDir=self.subtlDir+'GoT/English/GameOfThrones.'
            self.saveClstResult=self.saveClstResult+'GoT_'
            GoT=True
        elif tvs == 'B' or tvs == 'b':
            self.subDir=self.dirFeaturesBB
            self.subtlDir=self.subtlDir+'BB/English/BreakingBad.'
            self.saveClstResult=self.saveClstResult+'BB_'
            GoT=False
        else:
             print('you did not choose a TV-Series! bye bye')
    
        self.Season="Season0"+str(season)
        self.Episode=("Episode"+str(episode)) if int(episode) >= 10 else ("Episode0"+str(episode))
        self.fileName=self.Season+'.'+self.Episode
        self.subtlFile=self.subtlDir+self.fileName+'.en.srt'
        self.saveClstResult=self.saveClstResult+self.Season+'Result.ixt'
        print('You have chosen :')
        if GoT==True:
            print('Tv-Series: Game of Thrones')
        else:
            print('Tv-Series: Breaking bad')
        print('Season: ',self.season)
        print('Episode:',self.episode)
        print('Subtile File : ',self.subtlFile)
    
        self.sub=sb.Subtitle(self.tvs,season,episode)
        #return season,episode,tvs,subtlDir
        
    #season,episode,letter=printDetails()
    #Season="Season0"+str(season)
    #Episode=("Episode"+str(episode)) if int(episode) > 10 else ("Episode0"+str(episode))
    #fileName=Season+'.'+Episode
    #subtlFile=subtlDir+fileName+'.en.srt'
    #saveClstResult=saveClstResult+Season+'Result.ixt'
    #sub=sb.Subtitle(subtlFile)
    def frameGroundClusters(self,shotTimeStamp):
        #letter=input('Enter the beginning letter of the tv-series \n Game of Thrones(GoT),\n Breaking bad (bb)')    
        if self.tvs == 'G' or self.tvs == 'g':
            dirGot='/people/berhe/Bureau/TLP_thesis/codes/AnnotatedScene/Got/season'+str(self.season)+'/Scenes/'
        elif self.tvs == 'B' or self.tvs == 'b':
            dirGot='/people/berhe/Bureau/TLP_thesis/codes/AnnotatedScene/BreakingBad/season'+str(self.season)+'/Scenes/'
        #episode=(str(episodeNumber)) if int(self.episode) > 10 else ("0"+str(self.episode))
        if(self.episode)<10:
            fileGT='S0'+str(self.season)+'E0'+str(self.episode)
        else:
            fileGT='S0'+str(self.season)+'E'+str(self.episode)
        referencecluster=[]
        idx=0
        Df=pd.read_csv(dirGot+fileGT, delimiter="\t")
        episodeTime=Df['end']
        #Df.query('Episode==1')['end_time']
        episodeTime=[i/1000 for i in episodeTime]
        for i in shotTimeStamp:
            try:
                if i<=episodeTime[idx]:
                    referencecluster.append(idx)
                else:
                    idx=idx+1
                    referencecluster.append(idx)
            except IndexError:
                idx=idx+1
                leng=len(shotTimeStamp)-len(referencecluster)
                for j in range(1,leng):
                    referencecluster.append(idx)
                break
        return referencecluster,episodeTime
        
    def getAvgFeatures(self,features,timeStamp,shotEnd):
        avgFetures=[]
        tempList=[]
        c=0
        i=0
        for j in range(len(shotEnd)):
            for i in range(c,len(timeStamp)-1):
                #print(timeStamp[i]<=shotEnd[indx])
                if (timeStamp[i]/1000)<=shotEnd[j]:
                    tempList.append(features[i])
                else:
                    break
            avgFetures.append(np.mean(tempList,axis=0))
            tempList=[]
            c=i
        return avgFetures
    
    def getRandomFeat(self,features,timeStamp,shotEnd):
        randSampFetures=[]
        uniSmplFeatures=[]
        tempList=[]
        rand_smpl=[]
        uni_Smpl=[]
        step=0
        c=0
        i=0
        for j in range(len(shotEnd)):
            for i in range(c,len(timeStamp)-1):
                #print(timeStamp[i]<=shotEnd[indx])
                if (timeStamp[i]/1000)<=shotEnd[j]:
                    tempList.append(features[i])
                else:
                    break
            if len(tempList)>=5:
                rand_smpl = [ tempList[i] for i in sorted(random.sample(range(len(tempList)), 5)) ]
                randSampFetures.append(rand_smpl)
                step=int(len(tempList)/5)
                if step!=1:
                    uni_Smpl = [ tempList[i*step] for i in range(5) ]
                    uniSmplFeatures.append(uni_Smpl)
                else:
                    uni_Smpl = tempList[0:5]
                    uniSmplFeatures.append(uni_Smpl)
            else:
                randSampFetures.append(rand_smpl)
                uniSmplFeatures.append(uni_Smpl)
            tempList=[]
            c=i
        return randSampFetures,uniSmplFeatures
    
    def getUniformSamples(self,features,timeStamp,shotEnd):
        uniSmplFeatures=[]
        tempList=[]
        uni_Smpl=[]
        c=0
        i=0
        for j in range(len(shotEnd)):
            for i in range(c,len(timeStamp)-1):
                #print(timeStamp[i]<=shotEnd[indx])
                if (timeStamp[i]/1000)<=shotEnd[j]:
                    tempList.append(features[i])
                else:
                    break
            if len(tempList)>=5:
                step=int(len(tempList)/5)
                uni_Smpl = [ tempList[i*step] for i in range(int(len(tempList)/step)) ]
                uniSmplFeatures.append(uni_Smpl)
            else:
                uniSmplFeatures.append(uni_Smpl)
            tempList=[]
            c=i
        return uniSmplFeatures
                
    
    def getShotTextFeat(self,shotEnd):
        TR=tr.TextEmbeddings()
        if self.tvs =='g' or self.tvs == 'G':
            #model,sentences,words=TR.allBooks_w2V(4)
            model = Word2Vec.load("word2vec_GoT.model")
        else:
            model = Word2Vec.load("word2vec_bb.model")
            #model,sentences,words=TR.subtiles_W2V()
        shot_TE=[]
        for i in range(0,len(shotEnd)):
            texts=self.sub.getsubSentences(shotEnd[i-1],shotEnd[i])
            NLP=nlp.Pre_Processing(texts)
            shottext=NLP.removeStopwords(texts)
            shotSentences=''
            for tex in shottext:
                shotSentences=shotSentences+' '+tex
            shot_TE.append(TR.avg_feature_vector(shotSentences,model,300,model.wv.index2word))
        return shot_TE
    
    def getClusters(self,n_Clusters,clusterAlgo,simMatrix):
        if clusterAlgo=="specteral" or clusterAlgo=="sp":
            return cluster.SpectralClustering(n_Clusters,affinity='precomputed').fit_predict(simMatrix)
        if clusterAlgo=="affinity" or clusterAlgo=="af":
            return cluster.AffinityPropagation(affinity='precomputed').fit_predict(simMatrix)
        if clusterAlgo=='KMeans' or clusterAlgo=='km':
            return cluster.KMeans(n_clusters=n_Clusters, init='k-means++', max_iter=100, n_init=1).fit_predict(simMatrix)
    
        return "No cluster Labels"
        
    def getSimilarityMatrix(self,featuresArr,distance):
            return np.matrix((1-pairwise_distances(featuresArr,metric=distance)))
    
    def saveClstrresults(self,dist_out,refreClust,saveFile=saveClstResult,f='a'):
        clstrAlgo=['KMeans','specteral','affinity']
        n_clst=40
        for clal in clstrAlgo:
            predictedsClusters=self.getClusters(n_clst,clal,dist_out)
            pc=np.array(predictedsClusters)
            rc=np.array(refreClust)
            purity=round(em.purity_score(pc,rc),4)
            NMI=round(em.nmi(pc,rc),4)
            cov=round(em.coverage(pc,rc),4)
            with open(saveFile, 'a') as f:
                #f.write('Algorithm \t Number of classes \t distance '+'\n')
                if f=='r':
                    f.write('Random Features\n')
                else:
                    f.write('Averag Features\n')
                f.write('-'*50+'\n')
                f.write(str(clal) +'\t\t' +str(n_clst) +'\t\t' + 'Cosine'+'\n')
                f.write('-'*50+'\n')
                f.write('purity' +'\t\t' +'NMI' +'\t\t' + 'cov'+'\n')
                f.write(str(purity) +'\t\t' +str(NMI) +'\t\t' + str(cov)+'\n')
                f.write('#'*50+'\n')
                
    def saveEvaluations(self,truthValueManual,truthValueAuto,rType='macro',boundry=1):
        winDiff=em.windowdiff(truthValueManual,truthValueAuto,boundary=boundry)
        pk=em.pk(truthValueManual,truthValueAuto,boundary=boundry)
        truthValueAuto=np.array(truthValueAuto)
        truthValueManual=np.array(truthValueManual)
        coverage=em.coverage(truthValueManual,truthValueAuto)
        purity=em.purity_score(truthValueManual,truthValueAuto)
        recall=em.recall(truthValueManual,truthValueAuto,rType=rType)
        precision=em.precision(truthValueManual,truthValueAuto,rType=rType)
        #print('winDiff,pk,truthValueAuto,coverage,purity,recall,precision ... Computed!')
        
        return [round(winDiff,3),round(pk,3),round(coverage,3),round(purity,3),round(recall,3),round(precision,3)]
        
    
    def load_Data(self):
        for fn in os.listdir(self.subDir):
            if self.fileName in fn:
                if fn.endswith('.npy'):
                    framesFeatures=np.load(self.subDir+fn)
                if fn.endswith('.pkl'):    
                    with open(self.subDir+fn, 'rb') as f:
                        timeStamp=pickle.load(f)
                if fn.endswith('.json'):
                    shot=self.subDir+fn
        return framesFeatures, timeStamp,shot
        
        
    
    """loading one of the Tv series BB or GoT"""
    def load_ManAnnotation(self):
    
        manAnnotation=""
        if self.tvs=='G' or self.tvs == 'g':
            manAnnotation='/people/berhe/Bureau/TLP_thesis/codes/AnnotatedScene/Got/'+'season'+str(self.season)+'/Scenes/'
        else:
            manAnnotation='/people/berhe/Bureau/TLP_thesis/codes/AnnotatedScene/BreakingBad/'+'season'+str(self.season)+'/Scenes/'
        if(self.episode)<10:
            annFileName='S0'+str(self.season)+'E0'+str(self.episode)
        else:
            annFileName='S0'+str(self.season)+'E'+str(self.episode)
        
        Df=pd.read_csv(manAnnotation+annFileName, delimiter="\t")
        sc_end=Df['end']
        
        return sc_end
    
    def load_ManAnnotation2(self):
    
        manAnnotation=""
        if self.tvs=='G' or self.tvs == 'g':
            manAnnotation='/people/berhe/Bureau/GoT_Data/manualScenes/'
        else:
            manAnnotation='/people/berhe/Bureau/TLP_thesis/codes/AnnotatedScene/BreakingBad/'+'season'+str(self.season)+'/Scenes/'
        if(self.episode)<10:
            annFileName='GoT_S0'+str(self.season)+'E0'+str(self.episode)+'.txt'
        else:
           annFileName='GoT_S'+str(self.season)+'E'+str(self.episode)+'.txt'
        
        Df=pd.read_csv(manAnnotation+annFileName,sep='\t')
        sc_end=Df['end']
        
        return sc_end
        
    def getTruthValues_Auto(self,sepPos,shotEnd):
        truthValueAuto=[]
        sceneTime_auto=[]
        truthValueAuto_bin=[]
        truthValueAuto_bin_old=[]
        indx=0
        for i in range(len(shotEnd)):
            if i==sepPos[indx]:
                truthValueAuto_bin.append(0)
                truthValueAuto.append(indx)
                truthValueAuto_bin_old.append(1)
                sceneTime_auto.append(shotEnd[i])
                if indx<(len(sepPos)-1):
                    indx=indx+1
            else:
                truthValueAuto.append(indx)
                truthValueAuto_bin.append(1)
                truthValueAuto_bin_old.append(0)
        return truthValueAuto,truthValueAuto_bin,truthValueAuto_bin_old,sceneTime_auto
    
    def getTruthValue_Man(self,shotEnd,shotMakar):
        truthValueManual=[]
        truthValueManual_bin=[]
        truthValueManual_bin_old=[]
        indx=0
        sceneTime_man_bin=[]
        ls=[]
        ls_bin=[]
        ls_bin_old=[]
        for i in range(len(shotEnd)):
            if shotEnd[i]>=shotMakar[indx]:
                sceneTime_man_bin.append(shotEnd[i])
                indx=indx+1
                truthValueManual_bin.append(0)
                truthValueManual.append(indx)
                truthValueManual_bin_old.append(1)
                if indx>=len(shotMakar):
                    ls=[indx]*(len(shotEnd)-len(truthValueManual))
                    ls_bin=[1]*(len(shotEnd)-len(truthValueManual_bin))
                    ls_bin_old=[0]*(len(shotEnd)-len(truthValueManual_bin_old))
                    break
                else:
                    continue
            else:
                truthValueManual_bin.append(1)
                truthValueManual_bin_old.append(0)
                truthValueManual.append(indx)
        truthValueManual=truthValueManual+ls
        truthValueManual_bin=truthValueManual_bin+ls_bin
        truthValueManual_bin_old=truthValueManual_bin_old+ls_bin_old
        
        return truthValueManual,truthValueManual_bin,truthValueManual_bin_old,sceneTime_man_bin
        
    """   
    subDir,season,episode,letter=chooseTV()
    Season="Season0"+str(season)
    Episode=("Episode"+str(episode)) if episode > 10 else ("Episode0"+str(episode))
    fileName=Season+'.'+Episode
    
    shotStart,shotEnd=sg.getShots(shot)
    shotMakar=load_ManAnnotation(letter,episode,season)
    refreClust=frameGroundClusters(episode,shotEnd)
    len(refreClust)
    if letter=='g'
        subFile='/people/berhe/Bureau/TLP_thesis/subtitles/GoT/English/GameOfThrones.'+fileName+'.en.srt'
    else:
        subFile='/people/berhe/Bureau/TLP_thesis/subtitles/BB/English/BreakingBad.'+fileName+'.en.srt'
    sub=sb.Subtitle(subFile)
    avgFetures=getAvgFeatures(framesFeatures)
    shot_TE=getShotTextFeat(sub)
    randSampFetures=getRandomFeat(framesFeatures)
    print(shot_TE.shape,avgFetures.shape,randSampFetures.shape)
    conctFeat_T=np.concatenate((avgFetures,shot_TE),axis=1)
    conctFeat_TT=np.concatenate((randSampFetures,shot_TE),axis=1)
    dist_concT=getSimilarityMatrix(conctFeat,'cosine')
    dist_concTT=getSimilarityMatrix(conctFeat1,'cosine') 
    #Here we adda concatination of features: But we can also have results with out concatination
    randSampFetures=np.array(randSampFetures)
    randSampFetures=randSampFetures.reshape((randSampFetures.shape[0],randSampFetures.shape[1]*randSampFetures.shape[2]))
    randSampFetures.shape,avgFetures.shape
    conctFeat=np.concatenate((avgFetures,shot_TE),axis=1)
    #shot_TE1=shot_TE.reshape((810,1,300))
    #shot_TE1.shape
    conctFeat1=np.concatenate((randSampFetures,shot_TE),axis=1)
    conctFeat.shape, conctFeat1.shape
    """