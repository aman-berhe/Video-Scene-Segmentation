import os
import json
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import networkx as nx
from nltk import texttiling
import nltk
from matplotlib import pylab
from pyannote.core import Segment, Timeline
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from pyannote.algorithms.segmentation.sliding_window import SegmentationGaussianDivergence
from sklearn.metrics import recall_score,precision_score,f1_score
#from sklearn.metrics import precision_score, \recall_score, confusion_matrix, classification_report, \accuracy_score, f1_score

segmenter = SegmentationGaussianDivergence(duration=20, step=1)
shotdir='Desktop/TrecVid/5082189274976367100.shots.json'
threads='Desktop/TrecVid/5082189274976367100.threads.json'
manualScen='Desktop/TrecVid/Eastender_manual_segmentation_inSeconds.json'

"""
This function preprocess the shot threading format to prepare it to segmentation. It takes the .json format produced by pyannote-video by herve
"""
def preprocessingThreads (threads=threads):

    """
    with open(manualScen, 'r') as f:
        data = json.load(f)

    manualBoundry=[]
    for i in data['Segment']:
        try:
            manualBoundry.append(i['End time'])
        except:
            continue
"""
    with open(threads, 'r') as f:
        data1 = json.load(f)

    shotSeq=[]
    shotBondry=[]
    for x in data1['content']:
        try:
            shotSeq.append(x['label'])
            shotBondry.append(x['segment']['end'])
            #print (x['label']+":"),
        except:
            continue

    return  shotSeq, shotBondry, data1

"""
Get shot timing: start tile and end time from a json file of shots generated from pyannote
"""

def getShots(shotFile):
    with open(shotFile,'r') as f:
        data=json.load(f)
    shotStart=[]
    shotEnd=[]
    for i in data['content']:
        shotEnd.append(i['end'])
        shotStart.append(i['start'])
    return shotStart,shotEnd

"""
Preprocess the manually annotated data of Estenders: or any file that have a sentence with its start time. The scene segmentation is done manually.
It returns:
    SpeakerSequence: Sequence of speakers in the Estenders 00 test file
    Speakers Dataframe: includes the starting and ending time of each sentence
    textSent: the texts of sentences
    lemtizedTexts: the lemma of the sentences with stop words removed using english stop words: this is important for texttiling and c99 algorithms for text segmentation
"""
def preprocessAnnotationFile(fileName):

	SpeakerDF=pd.read_csv(fileName)
	SpeakerDF=SpeakerDF[['Sentence','Speaker','Duration end XML']]
	SpeakerDF['Duration end XML']=[float(x.replace(",", "")) for x in SpeakerDF['Duration end XML']]
	SpeakerDF=SpeakerDF[SpeakerDF.Speaker != '_']
	SpeakerDF=SpeakerDF[SpeakerDF.Speaker != 'ACT']
	SpeakerDF=SpeakerDF[SpeakerDF.Speaker !='nan']
	SpeakerDF.dropna()

	SpeakerSequence=list(SpeakerDF.Speaker)

	textSent=[]
	SpeakerDF=SpeakerDF.reset_index(drop=True)

	for j in range(len(SpeakerDF)):
		textSent.append(SpeakerDF.Sentence[j])

	lematizedText=[]
	lemmatizer = WordNetLemmatizer()
	for i in textSent:
		tokens = nltk.word_tokenize(i)
		words = [word for word in tokens if word.isalpha()]
		stop_words = set(stopwords.words('english'))
		words = [w for w in words if not w in stop_words]

		lemmaTmp=[]
		for w in words:
			lemmas=lemmatizer.lemmatize(w)
			lemmaTmp.append(lemmas)



		lematizedText.append(lemmaTmp)

	return SpeakerSequence,SpeakerDF, textSent,lematizedText


def sequenceSegmentation(speakerSequence,k,C):
    tempList=[]
    sepPosition=[]
    count=0
    tempList.append(speakerSequence[0:2])
    #print(tempList)
    for i in range(2,len(speakerSequence)):
        if speakerSequence[i]==tempList[-1]:
            count=0
            continue
        elif speakerSequence[i] in tempList:
            tempList.pop(0)
            tempList.append(speakerSequence[i])
            count=0
        else:
            count=count+1
            if len(tempList)<k:
                tempList.append(speakerSequence[i])
            if count>C:
                tempList=tempList[C-1:]
                sepPosition.append(i-C)
                count=0
    return sepPosition
    
def seqSegUsingSet(threads,window):
    sepPos=[]
    intValue=[]
    for i in range(window+1,(len(threads))):
        #print(threads[i-window-1:i-1],threads[i:i+window])
        prevWind=set(threads[i-window-1:i-1])
        nextWind=set(threads[i:i+window])
        intUnion=(len(prevWind.intersection(nextWind))/len(prevWind.union(nextWind)))
        intValue.append(intUnion)
        if intUnion<0.3:
            sepPos.append(i)
        else:
            continue
    c=0
    listscene=[]
    for i in sepPos:
        if (i-c)!=1:
            listscene.append(i)
            c=i
    #listscene.append(threads[c:])
    return listscene,intValue

def shotTruthValue(data,sceneBoundry):
    truthValue=""
    pos=0
    i=0

    for x in data['content']:
        try:
            if x['segment']['end']<=sceneBoundry[pos]:
                truthValue=truthValue+'0'
                i=i+1
            else:
                truthValue=truthValue+'1'
                i=i+1
                if pos <len(sceneBoundry):
                    pos=pos+1
                else:
                    break
        except:
            print('exception')
            break
    return truthValue,i

#Truth Value of the each senannotationtence of the annaotation file
def shotTruthValueNew(DF,scenesBoundry):
	truthValue=[]
	pos=0
	i=0
    #print('hello')
    # sceneSpeakerBoundry=[]
    # speakerBoundry=list(DF['Duration end XML'])
    # for i in scenes:
    # 	sceneSpeakerBoundry.append(speakerBoundry[i])
	for x in range(len(DF)):
		try:
			if DF['Duration end XML'][x]<scenesBoundry[pos]:
				truthValue.append(0)
				i=i+1
			else:
				truthValue.append(1)
				if pos <len(scenesBoundry):
					pos=pos+1
				else:
					print('else')
		except:
			print('exceprtion')
			break

	return truthValue,x

# Evaliuation techniques Pk Values measure
def pk(ref, hyp, k=None, boundary='1'):

    if k is None:
        #print(ref.count(boundary))
        k = int(round(len(ref) / (ref.count(boundary) * 2.)))

    err = 0
    for i in range(len(ref)-k +1):
        r =ref[i:i+k].count(boundary) >  0
        h = hyp[i:i+k].count(boundary) > 0
        #print(ref.count(boundary),hyp.count(boundary),boundary)
        #print (h)
        #print (r)
        if r != h:
           err += 1
    return err / (len(ref)-k +1.)

#Evaluation technique windowsDiff: it takes the segments
def windowdiff(seg1, seg2, k, boundary="1"):

    if k is None:
        #print(ref.count(boundary))
        k = int(round(len(seg1) / (seg1.count(boundary) * 2.)))

    #if len(seg1) != len(seg2):
       # raise ValueError("Segmentations have unequal length")
    wd = 0
    for i in range(len(seg1) - k):
        wd += abs((seg1[i:i+k+1].count(boundary)) - (seg2[i:i+k+1].count(boundary)))
    return (wd/(float(len(seg1)-k)))
#computing recall
def precision( ref, hyp):
    success =0
    fail = 0
    length=0
    if len(ref)<len(hyp):
    	length=len(ref)
    else:
    	length=len(hyp)

    for i in range(length):
        if hyp[i] == '1':
            if ref[i]== '1':
                success += 1
            else:
                fail +=1

    return (success/(success+fail))

def recall(X_test, y_test):
	tp =0
	fp = 0
	for i in range(len(y_test)):
	    if y_test[i] == 1:
	        if X_test[i] == 1:
	            tp += 1
	        else:
	            fp +=1

	tn =0
	fn = 0
	for i in range(len(y_test)):
	    if y_test[i] == 0:
	        if X_test[i] == 0:
	            tn += 1
	        else:
	            fn +=1

	return (tp/(tp+fn))

#computing the K vamlues for the evaluation techniques
def computeK(sequence, scences):
	k=len(sequence)/(2*len(scences))
	return k

#Plotting the Results of the metrics
def plotResults(k_values,pk_result,wd_result):
	plt.plot(k_values,pk_result)
	plt.plot(k_values,wd_result)
	plt.legend(['y= pk Results'],'y= WindowsDiff Results',loc='upper left')
	plt.show()

#constraucting and drwaing a graph of coversations which is weighted graph
def draw_graph(graph,edge_weights, labels=None, graph_layout='shell',
               node_size=1800, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='green', edge_alpha=0.3,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # create networkx graph
    G=nx.Graph()

    # add edges
    i=0
    for edge in graph:
        G.add_edge(edge[0], edge[1],weigt=edge_weights[i])
        i=i+1

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_weights,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font)

    if labels is None:
        labels = range(len(graph))

    edge_labels = dict(zip(graph, labels))
    #nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels,
                                 #label_pos=edge_text_pos)

    # show graph
    plt.show()
    return G

"""rules to identify who is speaking to whom ased on the Papers written by:
    Extraction and Analysis of Dynamic Conversational Networks from from TV Series by Xavier Bost et al.

    Based on the rules mensioned on their paper to identify who is talking to whom, it returns a wieghted graph that shows their interaction
"""
def verbalInreaction(speakerSequence):
    graph=[]
    startSpeaker=speakerSequence[0]
    secondSpeaker=speakerSequence[1]
    tempSequence=speakerSequence[:5]
    graph.append((startSpeaker,secondSpeaker))
    for i in range(2,len(speakerSequence)):
        if i <(len(speakerSequence)-2):
            if speakerSequence[i]==startSpeaker:
                graph.append((secondSpeaker,startSpeaker))
                startSpeaker=secondSpeaker
                secondSpeaker=speakerSequence[i-1]
                tempSequence.pop(0)
                tempSequence.append(speakerSequence[i+2])
            elif speakerSequence[i] in tempSequence[:2]:
                graph.append((speakerSequence[i],speakerSequence[i-1]))
                startSpeaker=secondSpeaker
                secondSpeaker=speakerSequence[i]
                tempSequence.pop(0)
                tempSequence.append(speakerSequence[i+2])
            elif speakerSequence[i] in tempSequence[3:5]:
                graph.append((speakerSequence[i],speakerSequence[i+1]))
                startSpeaker=secondSpeaker
                secondSpeaker=speakerSequence[i]
                tempSequence.pop(0)
                tempSequence.append(speakerSequence[i+2])
            else:
                graph.append((speakerSequence[i-1],speakerSequence[i]))
                startSpeaker=secondSpeaker
                secondSpeaker=speakerSequence[i]
                tempSequence.pop(0)
                tempSequence.append(speakerSequence[i+2])
        else:
            graph.append((speakerSequence[i-1],speakerSequence[i]))
    return graph


"""
plots the boundries of segments. It shows the boundry of manually segmented and automatically segmented scenes
"""
def plotBoundries(referenceBoundries, hypothesisBoundries, start=0,end=0):
    end=len(referenceBoundries)
    for segment in hypothesisBoundries[start:end]:
        plt.plot([segment, segment], [-10, -0.5], 'r')
    for segment in referenceBoundries[start:end]:
        plt.plot([segment, segment], [0.5, 10], 'g')

    plt.ylim(-11, 11);
    plt.xlim(0, segment);
    plt.xlabel('Time (seconds)');


	#def metricResults(ref, hyp):
