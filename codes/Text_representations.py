from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from textblob import TextBlob
import nlpPreprocessing as nlp
import os
import pysrt
import numpy as np


book1='/people/berhe/Bureau/TLP_thesis/Descriptions/AClashOfKings.txt'
book2='/people/berhe/Bureau/TLP_thesis/Descriptions/AGameOfThrones.txt'
book3='/people/berhe/Bureau/TLP_thesis/Descriptions/ADanceWithDragons.txt'
book4='/people/berhe/Bureau/TLP_thesis/Descriptions/AFeastForCrows.txt'

bookList=[book1,book2,book3,book4]

class TextEmbeddings:
    def __init__(self):
        self.fileName=book2
        self.Got='/people/berhe/Bureau/TLP_thesis/subtitles/GoT/English/'
        self.BBT='/people/berhe/Bureau/TLP_thesis/subtitles/BBT/English/'
        self.HP='/people/berhe/Bureau/TLP_thesis/subtitles/HarryPotter/English/'
        self.BB='/people/berhe/Bureau/TLP_thesis/subtitles/BB/English/'


    def allBooks_w2V(self,num_books):
        sentences=[]
        for i in range(num_books):
            f=open(bookList[i],'r')
            text=f.read()#.decode('utf-8')
            txtblb=TextBlob(text)
            for sent in txtblb.sentences:
                sentences.append(sent.split())
        model=Word2Vec(sentences=sentences,min_count=1,size=300)
        X=model[model.wv.vocab]
        #X is word vectors
        words=list(model.wv.vocab)
        model.save("word2vec_GoT.model")
        return model,sentences,words
    """
    w2V from a book.the input is the directory for the book and returns W2V model sentences list and words
    """
    def book_W2V(self,fi=book2):
        #self.fileName=fileName

        f=open(fi,'r')
        text=f.read()#.decode('utf-8')
        sentences=[]
        txtblb=TextBlob(text)
        for sent in txtblb.sentences:
            sentences.append(sent.split())
        model=Word2Vec(sentences=sentences,min_count=1,size=300)
        #X=model[model.wv.vocab]
        #X is word vectors
        words=list(model.wv.vocab)
        model.save("word2vec_GoT.model")
        return model,sentences,words
    """
    It takes returns the sentences of the subtitle, and w2V embedding using all the subtitles of the choosen tv series
    """
    def subtiles_W2V(self):
        letter=input('Enter the beginning letter of the tv-series \n Game of Thrones(GoT), \n Bing bang theory(BBT), \n Harry Potter(HP) \n Breaking bad (bb)')

        if letter == 'G' or letter == 'g':
            subDir=self.Got
        elif letter == 'B' or letter == 'b':
            subDir=self.BBT
        elif letter == 'H' or letter == 'h':
            subDir=self.HP
        elif letter=='bb':
            subDir=self.BB
            
        text=""
        for fl in os.listdir(subDir):
            if '.en.srt' in fl:
                print(fl)
                file=subDir+fl
                subs=pysrt.open(file,encoding='iso-8859-1')
                for s in subs:
                    text=text+s.text
        sentences=[]
        txtblb=TextBlob(text)
        for sent in txtblb.sentences:
            sentences.append(sent.split())

        model=Word2Vec(sentences=sentences,min_count=1,size=300)
        X=model[model.wv.vocab]
        #X is word vectors
        words=list(model.wv.vocab)
        model.save("word2vec_"+str(letter)+".model")
        return model,sentences,words

    """
    This line plots the embedding of words
    """
    def plotW2V(model):
        X=model[model.wv.vocab]
        pca=PCA(n_components=2)
        result=pca.fit_transform(X)
        pyplot.scatter(result[:, 0],result[:, 1])
        for i, word in enumerate(words):
            pyplot.annotate(word, xy=(result[i,0], result[i,1]))
        pyplot.show()

    """
    Getting avaerage vectors of of sentences for sentence embedding puposes.
    """
    def avg_feature_vector(self,sentence, model, num_features, index2word_set):
        words = sentence.split()
        feature_vec = np.zeros((num_features, ), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec
