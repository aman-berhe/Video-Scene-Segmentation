"""
Input: subtitle file
        start=
        end=endof(subtitle)
functions--> get subtitlext (start,end)
         --> get subSentences (start, end, remove=otional)
                --> remove=binary: to remove texts that are not speech
         --> alignSub(bySeconds)
                --> to forward or backward the subtitle if -bySeconds:
                fasting the subtile other wise delay subtitlext
        -->

outputs-->
"""
import pysrt
from string import punctuation
import re
#import nlpPreprocessing

Got='/people/berhe/Bureau/TLP_thesis/subtitles/GoT/English'
BB='/people/berhe/Bureau/TLP_thesis/subtitles/BB/English'
HP='/people/berhe/Bureau/TLP_thesis/subtitles/HarryPotter/English'

class Subtitle:

    def __init__(self,tvs,season,episode):
        s='0'+str(season) if season<10 else str(season)
        e='0'+str(episode) if season<10 else str(episode)
        if tvs=='g' or tvs=='GOT' or tvs=='got':
            self.subFile=Got+'/GameOfThrones.Season'+str(s)+'.Episode'+str(e)+'.en.srt'
        if tvs=='b' or tvs=='BB' or tvs=='bb':
            self.subFile=BB+'/BreakingBad.Season'+str(s)+'.Episode'+str(e)+'.en.srt'
        #self.subFile=subFile
        self.subtitleTexts=[]
        self.start=[]
        self.end=[]
        self.subDir=""

    def readSub(self):
        """
            Reads the subtitle file (.srt) and returns all the texts and their starting and endind time as lists
        """
        self.subs=pysrt.open(self.subFile)
        for i in self.subs:
            self.start.append(((i.start.hours*60)+i.start.seconds+(i.start.milliseconds/1000)))
            self.end.append(((i.end.hours*60)+i.end.seconds+(i.end.milliseconds/1000)))
            self.subtitleTexts.append(i.texts)
        return self.start,self.end,self.subtitleTexts

    def getsubSentences(self, startTime,endTime):
        """
        Takes the subtitle file with a starting and ending time (in seconds) and returns part of the title text in the interval
        """
        self.subs=pysrt.open(self.subFile)
        texts=self.subs.slice(starts_after={'minutes':(int(startTime/60)),'seconds':(startTime%60)},ends_before={'minutes':(int(endTime/60)),'seconds':(endTime%60)})

        #self.sentences=nlpPreprocessing.sentenceSplit(texts.text)
        if (len(texts.text)==0):
            return ''
        else:
            text=texts.text.translate(str.maketrans('', '', punctuation))
            return text
    
    def to_min_sec(self,st_ms,end_ms):
        Ssec=st_ms/1000
        Esec=end_ms/1000
        sm,ss=divmod(Ssec,60)
        em,es=divmod(Esec,60)
        return sm,ss,em,es
    
    def to_min_sec2(st_ms,end_ms):
        Ssec=st_ms
        Esec=end_ms
        sm,ss=divmod(Ssec,60)
        em,es=divmod(Esec,60)
        return sm,ss,em,es
        
    
    def clean_str(self,string):
        #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        #string = re.sub(r"\'s", "\'s", string)
        #string = re.sub(r"\'ve", "\'ve", string)
        #string = re.sub(r"n\'t", " not", string)
        #string = re.sub(r"\'re", "\'re", string)
        #string = re.sub(r"\'d", "\'d", string)
        #string = re.sub(r"\'ll", " will", string)
        #string = re.sub(r",", ",", string)
        #string = re.sub(r"!", "!", string)
        string=re.sub(r'[\(\)]',' ',string)
        #string = re.sub("([\(\[]).*?([\)\]])", "", string)
        #string = re.sub(r")", r"", string)
        #string = re.sub(r"\?", " \? ", string)
        #string = re.sub(r"\s{2,}", " ", string)
        string=re.sub(r"\-",'',string)
        return string.strip().lower()
        
    def getTextByTime(self,start, end):
        sm,ss,em,es=self.to_min_sec(start,end)
        subs=pysrt.open(self.subFile)
        texts=subs.slice(starts_after={'minutes':sm,'seconds':ss},ends_before={'minutes':em,'seconds':es})
        return self.clean_str(texts.text)