# -*- coding: utf-8 -*-
"""
Created on Sun Apr  16 22:50:57 2023

@author: Tarun
"""
 #nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
pos_words = []
neg_words = []

#Positive words
pos_words_file=open("./+ve.txt","r").read()
for word in pos_words_file.split('\n'):
    pos_words.append(word)

#Negative words
neg_words_file=open("./-ve.txt","r",errors='ignore').read()
for word in neg_words_file.split('\n'):
    neg_words.append(word)

class WordCount:
  def count_words(self,sentence):
    return len(sentence.strip().split())
  
  def count_uniq_words(self,sentence):
     return len(set(sentence.split()))
 
  def total_length_words(self,sentence):
      l=0
      for word in sentence.strip().split():
          l=l+len(word)
      return l
          
      
#rate non_stop and uniq_non_stop_words
class StopWords:
  filtered_sentence = list()
  def get_non_stopwords_count(self,sentence):
      stop_words = set(stopwords.words('english'))
      word_tokens = word_tokenize(sentence)
      self.filtered_sentence = [w for w in word_tokens if not w in stop_words]
      return len(self.filtered_sentence)
      
  def get_uniq_nonstopwords_count(self,sentence):
      return len(set(self.filtered_sentence))
  
class WordPolarity:
        
    def get_pos_word_count(self,sentence):
        # f1 = FileLoader()
        any_in = lambda a, b: any(i==b for i in a)
        pc = 0
        for word in word_tokenize(sentence):
            if(any_in(pos_words,word)):
                pc+=1
        return pc
    
    def get_neg_word_count(self,sentence):
        # f1 = FileLoader()
        any_in = lambda a, b: any(i==b for i in a)
        nc = 0
        for word in word_tokenize(sentence):
            if(any_in(neg_words,word)):
                nc+=1
        return nc