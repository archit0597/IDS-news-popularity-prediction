# -*- coding: utf-8 -*-
"""
Created on Sun Apr  16 22:50:57 2023

@author: Tarun
"""
import pandas as pd
from Filters import WordCount
from Filters import WordPolarity
from Filters import StopWords
# from Filters import FileLoader

#Objects of the filters
word_count = WordCount()
stop_words = StopWords()
word_polarity = WordPolarity()

class Attributes:
    n_tokens_title=0
    title="News Article"
    num_hrefs = 0
    num_imgs  = 0
    num_videos =0
    data_channel_is_lifestyle = 0
    data_channel_is_entertainment = 0
    data_channel_is_bus = 0
    data_channel_is_socmed = 0
    data_channel_is_tech = 0
    data_channel_is_world = 0
    weekday_is_monday = 0
    weekday_is_tuesday = 0
    weekday_is_wednesday = 0
    weekday_is_thursday = 0
    weekday_is_friday = 0
    weekday_is_saturday = 0
    weekday_is_sunday = 0
    is_weekend = 0
    n_tokens_content = 0
    n_unique_tokens = 0
    n_non_stop_words = 0
    n_non_stop_unique_tokens = 0
    average_token_length = 0
    global_rate_positive_words = 0
    global_rate_negative_words = 0
    rate_positive_words = 0
    rate_negative_words =0
    
    def set_depend_attributes(self):
        self.n_tokens_content = word_count.count_words(self.sentence)
        self.n_tokens_title=word_count.count_words(self.title)
        self.n_unique_tokens = word_count.count_uniq_words(self.sentence)
        self.n_non_stop_words = stop_words.get_non_stopwords_count(self.sentence)
        self.n_non_stop_unique_tokens = stop_words.get_uniq_nonstopwords_count(self.sentence)
        
        self.average_token_length =word_count.total_length_words(self.sentence)/self.n_tokens_content
       
        self.pwc = word_polarity.get_pos_word_count(self.sentence)
        self.nwc = word_polarity.get_neg_word_count(self.sentence)
        self.twc = word_count.count_words(self.sentence)
        if(self.twc==0 ):
                self.twc=1
        elif(self.pwc==0):
            self.pwc=1
        elif( self.nwc==0):
            self.nwc=1
        self.global_rate_positive_words = self.pwc/self.twc
        self.global_rate_negative_words =  self.nwc/self.twc
        self.rate_positive_words = self.pwc/(self.pwc+self.nwc)
        self.rate_negative_words = self.nwc/(self.pwc+self.nwc)
        
    def set_user_details(self,title,sentence,num_hrefs,num_imgs,num_videos,data_channel,weekday):
        self.title=title
        self.sentence = sentence
        self.num_hrefs = num_hrefs
        self.num_imgs = num_imgs
        self.num_videos = num_videos
        if(data_channel=='lifestyle'):
            self.data_channel_is_lifestyle=1
        elif(data_channel=='entertainment'):
            self.data_channel_is_entertainment=1
        elif(data_channel=='bus'):
            self.data_channel_is_bus=1
        elif(data_channel=='socmed'):
            self.data_channel_is_socmed=1
        elif(data_channel=='tech'):
            self.data_channel_is_tech=1
        else:
            self.data_channel_is_world=1
        if(weekday=='monday'):
            self.weekday_is_monday=1
        elif(weekday=='muesday'):
            self.weekday_is_tuesday=1
        elif(weekday=='wednesday'):
            self.weekday_is_wednesday=1
        elif(weekday=='thursday'):
            self.weekday_is_thursday=1
        elif(weekday=='friday'):
            self.weekday_is_friday=1
        elif(weekday=='saturday'):
            self.weekday_is_saturday=1
            self.is_weekend=1
        else:
            self.weekday_is_sunday=1
            self.is_weekend=1
        self.set_depend_attributes()
        return self.get_article_test_data()
    
    def get_article_test_data(self):
        list2 =[]
        list=["n_tokens_title","n_tokens_content","n_unique_tokens","n_non_stop_words","n_non_stop_unique_tokens","num_hrefs","num_imgs","num_videos","average_token_length","data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus","data_channel_is_socmed","data_channel_is_tech","data_channel_is_world","weekday_is_monday","weekday_is_tuesday","weekday_is_wednesday","weekday_is_thursday","weekday_is_friday","weekday_is_saturday","weekday_is_sunday","is_weekend","global_rate_positive_words","global_rate_negative_words","rate_positive_words","rate_negative_words"]
        list2=[[self.n_tokens_title,self.n_tokens_content,self.n_unique_tokens,self.n_non_stop_words,self.n_non_stop_unique_tokens,self.num_hrefs,self.num_imgs,self.num_videos,self.average_token_length,self.data_channel_is_lifestyle,self.data_channel_is_entertainment,self.data_channel_is_bus,self.data_channel_is_socmed,self.data_channel_is_tech,self.data_channel_is_world,self.weekday_is_monday,self.weekday_is_tuesday,self.weekday_is_wednesday,self.weekday_is_thursday,self.weekday_is_friday,self.weekday_is_saturday,self.weekday_is_sunday,self.is_weekend,self.global_rate_positive_words,self.global_rate_positive_words,self.rate_positive_words,self.rate_negative_words]]
        test = pd.DataFrame( list2,columns=list )   
        return test                
        
