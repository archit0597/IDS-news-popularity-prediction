# -*- coding: utf-8 -*-
"""
Created on Sun Apr  16 22:50:57 2023

@author: tarun
"""
import pickle
import statistics
from  Driver import Attributes
attributes=Attributes()
"""Vote Classifier"""

class VoteClassifier:
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, x):
        votes = []
        for c in self._classifiers:
            v = c.predict(x)
            votes.append(v[0])
            
        print(votes)
        n=statistics.mode(votes)
        return n
    
    def predict(self,x):
        shares = []
        for c in self._classifiers:
            shares.append(c.predict(x))
        n=(sum(shares)/len(shares))
        return round(float(n[0]))
    
    
"""Predicting the classification algorithms"""
 
#decision_tree_classifier
 
open_file = open("./pickled_algos/dectreeclassifier.pickle","rb")
dec_classifier_pkl = pickle.load(open_file)
open_file.close()

#random_forest_classifier

open_file = open("./pickled_algos/ranforestclassifier.pickle","rb")
ran_classifier_pkl = pickle.load(open_file)
open_file.close()


#ANN_classifier

open_file = open("./pickled_algos/annclassifier.pickle","rb")
ann_classifier_pkl = pickle.load(open_file)
open_file.close()

"""Predicting the Regression algorithms"""

#Linear_regression

open_file = open("./pickled_algos/linearregression.pickle","rb")
regressor_pkl = pickle.load(open_file)
open_file.close()


#Decision Tree Regression
open_file = open("./pickled_algos/dectreeregression.pickle","rb")
dec_regressor_pkl = pickle.load(open_file)
open_file.close()


#Random Forest Regression
open_file = open("./pickled_algos/randomForestregression.pickle","rb")
ran_regressor_pkl = pickle.load(open_file)
open_file.close()

#ANN Regression
open_file = open("./pickled_algos/annregression.pickle","rb")
ann_regressor_pkl = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(
                                  dec_classifier_pkl,
                                  ran_classifier_pkl,
                                  #ann_classifier_pkl,
                                  )

share_predictor = VoteClassifier(
                                 regressor_pkl,
                                 #dec_regressor_pkl,
                                 #ran_regressor_pkl,
                                 #ann_regressor_pkl
                                 )

def predict_popularity(x):
    if voted_classifier.classify(x) == 0:
        return "Your article should be improved to get popular:("
    else:
        return "Congrats Your article is all set to get popular:)"
            
def predict_popularity_shares(shares):
    if shares > 5000:
        return "Congrats Your article is all set to get popular:)"
    else:
        return "Your article should be improved to get popular:("
     

def predict_shares(x):
    return share_predictor.predict(x)

def start_testing(title,sentence,num_hrefs,num_imgs,num_videos,data_channel,weekday):
        data_test=attributes.set_user_details(title,sentence,num_hrefs,num_imgs,num_videos,data_channel,weekday)
        print(type(data_test))
        return [predict_shares(data_test),predict_popularity_shares(predict_shares(data_test))]




