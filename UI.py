# -*- coding: utf-8 -*-
"""
Created on Sun Apr  16 22:50:57 2023

@author: Tarun
"""

from flask import Flask
from flask import render_template
from flask import request
import onp_test as ot
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/post_data',methods=['POST'])
def post_data():
    title=request.form["title"]
    sentence=request.form["sentence"]
    num_hrefs=request.form["hrefs"]
    num_imgs=request.form["images"]
    num_videos=request.form["videos"]
    data_channel=request.form.get("data-channel")
    weekday=request.form.get("weekday")
    print(data_channel)
    print(weekday)
    result=ot.start_testing(title,sentence,num_hrefs,num_imgs,num_videos,data_channel,weekday)
    print( result)
    shares=result[0]
    review=result[1]    
    return render_template("Result.html",shares=shares,review=review)
    
if __name__ =="__main__":
    app.run()
