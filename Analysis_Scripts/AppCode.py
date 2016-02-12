# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:40:02 2016

Functions for App Code

Goals:
Take in input features from website
Plot a static histogram of the sessons for the task along with Threshold for evaluation

Plot an estimate of those features on the same plot as the histogram and classification 

These functions were used in the preparation of the pickle data variables for the website but not actively by the website itself.


@author: jgodlove
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import chdir
from os.path import isfile
from sklearn import linear_model
chdir('/home/jgodlove/gitrepo/Insight/ConstantTherapy')

def plot_hist(past_users,prediction):
    #Plots the histogram and prediction     
    fig = plt.figure()
    plt.hist(past_users,np.array(range(0,12)).astype(float)+.5)
    axis = plt.axis()
    plt.axis([.5,11.5,axis[2],axis[3]])
    plt.vlines(prediction,axis[2],axis[3],linewidth=3)
    plt.xticks(range(1,12))
    return fig

def load_parameters(domain,target_task):
    #loads data files needed for the page
    model = pickle.load(open('reg_model_%s_%i.pkl'%(domain,target_task),'rb'))
    past_users = np.array(pickle.load(open('past_users_%s_%i.pkl'%(domain,target_task),'rb')))
    user_params = pickle.load(open('default_params_%s_%i.pkl'%(domain,target_task),'rb'))
    return past_users,user_params,model

def define_default_parameters(domain,target_task):
    #sets median user parameters for the inital load of the page
    features = np.array(pickle.load(open('reg_features_%s_%i.pkl'%(domain,target_task),'rb')))
    
    features = np.concatenate((features[:,:target_task*3],features[:,-29:]),axis=1) #Removes tasks after the task performance
    features = features[:,:-28]
    features[features == 0] = np.nan
    user_params = np.nanmedian(features,axis=0)
    
    user_params = np.concatenate((user_params,np.array([0]*28))) #Sets disorders, deficits, and therapy types to 0
    user_params[-1] = 0 #default Gender: male
    user_params[-2] = 0 #default Condintion Since: <6M
    user_params[-3] = 4 #default Age: 50-70    
    pickle.dump(user_params,open('default_params_%s_%i.pkl'%(domain,target_task),'wb'))
    return user_params

def define_past_users(domain,target_task):
    #loads in previous users from the features pickle and sets it 
    features = np.array(pickle.load(open('reg_features_%s_%i.pkl'%(domain,target_task),'rb')))
    past_users = features[:,target_task*3]
    past_users[past_users>11] = 11    
    pickle.dump(past_users,open('past_users_%s_%i.pkl'%(domain,target_task),'wb'))
    return past_users    


domain = 'Arithmetic'
target_task = 14
threshold = 2.9


#define_past_users(domain,target_task)
#define_default_parameters(domain,target_task)
past_users,user_params,model = load_parameters(domain,target_task)

prediction = model.predict(user_params.reshape(1,-1))
if prediction < 1:
    prediction = 1
elif prediction > 11:
    prediction = 11
    
plot_hist(past_users,prediction)


print 'Predicted Sessions to Complete: %.1f'%prediction
if prediction < threshold:
    print 'User will find this task Easy'
else:
    print 'User will find this task Hard'
    
