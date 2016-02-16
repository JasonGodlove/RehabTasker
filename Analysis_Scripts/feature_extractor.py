# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:23:31 2016

Loads up features pkl, runs regression on it, and plots the data for a particular task


@author: jgodlove
"""


import mysql.connector
from mysql.connector import errorcode
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress 
from os import chdir
from os.path import isfile
from time import sleep
import scipy.stats as stats
import sklearn.ensemble as skRF
from sklearn import linear_model
from sklearn.preprocessing import normalize
from sklearn.feature_selection import RFECV
import sklearn.cross_validation as skcv
import sklearn.metrics as metrics
import seaborn as sns
chdir('/home/jgodlove/gitrepo/Insight/ConstantTherapy')

def drange(start, stop, step):
     r = start
     while r < stop:
     	yield r
     	r += step

def cross_val_feature(features,output,regr):
    #Does Cross Validation on each of the figures, calculating the average values of the feature parameters and returns the mean value of the parameters to determine feature importance
    parameters = []
    loo = skcv.LeaveOneOut(len(output))
    for i,test in loo:
        regr.fit(features[i,:],output[i])
        parameters.append(regr.coef_)
    parameters=np.array(parameters)
    rank = np.mean(parameters,axis=0)
    return rank
#    order = np.argsort(abs(rank))    
#    for i in order:
#    print '%s :%.2f'%(f_name[i],rank[i])
#end of cross_val_feature

def generate_feature_name(num_task):
    #generates a list of display names for the features
    feature_name=[]
    for i in range(num_task):#####################################
        feature_name.append('Task %i'%i)
        feature_name.append('Acc %i'%i)
        feature_name.append('Lat %i'%i)
    feature_name.append('Avg Session per Day')
    for i in range(1,10):
        feature_name.append('Deficit %i'%i)
    for i in range(1,8):
        feature_name.append('Disorder %i'%i)
    for i in range(1,10):
        feature_name.append('Therapy Type %i'%i)
    feature_name.append('Age')
    feature_name.append('Condition')
    feature_name.append('Gender')
    return feature_name
#End generate_feature_name

def plot_regression(predicted,output):
    #Plots the predicted vs Observed in a box plot and scatter plot
    plt.figure()
    # Plot outputs
    #plt.subplot(121)
    c = sns.color_palette('RdYlGn', 12)#"RdYlGn",'Spectral'
    c.reverse() #so red is on far right    
    df = pd.DataFrame({'observed':output,'predicted':predicted})
    sns.boxplot(x='observed',y='predicted',data=df,palette=c,fliersize=0,linewidth=2)
    
    sns.stripplot(x='observed',y='predicted',data=df,jitter=True,color='k',size=10,alpha=.5)
    #plt.scatter(output-1+np.random.uniform(-.1,.1,len(output)), predicted,  facecolor='k',s=60,linewidths=1,edgecolor='k',alpha=.5) #wierd alignment issue with output being +1
    plt.xticks(range(0,11),range(1,12),fontsize=24)
    plt.yticks(range(1,12),fontsize=24) 
    #plt.xlabel('')
    #plt.ylabel('')
    
    plt.xlabel('Observed # Sessions',fontsize=30)
    plt.ylabel('Predicted # Sessions',fontsize=30)
    plt.title('Linear Regression for Task %i'%target_task)
#end of plott_regression

def generate_model(features,output):
    #Generates the model for the task
    #Noise cleaning
    output[output>11]=11
    
    regr = linear_model.BayesianRidge(normalize=True)
    
    regr.fit(features,output)
    
    
    predicted = skcv.cross_val_predict(regr,f,output,cv=len(output)) #Use leave one out cross validation to get prediction values of the output 
    
    predicted[predicted<1] = 1 #Cleaning the output of the data to make logical sense and impose limits.
    predicted[predicted>11]=11
    return regr, predicted





#Setting up task parameters
domain = 'Arithmetic'      
target_task = 14 #division lvl 2   
num_task = 20 #set for Arithmetic domain
sample_size = []
#for target_task in range(20):
features = pickle.load(open('reg_features_Arithmetic_%i.pkl'%target_task,'rb'))




#Isolating features and creating the model

f = np.array(features)
f_name = np.array(generate_feature_name(num_task))
s_f = np.size(f,axis=1)

#handling Nan values for Accuracy and Latency            
for i in range(num_task):
    f[np.isnan(f[:,i*3+1]),i*3+1] = 1
    f[np.isnan(f[:,i*3+2]),i*3+2] = np.nanmin(f[:,i*3+2])


   
#reordering features to put the output on the outside
output = np.array(f[:,target_task*3])
   
#f[:,target_task*3] = f[:,-1] #copies gender over target_task sessions
#f = f[:,:-1]    #removes gender from the last column
f = np.concatenate((f[:,:target_task*3],f[:,-29:]),axis=1) #Removes tasks after the task performance

#f_name[target_task*3] = f_name[-1] #copies gender over target_task sessions
#f_name = f_name[:-1]    #removes gender from the last column
f_name = np.concatenate((f_name[:target_task*3],f_name[-29:]),axis=0)
       
temp = np.argsort(output)
output = output[temp]
f = f[temp,:]


regr, predicted = generate_model(f,output)

pickle.dump(regr,open('reg_model_%s_%i.pkl'%(domain,target_task),'wb'))
    

#Prints the model fit statistics

# The coefficients
print '\nTask %i'%target_task
print 'Total Users: %i'%len(output)
sample_size.append(len(output))
#print('Coefficients: \n', regr.coef_)
# The mean square error
print"Root Mean Squared Error %.2f"%metrics.mean_absolute_error(predicted,output)

yhat = predicted
y = output                         # or [p(z) for z in x]
ybar = np.sum(output)/len(output)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
Rsquared = ssreg / sstot
print 'R Squared = %.2f'%Rsquared



threshold = 2.99 #used for classification estimates

print"Classification Error Rate:"
print 100*metrics.confusion_matrix(predicted<threshold,output<threshold)/len(output)


# Explained variance score: 1 is perfect prediction
rank = cross_val_feature(f,output,regr)
temp = np.argsort(abs(rank))
for i in temp:
    print '%s: %.2f'%(f_name[i],rank[i])


plot_regression(predicted,output)
    