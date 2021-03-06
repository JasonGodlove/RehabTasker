from flask import Flask
from flask import render_template, request, make_response, session, redirect, url_for
from app import app
import StringIO
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import pickle
import pandas as pd
import numpy as np
from os import chdir
from os.path import isfile
from os import getcwd
from sklearn import linear_model
from matplotlib import pylab as plt
import seaborn as sns

#used to use functions from AppCode however, those relevent functions have been copied over

###############################################################
#Note: To ease conceptual understanding and terminology for outside companies, Sessions are refered to as Days to better convey it as a unit of time.
# in the code, sessions are still refered to as sessions, however plot printouts and website/slideshow references refer to it as Days. 
#################################################################

#chdir('/home/jgodlove/gitrepo/Insight/ConstantTherapy')
def drange(start, stop, step):
     r = start
     while r < stop:
     	yield r
     	r += step


def plot_hist(past_users,prediction):
    #Plots the histogram and prediction     
    sns.set(style="white")
    
    fig = plt.figure()
    sbin =  np.array(range(12))+.5   
    sns.distplot(past_users,bins=sbin,color='B',norm_hist=False)
    '''
    c = sns.color_palette('Spectral', 12)#"RdYlGn",'Spectral'
    c.reverse() #so red is on far right
    for i in range(12):
        sns.distplot(past_users,bins=sbin[i:],color=c[i],kde=False,norm_hist=False)
    '''
    
    #plt.hist(past_users,bins=np.array(range(12))+.5)
    axis = plt.axis()
    plt.axis([.5,11.5,axis[2],axis[3]])
    plt.vlines(prediction,axis[2],axis[3],linewidth=4,color='k')
    plt.yticks(list(drange(0,axis[3],.1)),fontsize=16)
    plt.xticks(range(1,12),fontsize=16)
    plt.xlabel('# of Days to Complete the Task',fontsize=16)
    plt.ylabel('Proportion of Users',fontsize=16    )
    
    return fig

def load_parameters(domain,target_task):
    #loads data files needed for the page
    print '\n'+getcwd()+'\n'
    model = pickle.load(open('app/static/pickles/reg_model_min_%s_%i.pkl'%(domain,target_task),'rb'))
    past_users = np.array(pickle.load(open('app/static/pickles/past_users_%s_%i.pkl'%(domain,target_task),'rb')))
    user_params = pickle.load(open('app/static/pickles/default_params_min_%s_%i.pkl'%(domain,target_task),'rb'))
    return past_users,user_params,model


def param_request(user_params):
    #Function is specfic to Arithmetic 14 layout    
    #user_params is setup with 71 features, but most are uneditable    
    
    reload_check = request.values.get('Age')    
    if reload_check is not None:#IF the page first loads, no values will be present to grab
        #If there are values uploaded, grab them
        for i in range(7,14):
            user_params[i*3] = int(request.values.get('Task%i'%i))
            user_params[i*3+1] = int(request.values.get('Task%iAcc'%i))
        
        #'Avg Session per Day' Skipped because it has little impact
        for i in range(1,10):
            user_params[i-29] = int(request.values.get('Deficit%i'%i)=='true')
        for i in range(1,8):
            user_params[i-20] = int(request.values.get('Disorder%i'%i)=='true')
        for i in range(1,10):
            user_params[i-13] = int(request.values.get('Therapy%i'%i)=='true')
        
        user_params[-3] = int(request.values.get('Age'))        
        #'Condition' Skipped becuase of it's low impact
        user_params[-1] = int(request.values.get('Gender'))
        
    return user_params



@app.route('/figure')
def figure_drawing():    
    domain = 'Arithmetic'
    target_task = 14
    past_users,user_params,model = load_parameters(domain,target_task)  
    user_params = param_request(user_params)
    
    prediction = model.predict(user_params.reshape(1,-1))
    if prediction < 1:
        prediction = 1
    elif prediction > 11: #Caps prediction to between 1-11 for better accuracy. Obviously more than 11 is Hard
        prediction = 11
    #fig = plt.figure()
    #plt.plot(user_params)
    #plt.title('Parameter Checker')
    
    fig = plot_hist(past_users,prediction)
    
    canvas = FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    plt.close(fig)
    return response


@app.route('/')

@app.route('/index')
def open_index():
    #pull 'ID' from input field and store it
    #city = request.args.get('ID')
    return render_template("index.html")

@app.route('/slides')
def open_slideshow():
    return render_template("slides.html")
    
@app.route('/contact')
def open_contact():
    return render_template("contact.html")
