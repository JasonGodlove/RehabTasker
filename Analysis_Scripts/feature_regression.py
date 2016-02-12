# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:53:06 2016
Using various features to predict number of sessions in a user data set.



@author: jgodlove
"""

''' Assembling the feature matrix
Customer features:
gender male,female, null int
therapy types 1-9 int

disorder 1-7 binary
deficits 1-9 binary
age group int
'>70'
'51-70'
'22-50'
'13-21'
'6-12'
'<6'

condition since int



User Features:
average sessions compeleted #sessions / (last session time - start session time)
total sessions per task, all tasks all levels, all 237 separate tasks
accuracy
latency


baseline task level
num_sessions in other tasks



end result:
num_sessions in a particular task 

'''


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
from sklearn.ensemble import RandomForestClassifier 
from sklearn import linear_model
from sklearn.preprocessing import normalize
from sklearn.feature_selection import RFECV
import sklearn.svm as svm

def query_patient_data(p,domain,target_task,cur):                
    query = """SELECT count(*) FROM constant_therapy.sessions where patient_id = %s and task_type_id = %i and task_level =%i"""%(p,domain_task[domain_task.progression_order.values==target_task].task_type_id.values,domain_task[domain_task.progression_order.values==target_task].task_level.values)
    cur.execute(query)
    result = cur.fetchall()
    if result[0][0] == 0: #no data to add to the model
        return None
    print 'Processing %i'%p
    
    temp_features = []
    
    for i in range(len(domain_task)):
        query = """SELECT count(*) FROM constant_therapy.sessions where patient_id = %s and task_type_id = %i and task_level =%i"""%(p,domain_task[domain_task.progression_order.values==i].task_type_id.values,domain_task[domain_task.progression_order.values==i].task_level.values)
        cur.execute(query)
        result = cur.fetchall()   
        temp_features.append(result[0][0])
        if result[0][0] != 0:    
            query = """SELECT accuracy,latency FROM constant_therapy.sessions where patient_id = %s and task_type_id = %i and task_level =%i order by start_time ASC limit 1"""%(p,domain_task[domain_task.progression_order.values==i].task_type_id.values,domain_task[domain_task.progression_order.values==i].task_level.values)
            #query = """SELECT min(accuracy), max(latency) FROM constant_therapy.sessions where patient_id = %s and task_type_id = %i and task_level =%i"""%(p,domain_task[domain_task.progression_order.values==i].task_type_id.values,domain_task[domain_task.progression_order.values==i].task_level.values)                    
            cur.execute(query)
            result = cur.fetchall()
            if len(result) == 0:
                temp_features.append(np.nan)
                temp_features.append(np.nan)
            else:
                if result[0][0] == None:
                    temp_features.append(np.nan)
                else:
                    temp_features.append(np.float(result[0][0]))
                if result[0][1] == None:
                    temp_features.append(np.nan)
                else:
                    temp_features.append(np.float(result[0][1]))
        else:
            temp_features.append(np.nan)
            temp_features.append(np.nan)
                    
    '''
    User Features:
    average sessions compeleted #sessions / (last session time - start session time)
    total sessions per task, all tasks all levels, all 237 separate tasks
    accuracy
    latency
    baseline task level
    num_sessions in other tasks
    '''
    
    
    query = """SELECT count(*)/datediff(max(start_time),min(start_time)),count(*) FROM constant_therapy.sessions where patient_id = %i"""%(p)
    cur.execute(query)
    result = cur.fetchall()
    if result[0][0] == None:
        temp_features.append(result[0][1]) #Average sessons attempted per day
    else:
        temp_features.append(np.float(result[0][0])) #Average sessons attempted per day
    
    
    ''' Assembling the feature matrix
    Customer features:
    
    
    disorder 1-7 binary
    deficits 1-9 binary
    therapy types 1-9 int    
    
    age group int
    '>70'
    '51-70'
    '22-50'
    '13-21'
    '6-12'
    '<6'
    
    condition since int    
    
    gender male,female, null int, M=1 F =2, everything else = 0
    therapy types 1-9 int    
    
    '''    
    #deficits 1-9 binary 0 for all = unanswered
    query = """SELECT deficit_id FROM ct_customer.customers a INNER JOIN ct_customer.customers_to_deficits b on a.id = b.customer_id
            where user_id = %i;"""%(p)
    cur.execute(query)
    result = cur.fetchall()
    result = [i[0] for i in result]
    for i in range(1,10):
        if i in result:
            temp_features.append(1) #Deficit present
        else:
            temp_features.append(0) #Deficit absent
    
    #disorder 1-7 binary
    query = """SELECT disorder_id FROM ct_customer.customers a INNER JOIN ct_customer.customers_to_disorders b on a.id = b.customer_id
            where user_id = %i;"""%(p)
    cur.execute(query)
    result = cur.fetchall()
    result = [i[0] for i in result]
    for i in range(1,8):
        if i in result:
            temp_features.append(1) #disorder present
        else:
            temp_features.append(0) #disorder absent
    
    #Therapy type 1-9 binary
    query = """SELECT therapy_type_id FROM ct_customer.customers a 
            INNER JOIN ct_customer.customers_to_therapy_types b on a.id = b.customer_id 
            where user_id = %i;"""%(p)
    cur.execute(query)
    result = cur.fetchall()
    result = [i[0] for i in result]
    for i in range(1,10):
        if i in result:
            temp_features.append(1) #therapy type present
        else:
            temp_features.append(0) #therapy type absent
    
    #Age, Time since condition, gender
    query = """SELECT age_group,condition_since,gender FROM ct_customer.customers where user_id = %i;"""%(p)
    cur.execute(query)
    result = cur.fetchall()
    temp_features.append(age.index(result[0][0])) #automatically adapts to changing age categories
    temp_features.append(condition_since.index(result[0][1])) #automatically adapts to changing age categories
    if result[0][2] == 'male':
        temp_features.append(1) #male
    elif result[0][2] == 'female':
        temp_features.append(2) #female
    else:
        temp_features.append(0) #Other
    return temp_features
#End of Query Patient Data

def patient_feature_groups(cur):
    #Grabs the age group and condition since categorys to make a reference table
    query = '''SELECT distinct age_group FROM ct_customer.customers order by age_group;'''
    cur.execute(query)
    result = cur.fetchall()
    age = [i[0] for i in result]
    print 'Updated Age Groups'
    
    query = '''SELECT distinct condition_since FROM ct_customer.customers order by condition_since;'''
    cur.execute(query)
    result = cur.fetchall()
    condition_since = [i[0] for i in result]
    print 'Updated Time Since Condition Groups'
    return age,condition_since
#End of patient_feature_groups


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

def patient_query(cur):
    #Collect Patient ID's of full schedules patients
    #200 yeilds 1034 patients
    query = """select n.patient_id from (select patient_id, count(*) as num_sessions
                        from constant_therapy.sessions
                        group by patient_id) n
                        where num_sessions > 10
                        and n.patient_id IN (SELECT distinct constant_therapy.schedules.patient_id FROM constant_therapy.schedules
                        WHERE constant_therapy.schedules.patient_id NOT IN (SELECT distinct constant_therapy.schedules.patient_id FROM constant_therapy.schedules
                        WHERE constant_therapy.schedules.patient_id = constant_therapy.schedules.clinician_id or constant_therapy.schedules.type is null));"""
    
    cur.execute(query)
    patient_list = pd.DataFrame(cur.fetchall(),columns=['id']) #problem line that freezes
    patient_list = patient_list.id.values.tolist()
    return patient_list



chdir('/home/jgodlove/gitrepo/Insight/ConstantTherapy')



task_progression = pickle.load(open('task_progression.pkl','rb'))
print 'Loaded task_progression pickle'



target_domains = {'Arithmetic':[np.nan],'Writing':[np.nan],'Production':[9,10,11],'Quantitative':[4,5,6],'AuditoryMemory':[16,0,7,6,5,3,15]} # numbers indicate progression IDs of overlaping tasks (generally just one task type) 

#Connect to SQL to get the task_types table to match display_name and id with system_name 


config = {
#Removed for Confidentiality
}

# open database connection
cnx = None
try:
    cnx = mysql.connector.connect(**config)
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
    raise

print 'Connected to Constant Therapy SQL'
cur = cnx.cursor()


age,condition_since = patient_feature_groups(cur)





patient_list = patient_query(cur) #Grabs all patients which are online only and have done at least 10 sessions
p_list = [];
                




target_domains = {'Arithmetic':[np.nan],'Writing':[np.nan],'Production':[9,10,11],'Quantitative':[4,5,6],'AuditoryMemory':[16,0,7,6,5,3,15]} # numbers indicate progression IDs of overlaping tasks (generally just one task type) 

for domain in target_domains:
    #domain = 'Arithmetic'##################### Used for doing one domain at a time instead of them all
    domain_task = task_progression[task_progression.domain.values == domain]
    num_task = len(domain_task)
    feature_name = generate_feature_name(num_task)

    for target_task in range(num_task):
        try:
            #setup the feature matrix
            #just to get it to work, do just number of sessions of other tasks in Arithmetic

            #target_task = 14####################### Used for only doing one task at a time instead of them all
            
            features = []
            #Grab patient feature data relevent to the task
            for p in patient_list:
                #check to make sure the patient has     sessions in that domain
                temp_features = query_patient_data(p,domain,target_task,cur)
                if temp_features is None:
                    continue
                p_list.append(p)
                features.append(temp_features)
            
            pickle.dump(features,open('reg_features_%s_%i.pkl'%(domain,target_task),'wb'))
            pickle.dump(p_list,open('reg_p_list_%s_%i.pkl'%(domain,target_task),'wb'))
    
            
       
            #Now handle tasks that were not attempted
            f = np.array(features)
            f_name = np.array(feature_name)
            s_f = np.size(f,axis=1)
            
            #handling Nan values for Accuracy and Latency            
            for i in range(num_task):
                f[np.isnan(f[:,i*3+1]),i*3+1] = 1
                f[np.isnan(f[:,i*3+2]),i*3+2] = np.nanmin(f[:,i*3+2])
            
            
            output = np.array(f[:,target_task*3])
            #output[output==1] = 0
            #output[np.where(np.logical_and(output>=4, output<=6))] = 1
            #output[output>1] = 1
            
            
            
            #Old Model attempts using all task data except the task in question (later changed to just preceding tasks
            
            f[:,target_task*3] = f[:,-1] #copies gender over target_task sessions
            f = f[:,:-1]    #removes gender from the last column
            f_name[target_task*3] = f_name[-1] #copies gender over target_task sessions
            f_name = f_name[:-1]    #removes gender from the last column
                        
            #test and training size before implimenting leave one out cross-validation (see feature_extractor.py) 
            train_size = int(np.floor(np.size(f,axis=0)*.8))
            
            train_X = f[:train_size]
            train_Y = output[:train_size]
            test_X = f[train_size:]
            test_Y = output[train_size:]
            #del f,output
            
            temp = np.argsort(test_Y)
            test_Y = test_Y[temp]
            test_X = test_X[temp]
            
            temp = np.argsort(train_Y)
            train_Y = train_Y[temp]
            train_X = train_X[temp]
            
            
            
            regr = linear_model.BayesianRidge(normalize=True)
            #regr = linear_model.LogisticRegression(penalty='l1',solver='liblinear')
            
            # Train the model using the training sets
            regr.fit(train_X,train_Y)
            
            
            #Old model regression plotting with stats, see feature_extractor.py for up-to-date feature plotting 
            # The coefficients
            print 'Linear Regression'
            print('Coefficients: \n', regr.coef_)
            # The mean square error
            print("Residual sum of squares: %.2f"
                  % np.mean((regr.predict(test_X) - test_Y) ** 2))
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % regr.score(test_X, test_Y))

            rfecv = RFECV(regr, step=1)
            rfecv.fit(test_X,test_Y)
            print 'The %i features are:'%rfecv.n_features_
            
            temp = np.sort(f_name[np.argsort(rfecv.ranking_)[:rfecv.n_features_]])
            print '\n'.join(i for i in temp) 
            
            
            plt.figure()
            
            # Plot outputs
            plt.subplot(221)
            plt.scatter(range(len(train_Y)), train_Y,  color='black')
            plt.plot(range(len(train_Y)), regr.predict(train_X), color='blue',
                     linewidth=3)
            plt.xticks(())
            plt.yticks(())
            plt.title('Linear Regression Training for Task %i'%target_task)
            
            plt.subplot(222)            
            plt.scatter(range(len(test_Y)), test_Y,  color='black')
            plt.plot(range(len(test_Y)), regr.predict(test_X), color='blue',
                     linewidth=3)
            plt.xticks(())
            plt.yticks(())
            plt.title('Linear Regression Test')
            
            
            
            
            #'''
            plt.figure()            
            forest = RandomForestClassifier(n_estimators = 1)
            #forest = svm.SVC()
            # Fit the training data to the Survived labels and create the decision trees
            forest = forest.fit(train_X,train_Y)
            
            # Take the same decision trees and run it on the test data
            print '\nRandom Forest'
            print("Residual sum of squares: %.2f"
                  % np.mean((forest.predict(test_X) - test_Y) ** 2))
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % forest.score(test_X, test_Y))
            
            # Plot outputs
            plt.subplot(223)
            plt.scatter(range(len(train_Y)), train_Y,  color='black')
            plt.plot(range(len(train_Y)), forest.predict(train_X), color='blue',
                     linewidth=3)
            plt.xticks(())
            plt.yticks(())
            plt.title('Random Forest Training')
            
            plt.subplot(224)
            plt.scatter(range(len(test_Y)), test_Y,  color='black')
            plt.plot(range(len(test_Y)), forest.predict(test_X), color='blue',
                     linewidth=3)
            plt.xticks(())
            plt.yticks(())
            plt.title('Random Forest Test')
            #'''
            plt.show()
            
            plt.savefig('Regression_%s_%i.png'%(domain,target_task))
            plt.close('all')
        except:
            print '************************************************************\nErrored on %s Task %i\n***************************************'%(domain,target_task)
            continue




'''
'''
# done with cursor
cur.close()             

# done with database
cnx.close()

plt.figure()
#counts=[]
for i in range(20):
    f= np.array(pickle.load(open('reg_features_Arithmetic_%i.pkl'%i,'rb')))
    #counts.append(np.size(features,axis=0))
    output = np.array(f[:,i*3])
    plt.subplot(4,5,i+1)
    plt.hist(output,range(20))
    plt.title(str(i))

         
