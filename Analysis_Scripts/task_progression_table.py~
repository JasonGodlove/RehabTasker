# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:59:29 2016

Creates the task_progression data frame that sets an order to all tasks and difficulty levels from the xls sheet 
Queries and plots the difficulty levels of a single patient



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

chdir('/home/jgodlove/gitrepo/Insight/ConstantTherapy')

def create_task_progression_df(cur,target_domains):
    #Importing xls file
    task_progression_xls = pd.read_excel('TaskProgressionRules_2016-01-14.xlsx',sheetname='task_progressions')
    print 'Loaded task progression xls file'
    '''formatting for a useable table
    Headers: 
    domain_id
    domain
    
    progression_order is index
    task_type_id
    system_name
    display_name
    task_level
    
    '''
    
    task_progression = pd.DataFrame(index=range(0),columns = ['domain',
                    'progression_order',
                    #'task_type_id',
                    'system_name',
                    #'display_name',
                    'task_level'])
    
    currentDomain = np.nan
    currentOrder = 0
    
    for i in range(len(task_progression_xls)):
        
        if not str(task_progression_xls['Domain'][i]) == 'nan': #if a New domain, set it up
            currentDomain = task_progression_xls['Domain'][i]
            currentOrder = 0
            continue
        
        if  str(task_progression_xls['Task'][i])=='nan' : #if a blank space, then skip it
            continue
    
        #check for multiple task difficulty levels imbeded in one entry
            
        if  len(str(task_progression_xls['Levels'][i])) == 1:
            
        #Load the task into the dataframe    
            row = pd.DataFrame({'domain':currentDomain,
                            'progression_order':[currentOrder],
                            #'task_type_id':np.nan,
                            'system_name':task_progression_xls['Task'][i],
                            #'display_name':np.nan,
                            'task_level':[int(task_progression_xls['Levels'][i])]})
            task_progression = task_progression.append(row)     
            currentOrder+=1
        else:
            levels = str(task_progression_xls['Levels'][i])
            startLvl = int(levels[0:levels.find('-')])
            endLvl = int(levels[levels.find('-')+1:])+1 #added an additional 1 for ease in using range and finding length of entries
            row = pd.DataFrame({'domain':[currentDomain] * (endLvl - startLvl),
                            'progression_order':range(currentOrder,currentOrder + endLvl - startLvl),
                            'system_name':[task_progression_xls['Task'][i]] * (endLvl - startLvl),
                            'task_level':range(startLvl,endLvl)})
            task_progression = task_progression.append(row)     
            currentOrder+= 1 * (endLvl - startLvl)
    #End of importing excell file

    print 'Formated task_progression DF from Excel Sheet'
    
    #for row in range(len(task_progression)):
    #    print task_progression.values[row]

    # first get the task types and their max levels using a SQL query
    query = "SELECT id,system_name,display_name FROM constant_therapy.task_types where is_active = 1"
    
    cur.execute(query)
    
    print 'Queried task_types table'
    
    task_types = cur.fetchall()
    
    
    task_types = pd.DataFrame(task_types,columns=['task_type_id','system_name','display_name'])
    
    task_progression = pd.merge(task_progression,task_types,on='system_name', how='outer')#task_progression.merge(task_types)
    
    print 'Queried and merged for final task_progression table'
    
    pickle.dump(task_progression,open("task_progression.pkl","wb"))
    
    return task_progression
#End of create_task_progression_df    

def plot_domain_distribution(cur):    
    #finds the number of sessions in different domain distributions     
    query = """select domain,count(*) from (SELECT * FROM constant_therapy.sessions where type='SCHEDULED' and task_type_id is not null) a
    inner join task_progression b
    on a.task_type_id = b.task_type_id
    group by domain;"""
    cur.execute(query)
    
    domain_distribution = cur.fetchall()
    domain_distribution = pd.DataFrame(domain_distribution,columns=['domain','num_sessions'])
    domain_distribution = domain_distribution[1:] #removes None tasks
    label = []
    for i in domain_distribution.domain.values:
        label.append(str(i))
    domain_distribution.plot(style='barh',y=label)
    return domain_distribution
#end of plot_domain_distriubtion


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

    

if isfile('task_progression.pkl'):
    task_progression = pickle.load(open('task_progression.pkl','rb'))
    print 'Loaded task_progression pickle'
else:
    task_progression = create_task_progression_df(cur)

domain_distribution = plot_domain_distribution(cur)#plots the number of sessions amoung the various domains





#Now time to get some Patient data

#domain = 'Arithmetic'
#num_task = 19


if isfile('p_list.pkl'):
    patient_list = pickle.load(open('p_list.pkl','rb'))
else:
    #query = "SELECT start_time,task_type_id,task_level,accuracy,latency FROM constant_therapy.sessions where patient_id = 24808 and type = 'SCHEDULED' and task_type_id is not null"
    '''query = """SELECT r.start_time,r.task_type_id,r.task_level,r.accuracy,r.latency
                	FROM constant_therapy.responses r
                    JOIN constant_therapy.sessions s on s.id = r.session_id 
                    WHERE r.patient_id = 15901 and r.task_type_id is not null and s.type = 'SCHEDULED' """
        
        #Query that finds patients that are full BOT
    #select distinct patient_id from constant_therapy.schedules where patient_id not in (SELECT distinct patient_id FROM constant_therapy.schedules s where s.patient_id = s.clinician_id or s.type is null);
    #Query below uses patient stats which has a lot less patients
    query = """SELECT p.patient_id FROM constant_therapy.patient_stats p 
                WHERE p.total_task_count > 200 and p.patient_id IN (SELECT distinct patient_id FROM constant_therapy.schedules s 
                    WHERE s.patient_id NOT IN (SELECT distinct s.patient_id FROM constant_therapy.schedules s 
                        WHERE s.patient_id = s.clinician_id or s.type is null))"""
    '''
    #Query below isolates 10730
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

print 'Processing %i users: IDs from %i to %i'%(len(patient_list),min(patient_list),max(patient_list))






task_tally = pd.DataFrame(range(0),columns=['patient_id','domain','tally'])#[]#np.array([0]*19) #mod for each domain
p_list = []
#for p in patient_list['patient_id']:#[:20]:

for p in patient_list:#[patient_list.index(48335):]:
    
    query = """SELECT s.id,s.start_time,s.task_type_id,s.task_level,s.accuracy,s.latency
                	FROM constant_therapy.sessions s
                     WHERE s.patient_id = %i and s.task_type_id is not null and s.type = 'SCHEDULED' and s.accuracy is not null""" % p
    
    
    #cur = cnx.cursor(buffered = True) #use to buffer small results in SQL to later be worked with
    cur.execute(query)
    
    sleep(2)
    
    patient_response = cur.fetchall() #problem line that freezes
    
    
    
    #patient_response = pd.DataFrame(patient_response,columns=['start_time','task_type_id','task_level','accuracy','latency'])
    patient_response = pd.DataFrame(patient_response,columns=['id','start_time','task_type_id','task_level','accuracy','latency'])
    patient_response = pd.merge(patient_response,task_progression,on=['task_type_id','task_level'],how='inner')
     #patient_response.merge(task_progression[list(['domain','progression_order','task_type_id','task_level'])])
    
    print 'Queried patient %i data and merged with task_progression' % p
    '''
    for domain in task_progression.domain.unique():
        if domain == 'nan':
            continue
        if any(patient_response.domain == domain):
            
            patient_response[patient_response.domain == domain].plot(x='start_time',y='progression_order',style='o-',title=domain)
        else:
            print 'No records for %s' % domain
    '''    
    
    patient_response = patient_response.sort_values(by='start_time')
    #temp = patient_response[patient_response.domain == domain].progression_order.diff()
    #temp = patient_response['start_time'][temp[abs(temp) > 1].index[-1]]
    #patient_response = patient_response[patient_response['start_time']>temp]
    #patient_response[patient_response.domain == domain].plot(x='start_time',y='progression_order',style='o-',title=str(p)+' '+ domain)

    
    for domain in target_domains:
        if sum(patient_response.domain.values==domain) < 10: #catch to speed up early processing
            continue
        num_task = np.sum(task_progression.domain.values==domain)
        
        pr = patient_response[patient_response.domain == domain].progression_order.values.tolist()
        pr_diff = np.diff(pr)
        pr_clone = list(pr)
        for i in range(len(pr_diff)-1,-1,-1):
            if (abs(pr_diff[i]) > 1) and (pr[i+1] in target_domains[domain]): #If the task jumps and the task is a known jumper
                if not pr[i] in target_domains[domain]: #Check to see if the previous task was a jumper, if not, delete this instance
                    pr_clone.pop(i+1)
        
        pr = list(pr_clone)
        del pr_clone
        
        
        #remove instances of overlaping task
        pr_diff = np.diff(pr)                
        try:
            if max(abs(pr_diff)) > 1:
                temp = np.where(abs(pr_diff) > 1)[0][-1]
            else:
                temp = -1
        except:
            continue    
        pr = pr[temp+1:]
        del pr_diff,temp
                
        
        '''
        plt.figure()
        plt.plot(pr,'o-')
        plt.title('%i %s' % (p,domain))
        plt.xlabel('Session#')
        plt.ylabel('Difficulty')
        plt.axis([0, len(pr), 0, num_task])
        '''
        #learning Regression
        try:
            x = range(len(pr))
            if max(pr) == num_task-1:
                x = range(len(pr[:pr.index(num_task-1)]))
            pr = np.array(pr)
            if len(pr) < 10: #catch to speed up processing
                continue
        except:
            continue
        
        try :
            slope, intercept, r_value, p_value, std_err = linregress(x,pr[x])
            x = np.array(x)    
            #plt.plot(x, slope*x + intercept, '-')
            #plt.title('%i %s Slope: %.2f  R^2: %.2f' % (p,domain,slope,r_value**2))
        except :
                slope = 0
        
        #update task_tally    
        if (slope > 0) and (r_value**2 > 0.1) and (len(x) > 10) :   
            count = np.histogram(pr,bins=range(0,num_task+1))[0]
            task_tally = task_tally.append(pd.DataFrame({'patient_id':p,'domain':domain,'tally':[count]}))
            #threshold = np.median(count[count>1])
            #task_tally[count>threshold] +=1
            #task_tally[[i for i in range(len(count)) if 0<count[i]<threshold]] -=1   
            #plt.savefig('%i_%s.png'%(p,domain))
            p_list.append(p)
            print '%i Valid for %s'%(p,domain)
        #plt.close('all')

pickle.dump(p_list,open('p_list.pkl','wb'))
pickle.dump(task_tally,open('task_tally.pkl','wb'))





###############################
for domain in target_domains:
    tally = task_tally[task_tally.domain.values==domain].tally.values.tolist() #Use this to get the tally to analyze
    num_task = np.sum(task_progression.domain.values==domain)
    
    #Analyzing the task tallys 
    task_tally_total = np.array([0]*num_task)
    task_tally_n =  np.array([0]*num_task)
    task_tally_norm = np.array([[np.NaN]*num_task]*len(tally))
    for i in range(len(tally)):
        row = np.array(tally[i]) #List to make it a copy and not a link
        threshold = np.median(row[row>0])    
        task_tally_n[row<>0] += 1
        task_tally_norm[i][row<>0] = np.array(row[row<>0]) - threshold
        row[row<>0] = row[row<>0] - threshold
        task_tally_total += np.sign(row)
    
    print np.nanmean(task_tally_norm,axis=0)
    task_se = [0]*num_task
    for i in range(num_task):
        task_se[i] = np.nanstd(task_tally_norm[:,i]) / np.sqrt(task_tally_n[i])
    
    for i in np.nanmean(task_tally_norm,axis=0) + task_se: 
        print '%.2f'%i
    
    task = task_progression[task_progression.domain.values == domain].sort_values('progression_order')
    task_p=[0]*num_task;
    for i in range(num_task):
        t,task_p[i] = stats.ttest_1samp(task_tally_norm[~np.isnan(task_tally_norm[:,i]),i],0)
        task_p[i] /= 2;    
        print 'Task %15s %i: M:%.2f p:%.2f' % (task['display_name'].values[i],task['task_level'].values[i],np.nanmean(task_tally_norm[:,i]),task_p[i])
    
    
    
    #Plot task_tally
    plt.figure(figsize=[20,10])
    #plt.barh(range(1,num_task+1),[a/b for a,b in zip(task_tally_total.astype(float),task_tally_n.astype(float))],align='center')
    temp = (np.array(task_p) < 0.11) & (task_tally_n >= 5)
    m = np.nanmean(task_tally_norm,axis=0)
    x = np.array(range(1,num_task+1)) #incrementing progression_order score by 1 for display purposes
    plt.barh(x[temp],m[temp],align='center',color='r')#,xerr = task_se)
    temp = (np.array(task_p)>=0.11) | (task_tally_n < 5)
    plt.barh(x[temp],m[temp],align='center',color='b')#,xerr = task_se)
    temp = list(plt.axis())
    temp[0] = min([-5,temp[0]])
    temp[1] = max([5,temp[1]])
    temp[2] = 0
    temp[3] = num_task+1
    plt.axis(temp)
    plt.title(domain)
    plt.ylabel('Task Progression')
    #plt.gca().invert_yaxis()    
    plt.xlabel('Sessions from the Mean (Negative=Easier, Positive=Harder)')    
    for row in range(0,num_task):    
        temp = '%s %i'%(task['display_name'].values[row],task['task_level'].values[row])
        if np.nanmean(task_tally_norm,axis=0)[row] < 0:
            plt.text(0,row+1,'%i : %s'%(task_tally_n[row],temp), horizontalalignment='left',verticalalignment='center',fontsize=22)
        else:
            plt.text(0,row+1,'%s : %i'%(temp,task_tally_n[row]), horizontalalignment='right',verticalalignment='center',fontsize=22)
    plt.savefig('task_tally_'+domain+'.png')

# done with cursor
cur.close()             

# done with database
cnx.close()
         






