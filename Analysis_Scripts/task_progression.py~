# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:59:29 2016

Calculates the Task Progression database from the xls file and saves to a pickle

Queries and plots the difficulty levels of a single patient



@author: jgodlove
"""

import mysql.connector
from mysql.connector import errorcode

import pandas as pd
import numpy as np
import matplotlib as plt
import pickle


def create_task_progression_df():



#Importing xls file
task_progression_xls = pd.read_excel('TaskProgressionRules_2016-01-14.xlsx',sheetname='task_progressions')

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
        endLvl = int(levels[levels.find('-')+1:])+1 +1#added an additional 1 for ease in using range and finding lenght of entries
        row = pd.DataFrame({'domain':[currentDomain] * (endLvl - startLvl),
                        'progression_order':range(currentOrder,currentOrder + endLvl - startLvl),
                        #'task_type_id':np.nan * (endLvl - startLvl),
                        'system_name':[task_progression_xls['Task'][i]] * (endLvl - startLvl),
                        #'display_name':np.nan * (endLvl - startLvl),
                        'task_level':range(startLvl,endLvl)})
        task_progression = task_progression.append(row)     
        currentOrder+= 1 * (endLvl - startLvl)



#for row in range(len(task_progression)):
#    print task_progression.values[row]











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



#DB Query parameters 
'''
Finding parameters for :
A single patient (24808)
a single Domain (Writing)

only from sessions.type = 'SCHEDULED'

Returning the type of task, as translated into a descrete level

#Returning the performance score for the task (accuracy & latency)

'''

patient_id = 24808
sessions_type = 'SCHEDULED' #sessions.type




# first get the task types and their max levels using a SQL query
query = "SELECT id, max_level FROM constant_therapy.task_types where is_active = 1"
cur = cnx.cursor()


#cur = cnx.cursor(buffered = True) #use to buffer small results in SQL to later be worked with
cur.execute(query)

taskTypes = []  # create a new empty list for taskTypes
rows = cur.fetchall()
for row in rows:
    taskTypes.append(TaskType(row[0], row[1]))


# done with cursor
cur.close()             

# done with database
cnx.close()
         

