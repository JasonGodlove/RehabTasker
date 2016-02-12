# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 09:35:29 2016
Task Type Descriptors 

Finds how much overlap there is with the tasks and plots it

@author: jgodlove

"""
import pickle
import pandas as pd
import numpy as np
task_progression = pickle.load(open('task_progression.pkl','rb'))
tasks = task_progression[task_progression.task_level == 1]
tasks = tasks[tasks.system_name != 'NaN']
tasks = tasks[~np.isnan(tasks.task_type_id[:])]


count, ids = np.histogram(tasks.task_type_id,range(int(max(tasks.task_type_id))))
ids = ids[:-1]
ids = ids[count>1]
print tasks.system_name[ids]

ids = pd.DataFrame(ids,columns=['task_type_id'])
overlap = pd.merge(ids,tasks,on='task_type_id',how='inner')

print 'Overlapping Tasks\n'
print '\n'.join(overlap.display_name.unique())
print '\nOverlapping Domains\n'
for i in overlap.domain.unique():
    if np.sum(overlap.domain.values==i) ==1:
        print '%s  %i: %s %i'%(i,np.sum(overlap.domain.values==i),str(overlap['display_name'][overlap['domain'].values == i].values[0]),int(overlap['task_type_id'][overlap['domain'].values == i].values[0]))        
    else:        
        print '%s  %i'%(i,np.sum(overlap.domain.values==i))
    
print '\n\n'
print '\n'.join(task_progression.domain.unique().astype(str))
print task_progression[['task_type_id','progression_order']][task_progression.domain.values=='AuditoryMemory'].sort('task_type_id')
