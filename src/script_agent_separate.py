# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:29:08 2019

@author: A19040016
"""

import pandas as pd
import numpy as np
import header.fund_selection.RUNHEADER as RUNHEADER

def getData(file_name, target_col):
    '''
    rtype : list, list
    
    check count, if date count is differenct with return count, notice
    '''
    data = pd.read_csv(file_name)
    date = data['date'].tolist()
    ret = data[target_col].tolist()    
    if len(date) != len(ret):
        print('return count error')
    return date, ret
    
from collections import Counter

def getCount(date):
    '''
    type date : list
    rtype : int, list
    
    return count, unique date
    check count, if date count is different, notice
    '''
    date_count = Counter(date)
    count = 0
    for i in date_count.items():
        if count == 0:
            count = i[1]
        elif count != i[1]:
            print('date count error')
        else:
            count = i[1] 
    return count, sorted(date_count.keys())

def agentReturn(ret, count, length):
    '''
    type ret : list
    type count : int
    type length : int
    rtype : numpy
    
    create agent base daily 5 day return data 
    '''
    ret_agent = np.zeros((count,length))
    for i in range(count):
        for j in range(length):
            ret_agent[i][j] = ret[i+j*count]        
    return ret_agent

def agentSum(ret_agent):
    '''
    type ret_agent : numpy
    rtype : numpy
    '''
    ret_sum =np.array(list(map(nextSum,ret_agent)))
    return ret_sum

def nextSum(agent):
    '''
    type agent : np.array
    rtype : np.array
    
    summation
    '''
    temp = np.zeros(agent.shape)
    for i in range(agent.shape[0]):
        temp[i] = sum(agent[:i+1])
    return temp

import os

def createFolder(file_name):
    '''
    type file_name : string
    
    create folder
    '''
    try:
        if not os.path.exists(file_name):
            os.makedirs(file_name)
    except OSError:
        print('exist name')

def writeCsv(file_name, date, target_col, ret_agent, ret_sum):
    '''
    type file_name : string
    type date : list
    type target_col : string
    type ret_agent : numpy
    type ret_sum : numpy
    '''
    for i in range(ret_agent.shape[0]):
        d = {'date': date, target_col:ret_agent[i], 'sum': ret_sum[i]}
        df = pd.DataFrame(data=d).set_index('date')
        df.to_csv('./{}/{}.csv'.format(file_name,str(i)))

if __name__=='__main__':
    m_name = RUNHEADER.m_name
    # m_name = './save/result/AddMb_MdSampling_sat_step5'
    _model_location = './save/model/rllearn/' + m_name
    _result = './save/result'
    _result = '{}/{}'.format(_result, _model_location.split('/')[-1])

    import os
    file_names = os.listdir(_result)
    for file_name in file_names:
        if 'test' in file_name:
            file_name = _result + '/' + file_name
            target_col = '0day_return'
            date, ret = getData(file_name, target_col)
            count, date = getCount(date)
            ret_agent = agentReturn(ret, count, len(date))
            ret_sum = agentSum(ret_agent)
            createFolder(file_name[:-4])
            writeCsv(file_name[:-4], date, target_col, ret_agent, ret_sum)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
