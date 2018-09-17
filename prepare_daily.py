# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:35:01 2018

@author: ldh
"""

# prepare_daily.py

import time
import os
import datetime as dt
import yaml
import numpy as np
import pandas as pd
from sqlalchemy.types import VARCHAR,DECIMAL,DATETIME
from podaci.guosen.data import get_stock_basic,get_stock_daily_data

# 参数
label_thresh_hold = 0.2
rolling_window = 60
# 读取配置
with open('etc.yaml','r') as f:
    etc = yaml.load(f)

train_data_path = etc['train_data_path_daily']
model_path = etc['model_path_daily']

# 确定股票池
stock_universe = get_stock_basic()

today = dt.datetime.today()
start_date = '20050101'

target_universe = stock_universe.loc[stock_universe['list_date'] <= start_date]
target_universe_list = target_universe['stock_code'].tolist()

#%% 获取目标股票池股票日线数据

stock_daily = get_stock_daily_data(start_date,'20180801',stock_universe = target_universe_list)
stock_daily_group = stock_daily.groupby('stock_code',sort = False)

def roll_func(arr):
    last = arr[-1]
    
    up_array = np.linspace(last,last * (1 + label_thresh_hold),num = rolling_window)
    down_array = np.linspace(last,last * (1 - label_thresh_hold),num = rolling_window)
    zero_array = np.linspace(last,last,num = rolling_window)
    
    down_square = np.power(arr - down_array,2)
    down_score = down_square.sum()
    up_square = np.power(arr - up_array,2)
    up_score = up_square.sum()
    zero_square = np.power(arr - zero_array,2)
    zero_score = zero_square.sum()
    
    score = np.array([zero_score,up_score,down_score]) # 0为震荡,1为向上,2为向下
    label = score.argmin()
    return label

for stk in target_universe_list:
    try:
        start_time = time.time()
        tmp = stock_daily.loc[stock_daily['stock_code'] == stk]
        tmp = tmp.sort_values('trade_date',ascending = False)
        tmp = tmp.set_index('trade_date')
        tmp_roll = tmp['close_price'].rolling(rolling_window)
    
        label_ser = tmp_roll.apply(roll_func,raw = True)
        label_ser.name = 'label'
        
        df = pd.DataFrame(label_ser)
        df.dropna(inplace = True)
        df = df.reset_index()
        df['stock_code'] = stk
        df.stock_code = df.stock_code.apply(lambda x:x.encode('utf8'))
        df.trade_date = df.trade_date.apply(lambda x:x.encode('utf8'))
     
        df.to_hdf(os.path.join(train_data_path,'label.h5'),
                  key = stk.encode('utf8'),append = True)       
        end_time = time.time()
        print '%s cost: %s'%(stk,(end_time - start_time))
    except Exception as e:
        print 'Error at %s, reason is %s'%(stk,e)
        continue

    
#%% 特征数据更新
import datetime as dt
import pandas as pd
from podaci.guosen.data import (save_into_db,execute_session,get_stock_daily_data)
from utils import stock_feature_handler

model_codes = pd.read_excel('models.xlsx',dtype = {'model_code':str})
model_codes = model_codes['model_code'].tolist()

last_update = '20050101'
# 获取增量更新数据
start_date = last_update
end_date = dt.datetime.today().strftime('%Y%m%d')    
stock_data = get_stock_daily_data(start_date,end_date,target_universe_list) #
print 'GET STOCK DATA SUCCESSFULLY'
    
# 计算特征
stock_data_group = stock_data.groupby('stock_code',sort = False)
stock_features = stock_data_group.apply(stock_feature_handler)
stock_features = stock_features.reset_index()
stock_features.drop('level_1',axis = 1,inplace = True)
stock_features['update_datetime'] = dt.datetime.today()
# 写入数据库
dtype_dict = {'stock_code':VARCHAR(8),'trade_date':VARCHAR(8),
                  'b':DECIMAL(20,6),'c':DECIMAL(20,6),
                  'u':DECIMAL(20,6),'l':DECIMAL(20,6),
                  'ol':DECIMAL(20,6),'cl':DECIMAL(20,6),
                  'ac':DECIMAL(20,6),'update_datetime':DATETIME}
save_into_db(stock_features,'LDH_features_source',
             dtype_dict,'gb',
             if_exists = 'replace')
sql_merge = '''
MERGE INTO LDH_features as T
USING LDH_features_source as S
ON (T.trade_date = S.trade_date AND T.stock_code = S.stock_code)
WHEN NOT MATCHED BY TARGET
THEN INSERT 
(stock_code,trade_date,b,c,u,l,ol,cl,ac,update_datetime)
VALUES
(S.stock_code,S.trade_date,S.b,S.c,S.u,S.l,S.ol,S.cl,S.ac,S.update_datetime);  
'''
execute_session(sql_merge,'gb')
    
#%% 样本数据获取与模型训练
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import precision_recall_curve,precision_recall_fscore_support,accuracy_score

from podaci.guosen.data import get_stock_features
col_names = ['b','c','u','l','ol','cl','ac']

for stk in target_universe_list:
    try:
        start_time = time.time()

        label = pd.read_hdf(os.path.join(train_data_path,'label.h5'),stk.encode('utf8'))
        label.loc[:,'label'].loc[label['label'] == 2] = 1

        # feeatures
        features = get_stock_features('20050101','20180831',[stk])
        
        features = features.sort_values('trade_date',ascending = True)
        
        for i in range(1,181):
            for col in col_names:
                features['%s_%s'%(col,i)] = features[col].shift(i)
        features.dropna(inplace = True)
        features.drop(col_names,axis = 1,inplace = True)
        
        # combine and get train&test data
        comb = label.join(features.set_index('trade_date').drop('stock_code',axis = 1),
                          on = 'trade_date')
        
        comb = comb.drop(['stock_code'],axis = 1)
        comb.dropna(inplace = True)
        
        if len(comb) <= 1500:
            print '%s sample data is too little'
            continue
        X = comb.drop(['label','trade_date'],axis = 1).values
        y = comb['label'].values
        
        # the model
        # ------------ 模型测试 ----------------
#        train_score = []
#        test_score = []
#        
#        for i in range(30):        
#            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = i)
#            mlp_model = MLPClassifier((200,100),activation = 'logistic')
#            mlp_model.fit(X_train,y_train)
#            train_score.append(mlp_model.score(X_train,y_train))
#            test_score.append(mlp_model.score(X_test,y_test))
#            print '%s complete'%i
#            
#        df = pd.DataFrame([train_score,test_score],index = ['train','test']).T
#        accuracy = accuracy_score(y_test,mlp_model.predict(X_test))
#        precision = precision_score(y_test,mlp_model.predict(X_test))
#        prf = precision_recall_fscore_support(y_test,mlp_model.predict(X_test))
        # ------------ 模型测试 ----------------
        
        
        # Model persistence
        mlp = MLPClassifier((200,100),activation = 'logistic')
        mlp.fit(X,y)
        joblib.dump(mlp,os.path.join(model_path,'%s.pkl'%stk))
        end_time = time.time()
        print '%s cost: %s'%(stk,(end_time - start_time))    
    except Exception as e:
        print '%s at %s'%(e,stk)
        
#%% 可用模型列表
import os
dirs = os.listdir(model_path)
stock_list = [each.split('.')[0] for each in dirs]
df = pd.Series(stock_list,name = 'model_code')
df.to_excel('models_daily.xlsx')