# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:35:38 2018

@author: ldh

重做模型训练，模型保存，特征更新以及信号插入的逻辑。
"""

# temp_fix.py

import time
import os
import datetime as dt
import yaml
import pandas as pd
from sqlalchemy.types import VARCHAR,DECIMAL,DATETIME
from podaci.guosen.data import (get_stock_basic,get_stock_daily_data,
                                save_into_db,execute_session)

with open('etc.yaml','r') as f:
    etc = yaml.load(f)
    
train_data_path = etc['train_data_path']
model_path = etc['model_path']

#%% 确定股票池
stock_universe = get_stock_basic()

today = dt.datetime.today()
start_date = (today - dt.timedelta(days = 365 * 3)).strftime('%Y%m%d')

target_universe = stock_universe.loc[stock_universe['list_date'] <= start_date]
target_universe_list = target_universe['stock_code'].tolist()

#%% 训练数据准备
def feature_handler(row):
    body =  (row['close_price'] - row['open_price']) / row['open_price']
    color = 1 if row['close_price'] >= row['open_price'] else 0
    upper = (row['high_price'] - max(row['close_price'],row['open_price'])) / max(row['close_price'],row['open_price'])
    lower = (min(row['close_price'],row['open_price']) - row['low_price'] ) / min(row['close_price'],row['open_price'])
    open_level = (row['open_price'] - row['prev_close_price']) / row['prev_close_price']
    close_level = (row['close_price'] - row['prev_close_price']) / row['prev_close_price']
    amount_change = (row['amount'] - row['prev_amount']) / row['prev_amount']
    return [row['trade_date'],body,color,upper,lower,open_level,close_level,amount_change]


for stk in target_universe_list:
    start_time = time.time()
    stock_data = get_stock_daily_data('20150801','20180831',[stk])
    stock_data = stock_data.sort_values('trade_date',ascending = True)
    stock_data['prev_close_price'] = stock_data['close_price'].shift(1)
    stock_data['prev_amount'] = stock_data['amount'].shift(1)
    stock_data.dropna(inplace = True)

    stock_feature = []
    
    for idx,row in stock_data.iterrows():
        stock_feature.append(feature_handler(row))
    
    stock_feature = pd.DataFrame(stock_feature,columns = ['trade_date','b','c','u','l','ol','cl','ac'])
    stock_feature['stock_code'] = stk
    stock_feature['update_datetime'] = dt.datetime.today()
    # ---------- 股票特征写入数据库 ----------------
    dtype_dict = {'stock_code':VARCHAR(8),'trade_date':VARCHAR(8),
                  'b':DECIMAL(20,6),'c':DECIMAL(20,6),
                  'u':DECIMAL(20,6),'l':DECIMAL(20,6),
                  'ol':DECIMAL(20,6),'cl':DECIMAL(20,6),
                  'ac':DECIMAL(20,6),'update_datetime':DATETIME}
    save_into_db(stock_feature.drop_duplicates(['stock_code','trade_date']),
                 'LDH_features_source',dtype_dict,'gb',if_exists = 'replace')
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
    end_time = time.time()
    print '[FEATURE CALCULATION]%s cost: %s'%(stk,(end_time - start_time))
    
#%% 模型训练
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

from podaci.guosen.data import get_stock_features
col_names = ['b','c','u','l','ol','cl','ac']

for stk in target_universe_list:
    
    try:
        start_time = time.time()
        label = pd.read_hdf(os.path.join(train_data_path,stk+'.h5'),'label')
        label['label'].loc[label['label'] == 2] = 1
        # feeatures
        features = get_stock_features('20150801','20180831',[stk])
        
        features = features.sort_values('trade_date',ascending = True)
        
        for i in range(1,31):
            for col in col_names:
                features['%s_%s'%(col,i)] = features[col].shift(i)
        features.dropna(inplace = True)
        features.drop(col_names,axis = 1,inplace = True)
        
        # combine and get train&test data
        comb = label.join(features.set_index('trade_date').drop('stock_code',axis = 1),
                          on = 'trade_date')
        
        comb = comb.drop(['stock_code'],axis = 1)
        comb.dropna(inplace = True)
        
        if len(comb) <= 250:
            print '%s sample data is too little'
            continue
        X = comb.drop(['label','trade_date'],axis = 1).values
        y = comb['label'].values
        
        # Model persistence
        mlp = MLPClassifier((100,100),activation = 'logistic')
        mlp.fit(X,y)
        joblib.dump(mlp,os.path.join(model_path,'%s.pkl'%stk))
        end_time = time.time()
        print '%s cost: %s'%(stk,(end_time - start_time))    
    except Exception as e:
        print '%s at %s'%(e,stk)
        
#%% 更新所有可用模型列表
import os
dirs = os.listdir(model_path)
stock_list = [each.split('.')[0] for each in dirs]
df = pd.Series(stock_list,name = 'model_code')
df.to_excel('models.xlsx')

#%% 更新到最新的特征
import os
os.system('python feature_refresh.py')

#%% 更新信号
os.system('python intra_signal_update.py')
