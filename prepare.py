# -*- coding: utf-8 -*-
"""
Created on Wed Sep 05 15:05:19 2018

@author: ldh

样本数据准备
"""

# prepare.py
import time
import os
import datetime as dt
import yaml
import numpy as np
import pandas as pd
from sqlalchemy.types import VARCHAR,DECIMAL,DATETIME
from podaci.guosen.data import get_stock_basic,get_stock_daily_data,get_stock_min_data_close_multi

# 参数
label_thresh_hold = 0.03

# 读取配置
with open('etc.yaml','r') as f:
    etc = yaml.load(f)

train_data_path = etc['train_data_path']
model_path = etc['model_path']

# 确定股票池
stock_universe = get_stock_basic()

today = dt.datetime.today()
start_date = (today - dt.timedelta(days = 365 * 3)).strftime('%Y%m%d')

target_universe = stock_universe.loc[stock_universe['list_date'] <= start_date]
target_universe_list = target_universe['stock_code'].tolist()

#%% 获取目标股票池分钟线收盘数据

for stk in target_universe_list[2000:]: 
    try:
        start_time = time.time()
        #-------------获取分钟线数据------------------
        min_data1 = get_stock_min_data_close_multi('20160101','20161231',[stk],2016)
        min_data2 = get_stock_min_data_close_multi('20170101','20171231',[stk],2017)
        min_data3 = get_stock_min_data_close_multi('20180101','20180831',[stk],2018) 
    
        comb = pd.concat([min_data1,min_data2,min_data3])
        comb = comb.sort_values('trade_dt',ascending = True)
        #-------------获取分钟线数据------------------
        
        last_close_price = None
        trade_dates = comb['trade_date'].unique()
        
        label_data = []
        for trade_date in trade_dates:
            tmp = comb.loc[comb['trade_date'] == trade_date]
            tmp = tmp.drop_duplicates('trade_dt')
            if last_close_price is None:
                last_close_price = tmp['close_price'].iloc[-1]
            else:
                up_array = np.linspace(last_close_price,
                                last_close_price * (1 + label_thresh_hold),
                                240)
                down_array = np.linspace(last_close_price,
                                last_close_price * (1 - label_thresh_hold),
                                 240)
                zero_array = np.ones(240) * last_close_price
                
                close_price = tmp['close_price'].values
                down_square = np.power(close_price - down_array,2)
                down_score = down_square.sum()
                up_square = np.power(close_price - up_array,2)
                up_score = up_square.sum()
                zero_square = np.power(close_price - zero_array,2)
                zero_score = zero_square.sum()
                
                score = np.array([zero_score,up_score,down_score]) # 0为震荡,1为向上,2为向下
                label = score.argmin()
                stock_code = tmp.stockcode.iloc[0].values[0]
                label_data.append([stock_code,trade_date,label])
                
                last_close_price = tmp['close_price'].iloc[-1]
                
        df = pd.DataFrame(label_data,columns = ['stock_code','trade_date','label'])
        df.stock_code = df.stock_code.apply(lambda x:x.encode('utf8'))
        df.trade_date = df.trade_date.apply(lambda x:x.encode('utf8'))
        df.to_hdf(os.path.join(train_data_path,stk + '.h5'),key = 'label',append = True)        
        end_time = time.time()
        print '%s cost: %s'%(stk,(end_time - start_time))
    except Exception as e:
        print 'Error at %s, reason is %s'%(stk,e)
        continue
    
#%% 特征数据计算
from podaci.guosen.data import save_into_db,execute_session
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
    save_into_db(stock_feature,'LDH_features_source',dtype_dict,'gb',if_exists = 'replace')
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
    print '%s cost: %s'%(stk,(end_time - start_time))
    
    
#%% 样本数据获取与模型训练
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import precision_recall_curve,precision_recall_fscore_support,accuracy_score

from podaci.guosen.data import get_stock_features
col_names = ['b','c','u','l','ol','cl','ac']

for stk in target_universe_list[1833:]:
    
    try:
        start_time = time.time()
    # label
        label = pd.read_hdf(os.path.join(train_data_path,stk+'.h5'),'label')
        label['label'].loc[label['label'] == 2] = 1
    #    label['label'].loc[label['label'] == 0] = 'zd'    
    #    label['label'].loc[label['label'] == 1] = 'qs' 
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
        
        # the model
        # ------------ 模型测试 ----------------
    #    train_score = []
    #    test_score = []
    #    
    #    for i in range(30):        
    #        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = i)
    #        mlp_model = MLPClassifier((200,100),activation = 'logistic')
    #        mlp_model.fit(X_train,y_train)
    #        train_score.append(mlp_model.score(X_train,y_train))
    #        test_score.append(mlp_model.score(X_test,y_test))
    #        print '%s complete'%i
    #        
    #    df = pd.DataFrame([train_score,test_score],index = ['train','test']).T
    #    accuracy = accuracy_score(y_test,mlp_model.predict(X_test))
    #    precision = precision_score(y_test,mlp_model.predict(X_test))
    #    prf = precision_recall_fscore_support(y_test,mlp_model.predict(X_test))
        # ------------ 模型测试 ----------------
        
        
        # Model persistence
        mlp = MLPClassifier((100,100),activation = 'logistic')
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
df.to_excel('models.xlsx')