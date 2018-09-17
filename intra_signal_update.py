# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 13:54:27 2018

@author: ldh

此脚本更新日内模型预测信号。
"""

# intra_signal_update.py
import os
import datetime as dt
import yaml
from sqlalchemy.types import VARCHAR,DATETIME,DECIMAL,INT
from sklearn.externals import joblib
import pandas as pd
from podaci.guosen.data import (get_trade_calendar,
                                get_data,
                                save_into_db,
                                execute_session,
                                get_stock_features)
                                

etc_path = os.path.join(os.path.split(__file__)[0],'etc.yaml')

with open(etc_path,'r') as f:
    etc = yaml.load(f)
    
model_path = etc['model_path']

model_codes = pd.read_excel('models.xlsx',dtype = {'model_code':str})
model_codes = model_codes['model_code'].tolist()

today = dt.datetime.today()
today_str = today.strftime('%Y%m%d')
trade_calendar = get_trade_calendar('20150101',(today + dt.timedelta(days = 60)).strftime('%Y%m%d'))

SQL_LAST_TARGET_DATE = '''
SELECT MAX(target_date)
FROM LDH_pattern_pred_minute
'''
last_target_date = get_data(SQL_LAST_TARGET_DATE,'gb').dropna()
if len(last_target_date) == 0:
    last_target_date = '20160104'
else:
    last_target_date = last_target_date.values[0][0]
    
end_target_date = trade_calendar.loc[trade_calendar.loc[trade_calendar['trade_date'] == \
                                                        today_str].index.values[0] + 1]['trade_date']

feature_start_date = (dt.datetime.strptime(last_target_date,'%Y%m%d') - \
                      dt.timedelta(days = 120)).strftime('%Y%m%d')


SQL_GET_FEATURES = '''
SELECT 
[stock_code]
,[trade_date]
,[b]
,[c]
,[u]
,[l]
,[ol]
,[cl]
,[ac]
FROM [LDH_features]
WHERE trade_date >= '{start_date}'
AND trade_date <= '{end_date}'
'''
features = get_data(SQL_GET_FEATURES.format(start_date = feature_start_date,
                                   end_date = end_target_date),'gb')

col_names = ['b','c','u','l','ol','cl','ac']

signals = pd.DataFrame()

for stk in model_codes:
    # 调取模型
    model = joblib.load(os.path.join(model_path,'%s.pkl'%stk))
    
    # 样本获取与处理
    stock_features = features.loc[features['stock_code'] == stk]
    stock_features = stock_features.sort_values('trade_date',ascending = True)
    
    for i in range(1,30):
        for col in col_names:
            stock_features['%s_%s'%(col,i)] = stock_features[col].shift(i)
    stock_features.dropna(inplace = True)
    stock_features = trade_calendar.join(stock_features.set_index('trade_date'),on = 'trade_date',
                                         how = 'left')
    stock_features['target_date'] = stock_features['trade_date'].shift(-1)
    stock_features.dropna(inplace = True)
    target_date_list = stock_features['target_date'].tolist()

    X = stock_features.drop(['trade_date','stock_code','target_date'],axis = 1).values
    
    if len(X) < 1:
        # 样本数据宽度数量不足扩大取数据的宽度
        stock_features = get_stock_features(start_date = '20100101',end_date = today_str,
                                            stock_universe = [stk])
        stock_features = stock_features.sort_values('trade_date',ascending = True)
    
        for i in range(1,30):
            for col in col_names:
                stock_features['%s_%s'%(col,i)] = stock_features[col].shift(i)
        stock_features.dropna(inplace = True)
        stock_features = trade_calendar.join(stock_features.set_index('trade_date'),on = 'trade_date',
                                             how = 'left')
        stock_features['target_date'] = stock_features['trade_date'].shift(-1)
        stock_features.dropna(inplace = True)
        target_date_list = stock_features['target_date'].tolist()

        X = stock_features.drop(['trade_date','stock_code','target_date'],axis = 1).values
        if len(X) < 1:
            continue # 放弃更新此股票信号
            
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    y_max_proba = y_proba.max(axis = 1)
    
    stock_label = pd.DataFrame([target_date_list,
                                y_pred.tolist(),
                                y_max_proba.tolist()],index = ['target_date','signal_type','signal_prob']).T
    stock_label['stock_code'] = stk
    stock_label['pred_dt'] = today
    signals = pd.concat([signals,stock_label])
    print '%s stock finished'%stk
    
dtype_dict = {'stock_code':VARCHAR(10),
              'target_date':VARCHAR(16),
              'pred_dt':DATETIME,
              'signal_type':INT, # (0为震荡,1为趋势)
              'signal_prob':DECIMAL(10,6)}
save_into_db(signals,'LDH_pattern_pred_minute_source',dtype_dict,'gb','replace')
SQL_MERGE_SIGNALS = '''
MERGE INTO LDH_pattern_pred_minute as T
USING LDH_pattern_pred_minute_source as S
ON (T.target_date = S.target_date AND T.stock_code = S.stock_code)
WHEN NOT MATCHED BY TARGET
THEN INSERT 
(stock_code,target_date,pred_dt,signal_type,signal_prob)
VALUES
(S.stock_code,S.target_date,S.pred_dt,S.signal_type,S.signal_prob);
'''
execute_session(SQL_MERGE_SIGNALS,'gb')