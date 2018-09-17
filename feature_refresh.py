# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 14:56:00 2018

@author: ldh

此脚本负责增量更新k线特征到数据库.
"""

# feature_refresh.py
import datetime as dt
import pandas as pd
from sqlalchemy.types import VARCHAR,DECIMAL,DATETIME
from podaci.guosen.data import (save_into_db,execute_session,
                                get_data,get_stock_daily_data)
from utils import stock_feature_handler

model_codes = pd.read_excel('models.xlsx',dtype = {'model_code':str})
model_codes = model_codes['model_code'].tolist()

# 确定上次特征更新日期
sql = '''
SELECT MAX(trade_date)
FROM LDH_features
'''
last_update = get_data(sql,'gb')
last_update = last_update.values[0][0]

if last_update is None:
    last_update = '20050101'
# 获取增量更新数据
start_date = last_update
end_date = dt.datetime.today().strftime('%Y%m%d')

if start_date == end_date:
    exit()
    
stock_data = get_stock_daily_data(start_date,end_date,model_codes) # 目标股票池所有股票数据

if len(stock_data) == 0:
    exit()
    
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
save_into_db(stock_features,'LDH_features_source',dtype_dict,'gb',if_exists = 'replace')
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
    