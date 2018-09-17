# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:33:45 2018

@author: ldh
该脚本将最新数据刷入40.
"""

# update_latest_signal_to40.py
from sqlalchemy.types import VARCHAR,DATETIME,INT,DECIMAL
from podaci.guosen.data import get_data,save_into_db,execute_session

#%% 日内
SQL_GET_LAST_SIGNAL_MINUTE = '''
SELECT 
a.[stock_code]
,a.[target_date]
,[pred_dt]
,[signal_type]
,[signal_prob]
FROM LDH_pattern_pred_minute a
RIGHT JOIN (
SELECT 
max(target_date) as target_date,
stock_code 
FROM LDH_pattern_pred_minute
GROUP BY stock_code) b
ON a.target_date = b.target_date AND a.stock_code = b.stock_code
'''
intra_last_signal = get_data(SQL_GET_LAST_SIGNAL_MINUTE,'gb')
dtype_dict = {'stock_code':VARCHAR(10),
              'target_date':VARCHAR(16),
              'pred_dt':DATETIME,
              'signal_type':INT, # (0为震荡,1为趋势)
              'signal_prob':DECIMAL(10,6)}
save_into_db(intra_last_signal,'Yi_mgt_pattern_pred_minute_source',dtype_dict,'xiaoyi40',
             'replace')
SQL_MERGE_SIGNAL_MINUTE = '''
MERGE INTO Yi_mgt_pattern_pred_minute as T
USING Yi_mgt_pattern_pred_minute_source as S
ON (T.target_date = S.target_date AND T.stock_code = S.stock_code)
WHEN NOT MATCHED BY TARGET
THEN INSERT 
(stock_code,target_date,pred_dt,signal_type,signal_prob)
VALUES
(S.stock_code,S.target_date,S.pred_dt,S.signal_type,S.signal_prob)
WHEN NOT MATCHED BY SOURCE
THEN DELETE;
'''
execute_session(SQL_MERGE_SIGNAL_MINUTE,'xiaoyi40')


#%% 日间
SQL_GET_LAST_SIGNAL_DAILY = '''
SELECT 
a.[stock_code]
,a.[target_date]
,[pred_dt]
,[signal_type]
,[signal_prob]
FROM LDH_pattern_pred_daily a
RIGHT JOIN (
SELECT 
max(target_date) as target_date,
stock_code 
FROM LDH_pattern_pred_daily
GROUP BY stock_code) b
ON a.target_date = b.target_date AND a.stock_code = b.stock_code
'''
daily_last_signal = get_data(SQL_GET_LAST_SIGNAL_DAILY,'gb')
dtype_dict = {'stock_code':VARCHAR(10),
              'target_date':VARCHAR(16),
              'pred_dt':DATETIME,
              'signal_type':INT, # (0为震荡,1为趋势)
              'signal_prob':DECIMAL(10,6)}
save_into_db(daily_last_signal,'Yi_mgt_pattern_pred_daily_source',dtype_dict,'xiaoyi40',
             'replace')
SQL_MERGE_SIGNAL_DAILY = '''
MERGE INTO Yi_mgt_pattern_pred_daily as T
USING Yi_mgt_pattern_pred_daily_source as S
ON (T.target_date = S.target_date AND T.stock_code = S.stock_code)
WHEN NOT MATCHED BY TARGET
THEN INSERT 
(stock_code,target_date,pred_dt,signal_type,signal_prob)
VALUES
(S.stock_code,S.target_date,S.pred_dt,S.signal_type,S.signal_prob)
WHEN NOT MATCHED BY SOURCE
THEN DELETE;
'''
execute_session(SQL_MERGE_SIGNAL_DAILY,'xiaoyi40')
