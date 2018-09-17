# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 15:58:21 2018

@author: ldh
"""

# utils.py

import pandas as pd

def feature_handler(row):
    '''
    DataFrame格式K线特征处理函数。
    
    Notes
    ----------
    必须列名: open_price,high_price,low_price,close_price,trade_date,prev_close_price,prev_amount
    '''
    body =  (row['close_price'] - row['open_price']) / row['open_price']
    color = 1 if row['close_price'] >= row['open_price'] else 0
    upper = (row['high_price'] - max(row['close_price'],row['open_price'])) / max(row['close_price'],row['open_price'])
    lower = (min(row['close_price'],row['open_price']) - row['low_price'] ) / min(row['close_price'],row['open_price'])
    open_level = (row['open_price'] - row['prev_close_price']) / row['prev_close_price']
    close_level = (row['close_price'] - row['prev_close_price']) / row['prev_close_price']
    amount_change = (row['amount'] - row['prev_amount']) / row['prev_amount']
    return [row['trade_date'],body,color,upper,lower,open_level,close_level,amount_change]

def stock_feature_handler(df):
    df_ = df.sort_values('trade_date',ascending = True)
    df_['prev_close_price'] = df_['close_price'].shift(1)
    df_['prev_amount'] = df_['amount'].shift(1)
    df_.dropna(inplace = True)
    
    features = []
    
    for idx,row in df_.iterrows():
        features.append(feature_handler(row))
        
    return pd.DataFrame(features,columns = ['trade_date','b','c','u','l','ol','cl','ac'])
    