# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 15:02:46 2018

@author: ldh

主控程序。控制脚本运行。
"""

# main.py

import os

# 更新k线
os.system('python feature_refresh.py')

# 更新信号
os.system('python intra_signal_update.py')
os.system('python daily_signal_update.py')
os.system('python update_latest_signal_to40.py')