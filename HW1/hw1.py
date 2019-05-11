# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:01:24 2019

@author: JasonHuang
"""
import csv
path = "https://github.com/thechaudharysab/Chipotle-data_analysis-example/blob/master/chipotle.tsv"
chipo = pd.read_csv(path, sep = '\t')
