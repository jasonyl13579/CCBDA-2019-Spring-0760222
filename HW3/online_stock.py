# -*- coding: utf-8 -*-
"""
Created on Fri May  3 23:22:12 2019

@author: Mintlab
"""
import unirest

response = unirest.get("https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/get-histories?region=US&lang=en&symbol=INTC&from=1399046400&to=1556812800&events=div&interval=1d",
  headers={
    "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com",
    "X-RapidAPI-Key": "013cb4bdcdmsh320c96ba564b804p15d41djsn355f71ed0fc0"
  }
)

print (response.body)