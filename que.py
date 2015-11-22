
import matplotlib.pyplot as plt
import datetime
import pandas.io.data as web
import pandas as pd
dfold = pd.read_csv('Nifty2015.csv', index_col = 'Date', parse_dates=True)
dfnew=  pd.read_csv('Nifty2015New.csv', index_col = 'Date', parse_dates=True)
dfnew = dfnew.drop_duplicates().fillna(0)
dfold=dfold.drop_duplicates().fillna(0)
print dfnew.tail(5)
print dfold.tail(5)
#dfupdate=pd.merge(dfold, dfnew, on='Date') # Pankaj: This won't work because you have declared Date column as index while reading file. 
dfupdate=pd.merge(dfold, dfnew, left_index=True, right_index=True) # Here I am telling to join on index column from both tables.
dfupdate.to_csv('NiftyFirst.csv', index=True)

