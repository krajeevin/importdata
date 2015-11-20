import pandas as pd
import datetime
import pandas.io.data as web
import matplotlib.pyplot as plt

from matplotlib import style

style.use('fivethirtyeight')

#str1=raw_input("Enter the date as From(DD-MM-YYYY) :")
#str1=int("Enter the year")
int1=input("Enter the  1 for Share :")
if int1 == 1:
        str4="^NSEI"
else:
        str4=raw_input("Enter the Share Name :")
        str4=str4+".NS"


#str2=raw_input("Enter the End Date (DD-MM-YYYY) : ")

#str2=raw_input("Enter the End Date (DD-MM-YYYY) : ")

#var1d=str1[0:2]
#var1m=str1[3:5]
#var1y=str1[6:10]
#datetime1=var1y + ", " +var1m + ", " + var1d
#print str4
#print datetime1
#int1y=var1y
#int1m=var1m
#int1d=var1d
#print(var1d)
#print(var1m)
#print(var1y)
#start = datetime.date(datetime1)
start = datetime.datetime(2014, 10, 1)

end = datetime.datetime.now()
#end = datetime.date.now()


#start = datetime.datetime(datetime, 10, 1)

df = web.DataReader(str4 , "yahoo", start, end)
df['4MA'] = pd.rolling_mean(df['Close'], 4)
df['4dMoving'] = df['4MA']
print(df)
print(df.describe())
#df['Open'].plot()
#df['High'].plot()
#df['Low'].plot()
df['Close'].plot()
df['4dMoving'].plot()

plt.legend()
#plt._interactive_bk
plt.show()


