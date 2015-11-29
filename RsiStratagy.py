import datetime
import numpy as np
import matplotlib.colors as colors
import matplotlib.finance as finance
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas as pd
import pandas.io.data as web

from nsepy.archives import get_price_history
from datetime import date
startdate = datetime.date(2015, 6, 1)
today = enddate = datetime.date.today()
start=startdate
end=enddate

def relative_strength(prices, n=14):

    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n - 1) + upval)/n
        down = (down*(n - 1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi
def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a
share = pd.read_csv('ind_nifty50list.csv', parse_dates=True)
ints1=0
columns = ['Share', 'RSI','50DMA','Close','MRsi5Day']
#columns = ['Share', 'RSI']
dfnew = pd.DataFrame(columns=columns)


while (ints1 <100) :
            sharen=share.at[ints1,'Symbol']
            ints1 = ints1 + 1
            strs = sharen
            df=get_price_history(stock = strs,start = start,end = enddate)
            r=df

            r.sort()
            prices = r['Close']
            rsi = relative_strength(prices)

            ma50 = moving_average(prices, 50, type='simple')
            ints=np.size(rsi)
            dfrsi = pd.DataFrame(rsi)
            x = np.asarray(rsi)
            Moving=np.asarray(ma50)
            dfnew.loc[len(dfnew)+1]=[strs, x[ints-1 ],Moving[ints-1 ],prices[ints-1], dfrsi.tail().max() ]
            if ints1>30:
                break



else:
            print "Good bye!"

print dfnew
dfnew.to_csv('RsiFirst.csv', index=True)

print "Finished bye!"