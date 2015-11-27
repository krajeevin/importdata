import datetime
import numpy as np
import matplotlib.colors as colors
import matplotlib.finance as finance
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas.io.data as web


import pandas as pd

startdate = datetime.date(2014, 1, 1)
today=enddate = datetime.date.today()

ticker = "Nifty"
##str5 = "Nifty"
##print str5
##int4 =raw_input("Enter the index Name(For Niftty :1,Bank Nifty2 ) :")
##if int4 == 1:
 ##   str5 ="^NSEI"

##elif int4 == 2:
 ##   str5 ="^NSEBANK"
#fh=web.DataReader("^NSEI","yahoo", start , end)
fh = finance.fetch_historical_yahoo("^NSEI" , startdate, enddate)
# a numpy record array with fields: date, open, high, low, close, volume, adj_close)





# new=pd.DataFrame({'Open':fh[:,0],'High':fh[:,1],'Low':fh[:,2],'Close':fh[:,3],'volume':fh[:,4]})
#new=pd.DataFrame(r)

r = mlab.csv2rec(fh)
fh.close()
r.sort()


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


def relative_strength(prices, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """

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


def moving_average_convergence(x, nslow=26, nfast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = moving_average(x, nslow, type='exponential')
    emafast = moving_average(x, nfast, type='exponential')
    return emaslow, emafast, emafast - emaslow


plt.rc('axes', grid=True)
plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

textsize = 9
left, width = 0.1, 0.8
rect1 = [left, 0.7, width, 0.2]
rect2 = [left, 0.3, width, 0.4]
rect3 = [left, 0.1, width, 0.2]


fig = plt.figure(facecolor='yellow')
axescolor = '#f6f6f6'  # the axes background color

ax1 = fig.add_axes(rect1, axisbg=axescolor)  # left, bottom, width, height
ax2 = fig.add_axes(rect2, axisbg=axescolor, sharex=ax1)
ax2t = ax2.twinx()
ax3 = fig.add_axes(rect3, axisbg=axescolor, sharex=ax1)


# plot the relative strength indicator
prices = r.adj_close
rsi = relative_strength(prices)
fillcolor = 'darkgoldenrod'

ax1.plot(r.date, rsi, color=fillcolor)
ax1.axhline(70, color=fillcolor)
ax1.axhline(30, color=fillcolor)
ax1.fill_between(r.date, rsi, 70, where=(rsi >= 70), facecolor=fillcolor, edgecolor=fillcolor)
ax1.fill_between(r.date, rsi, 30, where=(rsi <= 30), facecolor=fillcolor, edgecolor=fillcolor)
ax1.text(0.6, 0.9, '>70 = overbought', va='top', transform=ax1.transAxes, fontsize=textsize)
ax1.text(0.6, 0.1, '<30 = oversold', transform=ax1.transAxes, fontsize=textsize)
ax1.set_ylim(0, 100)
ax1.set_yticks([30, 70])
ax1.text(0.025, 0.95, 'RSI (14)', va='top', transform=ax1.transAxes, fontsize=textsize)
ax1.set_title('%s daily' % ticker)

# plot the price and volume data
dx = r.adj_close - r.close
low = r.low + dx
high = r.high + dx

deltas = np.zeros_like(prices)
deltas[1:] = np.diff(prices)
up = deltas > 0
ax2.vlines(r.date[up], low[up], high[up], color='black', label='_nolegend_')
ax2.vlines(r.date[~up], low[~up], high[~up], color='black', label='_nolegend_')
ma4 = moving_average(prices, 4, type='simple')
ma50 = moving_average(prices, 50, type='simple')
ma200 = moving_average(prices, 200, type='simple')
# ma4 = moving_average(prices, 4, type='simple')

linema4, = ax2.plot(r.date, ma4, color='yellow', lw=2, label='MA (4)')
linema50, = ax2.plot(r.date, ma50, color='blue', lw=2, label='MA (50)')
linema200, = ax2.plot(r.date, ma200, color='red', lw=2, label='MA (200)')


last = r[-1]
s = '%s O:%1.2f H:%1.2f L:%1.2f C:%1.2f, V:%1.1fM Chg:%+1.2f' % (
    today.strftime('%d-%b-%Y'),
    # Want to use actual day time of

    last.open, last.high,
    last.low, last.close,
    last.volume*1e-6,
    last.close - last.open)

t4 = ax2.text(0.3, 0.9, s, transform=ax2.transAxes, fontsize=textsize)

props = font_manager.FontProperties(size=10)
leg = ax2.legend(loc='center left', shadow=True, fancybox=True, prop=props)
leg.get_frame().set_alpha(0.5)


volume = (r.close*r.volume)/1e6  # dollar volume in millions
vmax = volume.max()
poly = ax2t.fill_between(r.date, volume, 0, label='Volume', facecolor=fillcolor, edgecolor=fillcolor)
ax2t.set_ylim(0, 5*vmax)
ax2t.set_yticks([])


# compute the MACD indicator
fillcolor = 'darkslategrey'
nslow = 26
nfast = 12
nema = 9
emaslow, emafast, macd = moving_average_convergence(prices, nslow=nslow, nfast=nfast)
ema9 = moving_average(macd, nema, type='exponential')
ax3.plot(r.date, macd, color='black', lw=2)
ax3.plot(r.date, ema9, color='blue', lw=1)
ax3.fill_between(r.date, macd - ema9, 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)


ax3.text(0.025, 0.95, 'MACD (%d, %d, %d)' % (nfast, nslow, nema), va='top',
         transform=ax3.transAxes, fontsize=textsize)

#ax3.set_yticks([])
# turn off upper axis tick labels, rotate the lower ones, etc
for ax in ax1, ax2, ax2t, ax3:
    if ax != ax3:
        for label in ax.get_xticklabels():
            label.set_visible(False)
    else:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')

    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')


class MyLocator(mticker.MaxNLocator):
    def __init__(self, *args, **kwargs):
        mticker.MaxNLocator.__init__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return mticker.MaxNLocator.__call__(self, *args, **kwargs)

# at most 5 ticks, pruning the upper and lower so they don't overlap
# with other ticks
#ax2.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))
#ax3.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))

ax2.yaxis.set_major_locator(MyLocator(5, prune='both'))
ax3.yaxis.set_major_locator(MyLocator(5, prune='both'))



#For detail data analysing and printing
new = web.DataReader("^NSEI","yahoo", startdate, enddate)

#new = pd.read_csv('D:/share/Nifty2015Graph.csv', parse_dates=True)
#dfnew=  pd.read_csv('Nifty2015New.csv', parse_dates=True)

#dfnew = dfnew.drop_duplicates().fillna(0)
#dfold= dfold.drop_duplicates().fillna(0)

#new= pd.concat([dfold,dfnew.tail(50)])
new['4DayMA'] = pd.stats.moments.rolling_mean(new['Open'], 4)
new['50DayMA'] = pd.stats.moments.rolling_mean(new['Open'], 50)

new.tail(300)
new['300DayHigh'] = new['High'].tail(300).max()
new['300DayLow'] = new['Low'].tail(300).min()
int1 = new['300DayHigh']-new['300DayLow']
int1=int1*61.8/100
new['up300  61.8%']=new['300DayLow']+int1
new['Down300 61.8%']=new['300DayHigh']-int1

new.tail(50)
new['50DayHigh'] = new['High'].tail(50).max()
new['50DayLow'] = new['Low'].tail(50).min()
int1 = new['50DayHigh']-new['50DayLow']
int1=int1*61.8/100
new['up50 61.8%']=new['50DayLow']+int1
new['Down50 61.8%']=new['50DayHigh']-int1

new.tail(20)
new['20DayHigh'] = new['High'].tail(20).max()
new['20DayLow'] = new['Low'].tail(20).min()
int1 = new['20DayHigh']-new['20DayLow']
int1=int1*61.8/100
new['20Day 61.8%']=new['20DayLow']+int1
new['20Down 61.8%']=new['20DayHigh']-int1

new.tail(1)
transpose = new.tail(1).T
print transpose


##top = plt.subplot2grid((8,8), (0, 0), rowspan=6, colspan=8)
##top.plot(new1.index, new1["Close"])
##top.plot(new1.index, new1['50DayMA'])

#bottom = plt.subplot2grid((8,8), (6,0), rowspan=2, colspan=8)
#bottom.bar(new.index, new['Volume'])
#plt.gcf().set_size_inches(15,8)



#for RSI Printing
ints=np.size(rsi)
x = np.asarray(rsi)
print  x[ints-1]



plt.show()