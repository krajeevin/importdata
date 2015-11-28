from nsepy.archives import get_price_history
from nsepy import indices
from datetime import date
#Stock price history
sbin = get_price_history(stock = 'SBIN',
                        start = date(2015,1,1), 
                        end = date(2015,1,10))
sbin[[ 'VWAP', 'Turnover']].plot(secondary_y='Turnover')
#Index price history
nifty = indices.archives.get_price_history(index = "NIFTY 50", 
                                            start = date(2015,9,1), 
                                            end = date(2015,9,24))
nifty[['Close', 'Turnover']].plot(secondary_y='Turnover')
#Index P/E ratio history
nifty_pe = indices.archives.get_pe_history(index = "NIFTY 50", 
                                            start = date(2015,9,1), 
                                            end = date(2015,9,24))
nifty_pe['Index'] = nifty['Close']
nifty_pe[['Index', 'P/E']].plot(secondary_y='P/E')