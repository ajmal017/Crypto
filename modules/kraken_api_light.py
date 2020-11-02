import datetime as datetime
import time
import numpy as np
import pandas as pd
import warnings
import talib
import krakenex

warnings.filterwarnings("ignore")


class KrakenLight():
    
    def __init__(self, pair, api_key_file, minimum_fund, percent_alloc, sleepage):
        
        self.pair = pair
        self.api_key_file = api_key_file
        self.minimum_fund = minimum_fund 
        self.k =  krakenex.API()
        self.k.load_key(self.api_key_file)
        self.balance = self.k.query_private('Balance')['result']
        self.open_orders = self.k.query_private('OpenOrders')['result']['open']
        self.assets = list(self.balance.keys())
        self.tickers = {'BTC/EUR':'XXBT','ETH/EUR':'XETH','DASH/EUR':'XDASH',
                        'XRP/EUR':'XXRP','DOT/EUR':'DOT','BAT/EUR':'BAT','ADA/EUR':'ADA',
                        'SC/EUR':'SC','ETC/EUR':'ETC'}
        self.percent_alloc = percent_alloc
        self.sleepage = sleepage
        self.fund = float(self.balance['ZEUR'])
        self.buying_power = self.fund * self.percent_alloc
        try:
            self.crypto_value = float(self.balance[self.tickers[self.pair]])
        except:
            pass

    
    def live_pattern(self, interval_5min, bars_5min, period_ema_5min, 
                           interval_hourly, bars_hourly, period_ssl,quantiles):
        
        while True:
            
            print ('------------------------------------------------------------------------')

            df_5min = self.ohlcv(interval_5min, bars_5min)
            df_hourly = self.ohlcv(interval_hourly, bars_hourly)

            df_ema = self.ema(df_5min, period_ema_5min)
            df_ssl = self.ssl(df_hourly, period_ssl)

            #df_ema = self.average_bar(30, df_ema)

            print (df_ssl.index[-1],'hourly candle in progress')
            print ('starting',df_ema.index[-1],'5min candle')
            print ('5 min opening price',df_ema.close[-1])

            #################################################################################
            self.research_pattern_ssl(df_ema, df_ssl)
            #################################################################################

            last = datetime.datetime.strptime(df_ema.index[-1], '%Y-%m-%d %H:%M:%S').timestamp()
            next_time = last + (interval_5min*60)
            sleep = float(next_time) - time.time() + self.sleepage

            time.sleep(sleep)
                
    def ohlcv(self, interval, bars):

        tod = datetime.datetime.today()
        since = tod - datetime.timedelta(hours=0, minutes=interval * bars)
        since = since.timestamp()

        ret = self.k.query_public(method='OHLC', data = {'pair': self.pair, 'since': since, 'interval': interval})

        array = ret['result'][self.pair]

        df = pd.DataFrame(data = array, columns = ['time','open','high','low','close','vwap','volume','count'])
        df = df.astype('float')
        df['time'] = df['time'].apply(lambda x:datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        
        df.set_index('time', inplace =True)
        df.dropna(inplace=True)

        return df
    
    def fibonacci_retracement(self, df):
        
        
        diff = max(df.high) - min(df.low)
        level0 = max(df.high) - 0.263 * diff
        level1 = max(df.high) - 0.382 * diff
        level2 = max(df.high) - 0.500 * diff
        level3 = max(df.high) - 0.618 * diff
        level4 = max(df.high) - 0.786 * diff
        
        
        return max(df.high), level0, level1, level2, level3, level4, min(df.low)
        
    def ssl(self, df_ssl, period_ssl):

        #function ssl
        low = df_ssl.low.values
        high = df_ssl.high.values

        df_ssl['sma_low'] = talib.SMA(low, timeperiod=period_ssl)
        df_ssl['sma_high'] = talib.SMA(high, timeperiod=period_ssl)

        df_ssl['green'] = pd.Series()
        df_ssl['red'] = pd.Series()

        for t in df_ssl.index:

            close = df_ssl.close.loc[t]
            sma_high = df_ssl.sma_high.loc[t]
            sma_low = df_ssl.sma_low.loc[t]
            try:
                if (close < sma_low):
                    df_ssl.loc[slice(t,df_ssl.index[-1]),:]['red'] = df_ssl.loc[slice(t,df_ssl.index[-1]),:]['sma_high']
                    df_ssl.loc[slice(t,df_ssl.index[-1]),:]['green'] = df_ssl.loc[slice(t,df_ssl.index[-1]),:]['sma_low']

                elif (close > sma_high):
                    df_ssl.loc[slice(t,df_ssl.index[-1]),:]['red'] = df_ssl.loc[slice(t,df_ssl.index[-1]),:]['sma_low']
                    df_ssl.loc[slice(t,df_ssl.index[-1]),:]['green'] = df_ssl.loc[slice(t,df_ssl.index[-1]),:]['sma_high']
            except:
                pass
        return df_ssl
    
    def ema(self, df, period):

        ema = talib.EMA(df.close, timeperiod=period)

        df['ema'] = ema
        df['ema_to_close']  = (ema - df.close.values)/df.close.values 
        
        return df
      
    def sma(self, df, period):

        sma = talib.SMA(df.close, timeperiod=period)

        df['sma'] = sma
        df['sma_to_close']  = (sma - df.close.values)/df.close.values 
        
        return df    
        
    def stochastic(self, df, fastk_period, slowk_period, slowd_period):
        
        df['slowk'],df['slowd'] = talib.STOCH(df.high, df.low, df.close, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=0, slowd_period=slowd_period, slowd_matype=0)
        
        return df
    
    def wma(self, df, period):
        
        df['wma'] = talib.WMA(df.close, timeperiod=period)
        
        return df
    
    def research_pattern_ssl(self, df_ema, df_ssl):
        
        maxx, level0, level1, level2, level3, level4, minn = self.fibonacci_retracement(df_ssl)
        bull_zone = (df_ssl.close[-1] < level4)
        bear_zone = (df_ssl.close[-1] > level0)
      
        ######################################################################################
        
        ssl_trail_bull = ((df_ssl.green[-2] > df_ssl.red[-2]) and (df_ssl.green[-3] < df_ssl.red[-3]) and (df_ssl.green[-4] < df_ssl.red[-4]) and (df_ssl.green[-5] < df_ssl.red[-5]) and (df_ssl.green[-6] < df_ssl.red[-6]) and (df_ssl.green[-7] < df_ssl.red[-7]))
        
        ssl_trail_bear = ((df_ssl.green[-2] < df_ssl.red[-2]) and (df_ssl.green[-3] > df_ssl.red[-3]) and (df_ssl.green[-4] > df_ssl.red[-4]) and (df_ssl.green[-5] > df_ssl.red[-5]) and (df_ssl.green[-6] > df_ssl.red[-6]) and (df_ssl.green[-7] > df_ssl.red[-7]))
        

        ######################################################################################
        
        if ssl_trail_bull and bull_zone:
            
            print ('we are in bullish pattern...its a buy')
            print ('-------------------------------------------------')
            print ('price: {}'.format(df_ema.loc[df_ema.index[-1],'close']))
            print ('-------------------------------------------------')
            print (self.buying_power,'euros to buy')
            volume = self.buying_power/df_ema.loc[df_ema.index[-1],'close']
            
            if self.buying_power > self.minimum_fund:
    
                response = self.k.query_private('AddOrder',
                                                {'pair': self.pair,
                                                 'type': 'buy',
                                                 'ordertype': 'market',
                                                 'volume': volume,
                                                 'close[ordertype]': 'stop-loss-profit',
                                                 'close[price]': 0.98 * df_ema.loc[df_ema.index[-1],'close'],
                                                 'close[price2]':1.015 * df_ema.loc[df_ema.index[-1],'close']
                                                })
                print (response)
            else:
                print ('insufficient funds to buy')
                
        elif ssl_trail_bear and bear_zone:
            
            print ('we are in bearish pattern...its a sell')
            print ('-------------------------------------------------')
            print ('price: {}'.format(df_ema.loc[df_ema.index[-1],'close']))
            print ('-------------------------------------------------')
            
            try:
                print (self.crypto_value,self.tickers[self.pair],'to sell')

                if self.crypto_value * df_ema.close[-1] > self.minimum_fund:

                    response = self.k.query_private('AddOrder',
                                                    {'pair': self.pair,
                                                     'type': 'sell',
                                                     'ordertype': 'market',
                                                     'volume': self.crypto_value
                                                    })
                    print (response)
                else:
                    print ('insufficient funds to sell')
            except:
                print ('the asset is not in the balance')
        else:
            print ('pattern not met')    

        
        
        
        
        
        