import krakenex
import datetime as datetime
import decimal
import time
import numpy as np
import pandas as pd
import talib
import warnings
from findiff import FinDiff 
from .trendline import TrendLine

warnings.filterwarnings("ignore")


class Kraken():
    
    def __init__(self, pair, api_key_file, minimum_fund):
        
        self.pair = pair
        self.api_key_file = api_key_file
        self.minimum_fund = minimum_fund 
        self.k =  krakenex.API()
        self.k.load_key(self.api_key_file)
        self.balance = self.k.query_private('Balance')['result']
        self.open_orders = self.k.query_private('OpenOrders')['result']['open']
        #self.trades = self.k.query_private('TradesHistory')
        self.assets = list(self.balance.keys())
        self.tickers = {'BTC/EUR':'XXBTZEUR','ETH/EUR':'XETHZEUR','DASH/EUR':'XDASHZEUR','XRP/EUR':'XXRPZEUR'}
        self.fund = float(self.balance['ZEUR'])    
    
    def live_pattern(self, interval_ssl, bars_ssl, period_ssl, 
                                     interval_ema, bars_ema, period_ema):
        
        while True:
            
            print ('------------------------------------------------------------------------')
            df_ssl = self.ohlcv(interval_ssl,bars_ssl)
            df_ema = self.ohlcv(interval_ema,bars_ema)
            
            #function ssl
            df_ssl = self.ssl(period_ssl, df_ssl)
            df_ema = self.ema(period_ema, df_ema)
            #df_ema = self.average_bar(30, df_ema)
            
            print ('starting',df_ema.index[-1],'5min candle')
            print (df_ssl.index[-1],'hourly candle in progress')
            #################################################################################
            #self.engulfing_pattern(df_ssl, df_ema)
            #self.close_above_high_pattern(df_ssl, df_ema)
            #self.hammer_pattern(df_ssl, df_ema)
            #self.three_bar_pattern(df_ssl, df_ema)
            self.pivot_pattern(df_ema,df_ssl)
            #################################################################################

            last = datetime.datetime.strptime(df_ema.index[-1], '%Y-%m-%d %H:%M:%S').timestamp()
            next_time = last + (interval_ema*60)
            sleep = float(next_time) - time.time()

            print (np.round(sleep/60,2), 'minutes sleep time')

            time.sleep(sleep)
            
            
    def historical_data(self, interval_ssl, bars_ssl, period_ssl, 
                           interval_ema, bars_ema, period_ema):
        
        df_ssl = self.ohlcv(interval_ssl,bars_ssl)
        df_ema = self.ohlcv(interval_ema,bars_ema)

        #function ssl
        df_ssl = self.ssl(period_ssl, df_ssl)
        df_ema = self.ema(period_ema, df_ema)
        df_ema = self.average_bar(30, df_ema)

        return df_ssl, df_ema
            
    def engulfing_pattern(self, ssl, ema):
        
        if ssl.iloc[-2]['green'] > ssl.iloc[-2]['red']:
      
            print ('we are in ssl uptrend...')

            green_ema50 = (ema.loc[ema.index[-3],'ema50_to_close'] > 0)

            green_candle_bull = (ema.loc[ema.index[-2],'candle'] == 1)
            red_candle_bull = (ema.loc[ema.index[-3],'candle'] == 0)

            engulfing_red_bull = (ema.loc[ema.index[-2],'open'] < ema.loc[ema.index[-3],'close'])
            engulfing_green_bull = (ema.loc[ema.index[-3],'open'] < ema.loc[ema.index[-2],'close'])

            if  green_candle_bull and red_candle_bull and engulfing_green_bull and engulfing_red_bull and green_ema50:

                print ('we are in bullish engulfing pattern...its a buy')
                print ('price: {}'.format(ema.loc[ema.index[-1],'close']))

            else:
                 print ('bullish engulfing pattern not met')
            
        elif ssl.iloc[-2]['green'] < ssl.iloc[-2]['red']:
                
            print ('we are in ssl downtrend...')

            red_ema50 = (ema.loc[ema.index[-3],'ema50_to_close'] < 0)

            green_candle_bear = (ema.loc[ema.index[-3],'candle'] == 1)
            red_candle_bear = (ema.loc[ema.index[-2],'candle'] == 0)

            engulfing_red_bull = (ema.loc[ema.index[-2],'open'] > ema.loc[ema.index[-3],'close'])
            engulfing_green_bull = (ema.loc[ema.index[-3],'open'] > ema.loc[ema.index[-2],'close'])

            if green_candle_bear and red_candle_bear and engulfing_red_bull and engulfing_green_bull and red_ema50:

                print ('we are in bearish engulfing pattern...its a sell')
                print ('price: {}'.format(ema.loc[ema.index[-1],'close']))

            else:
                print ('bearish engulfing pattern not met')
        
    def close_above_high_pattern(self, ssl, ema):
        
        if ssl.iloc[-2]['green'] > ssl.iloc[-2]['red']:
      
            print ('we are in ssl uptrend...')

            green_ema50_second = (ema.loc[ema.index[-3],'ema50_to_close'] > 0)

            green_candle_bull = (ema.loc[ema.index[-2],'candle'] == 1)
            close_above_high = ema.loc[ema.index[-2],'close'] > ema.loc[ema.index[-3],'high']

            if green_ema50_second and green_candle_bull and close_above_high:

                print ('we are in close above high pattern...its a buy')
                print ('price: {}'.format(ema.loc[ema.index[-1],'close']))
                
            else:
                print ('close above high pattern not met')    
        
        elif ssl.iloc[-2]['green'] < ssl.iloc[-2]['red']:
                
            print ('we are in ssl downtrend...')
            
            red_ema50_second = (ema.loc[ema.index[-3],'ema50_to_close'] < 0)
            
            red_candle_bear = (ema.loc[ema.index[-2],'candle'] == 0)
            close_below_low = ema.loc[ema.index[-2],'close'] < ema.loc[ema.index[-3],'low']

            if red_ema50_second and red_candle_bear and close_below_low:

                print ('we are in close below low pattern...its a sell')
                print ('price: {}'.format(ema.loc[ema.index[-1],'close']))
            else:
                print ('close below low pattern not met')    
                    
    def three_bar_pattern(self, ssl, ema):
        
        if ssl.iloc[-2]['green'] > ssl.iloc[-2]['red']:
            
            print ('we are in ssl uptrend...')
            
            fourth_below = (ema.loc[ema.index[-4],'ema50_to_close'] > 0)
            third_above = (ema.loc[ema.index[-3],'ema50_to_close'] < 0)
            second_above = (ema.loc[ema.index[-3],'ema50_to_close'] < 0)
                               
            fourth_green = (ema.loc[ema.index[-4],'candle'] == 1)
            fourth_len = (np.abs(ema.loc[ema.index[-4],'bar_size']) > (2 * ema.loc[ema.index[-4],'rolling_average_bar_size']))
            
            third_red = (ema.loc[ema.index[-3],'candle'] == 0)
            third_len = (np.abs(ema.loc[ema.index[-3],'bar_size']) < (0.33*fourth_len))
            close_above = (ema.loc[ema.index[-4],'close'] > ema.loc[ema.index[-3],'open']) 
                               
            second_green = (ema.loc[ema.index[-2],'candle'] == 1) 
            second_len = (np.abs(ema.loc[ema.index[-2],'bar_size']) > 2 * ema.loc[ema.index[-2],'rolling_average_bar_size'])
            close_below = (ema.loc[ema.index[-2],'open'] < ema.loc[ema.index[-3],'close']) 
                               
            if fourth_below and third_above and second_above and fourth_green and third_red and second_green and fourth_len and third_len and second_len and close_above and close_below:

                print ('we are in bullish three bar high pattern...its a buy')
                print ('price: {}'.format(ema.loc[ema.index[-1],'close']))
                
            else:
                print ('bullish three bar pattern not met')   
        
        
        elif ssl.iloc[-2]['green'] < ssl.iloc[-2]['red']:
                
            print ('we are in ssl downtrend...')
        
            fourth_above = (ema.loc[ema.index[-4],'ema50_to_close'] < 0)
            third_below = (ema.loc[ema.index[-3],'ema50_to_close'] > 0)
            second_below = (ema.loc[ema.index[-3],'ema50_to_close'] > 0)
                               
            fourth_red = (ema.loc[ema.index[-4],'candle'] == 0)
            fourth_len = (np.abs(ema.loc[ema.index[-4],'bar_size']) > (2 * ema.loc[ema.index[-4],'rolling_average_bar_size']))
            
            third_green = (ema.loc[ema.index[-3],'candle'] == 1)
            third_len = (np.abs(ema.loc[ema.index[-3],'bar_size']) < (0.33*fourth_len))
            close_below = (ema.loc[ema.index[-4],'close'] < ema.loc[ema.index[-3],'open']) 
                               
            second_green = (ema.loc[ema.index[-2],'candle'] == 1) 
            second_len = (np.abs(ema.loc[ema.index[-2],'bar_size']) > 2 * ema.loc[ema.index[-2],'rolling_average_bar_size'])
            open_above = (ema.loc[ema.index[-2],'open'] < ema.loc[ema.index[-3],'close']) 
                               
            if fourth_above and third_below and second_below and fourth_red and third_green and second_red and fourth_len and third_len and second_len and close_below and open_above:

                print ('we are in bearish three bar pattern...its a sell')
                print ('price: {}'.format(ema.loc[ema.index[-1],'close']))
                
            else:
                print ('bearish three bar pattern not met') 
    
    
    
    def ohlcv(self, interval, bars):

        tod = datetime.datetime.today()
        since = tod - datetime.timedelta(hours=0, minutes=interval * bars)
        since = since.timestamp()

        ret = self.k.query_public(method='OHLC', data = {'pair': self.pair, 'since': since, 'interval': interval})

        array = ret['result'][list(ret['result'].keys())[0]]

        df = pd.DataFrame(data = array, columns = ['time','open','high','low','close','vwap','volume','count'])
        df = df.astype('float')
        df['time'] = df['time'].apply(lambda x:datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        
        df.set_index('time', inplace =True)
        df.dropna(inplace=True)

        return df
            
    def ssl(self,period_ssl, df_ssl):

        #function ssl
        low = df_ssl.low.values
        high = df_ssl.high.values

        df_ssl['sma_low'] = talib.SMA(low, timeperiod=period_ssl)
        df_ssl['sma_high'] = talib.SMA(high, timeperiod=period_ssl)

        df_ssl['green'] = pd.Series()
        df_ssl['red'] = pd.Series()

        for t in df_ssl.index:

            close = df_ssl.loc[t]['close']
            sma_high = df_ssl.loc[t]['sma_high']
            sma_low = df_ssl.loc[t]['sma_low']

            if close < sma_low:
                df_ssl.loc[slice(t,df_ssl.index[-1]),:]['red'] = df_ssl.loc[slice(t,df_ssl.index[-1]),:]['sma_high']
                df_ssl.loc[slice(t,df_ssl.index[-1]),:]['green'] = df_ssl.loc[slice(t,df_ssl.index[-1]),:]['sma_low']

            elif close > sma_high:
                df_ssl.loc[slice(t,df_ssl.index[-1]),:]['red'] = df_ssl.loc[slice(t,df_ssl.index[-1]),:]['sma_low']
                df_ssl.loc[slice(t,df_ssl.index[-1]),:]['green'] = df_ssl.loc[slice(t,df_ssl.index[-1]),:]['sma_high']


        return df_ssl
    
    def ema(self, period_ema, df_ema):

        raw = df_ema.close.values
        ema = talib.EMA(raw, timeperiod=period_ema)

        df_ema['ema50'] = ema
        df_ema['ema50_to_close']  = (ema - raw)/raw 
        
        return df_ema
        
        
    def average_bar(self, period, df):
        
        df['bar_size'] = df['close'] - df['open']
        df['rolling_average_bar_size'] = df['bar_size'].rolling(period).mean()
        
        for t in df.index:
            if df.loc[t,'bar_size']>0:
                df.loc[t,'candle'] = 1
            else:
                df.loc[t,'candle'] = 0
        return df
    
    def pivot_pattern(self, df_ema, df_ssl):
            
        tr_ssl = TrendLine(df = df_ssl, errpct= 0.005, col = 'close')
        minimaIdxs_ssl, maximaIdxs_ssl = tr_ssl.get_extrema(True), tr_ssl.get_extrema(False)
        mintrend_ssl, maxtrend_ssl = tr_ssl.sorted_slope_trendln(minimaIdxs_ssl), tr_ssl.sorted_slope_trendln(maximaIdxs_ssl)
        df_ssl,m_mid_ssl,c_mid_ssl = tr_ssl.trendln(mintrend_ssl,maxtrend_ssl)

        ssl_trail_bull = ((df_ssl.green[-1] > df_ssl.red[-1]) and (df_ssl.green[-2] < df_ssl.red[-2]) and (df_ssl.green[-3] < df_ssl.red[-3]) and (df_ssl.green[-4] < df_ssl.red[-4]) and (df_ssl.green[-5] < df_ssl.red[-5]) and (df_ssl.green[-6] < df_ssl.red[-6]))
        
        ssl_trail_bear = ((df_ssl.green[-1] < df_ssl.red[-1]) and (df_ssl.green[-2] > df_ssl.red[-2]) and (df_ssl.green[-3] > df_ssl.red[-3]) and (df_ssl.green[-4] > df_ssl.red[-4]) and (df_ssl.green[-5] > df_ssl.red[-5]) and (df_ssl.green[-6] > df_ssl.red[-6]))
        
        ######################################################################################
        
        tr_ema = TrendLine(df = df_ema, errpct= 0.005, col = 'close')
        minimaIdxs_ema, maximaIdxs_ema = tr_ema.get_extrema(True), tr_ema.get_extrema(False)
        mom_ema, momacc_ema = tr_ema.momentum()
        mintrend_ema, maxtrend_ema = tr_ema.sorted_slope_trendln(minimaIdxs_ema), tr_ema.sorted_slope_trendln(maximaIdxs_ema)
        df_ema,m_mid_ema,c_mid_ema = tr_ema.trendln(mintrend_ema,maxtrend_ema)
        
        df_ema['Mom Acc'] = np.round(momacc_ema,2)
        df_ema['Mom'] = np.round(mom_ema,2)
        
        
        bull_pivot_ema = df_ema.close.iloc[minimaIdxs_ema][-1]
        bear_pivot_ema = df_ema.close.iloc[maximaIdxs_ema][-1]

        mom_bear_ema = df_ema['Mom'].iloc[maximaIdxs_ema][-1] < 0
        mom_acc_bear_ema = df_ema['Mom Acc'].iloc[maximaIdxs_ema][-1] < 0

        mom_bull_ema = df_ema['Mom'].iloc[minimaIdxs_ema][-1] > 0
        mom_acc_bull_ema = df_ema['Mom Acc'].iloc[minimaIdxs_ema][-1] > 0
        
        ema_to_close_bull = (df_ema['ema50_to_close'][-1]>0)
        ema_to_close_bear = (df_ema['ema50_to_close'][-1]<0)
        
        print ('ema pivot bull',df_ema.iloc[minimaIdxs_ema].index[-1], 'Mom:', df_ema['Mom'].iloc[minimaIdxs_ema][-1], 'Acc:', df_ema['Mom Acc'].iloc[minimaIdxs_ema][-1])
        print ('ema pivot bear',df_ema.iloc[maximaIdxs_ema].index[-1], 'Mom:', df_ema['Mom'].iloc[maximaIdxs_ema][-1], 'Acc:', df_ema['Mom Acc'].iloc[maximaIdxs_ema][-1])
        print ('ssl mid trend slope',m_mid_ssl)
        print ('ema mid trend slope',m_mid_ema)
        
        ######################################################################################
        
        if ssl_trail_bull:
            
            
            print ('we are in bullish pivot pattern...its a buy')
            print ('-------------------------------------------------')
            print ('price: {}'.format(df_ema.loc[df_ema.index[-1],'close']))
            print ('-------------------------------------------------')
            print (self.fund,'euros to buy')
            print ('-------------------------------------------------')
            print ('closing price set to', 1.003 * df_ema.close[-1])
            
            if self.fund > self.minimum_fund:
    
                response = k.query_private('AddOrder',
                                                {'pair': self.tickers[self.pair],
                                                 'type': 'buy',
                                                 'ordertype': 'market',
                                                 'volume': self.fund,
                                                 'close[ordertype]': 'limit',
                                                 'close[price]': 1.003 * df_ema.close[-1]})
        
            
            else:
                print ('not enough fund to buy')
                
        elif ssl_trail_bear:
            
            print ('we are in bearish pivot pattern...its a sell')
            print ('-------------------------------------------------')
            print ('price: {}'.format(df.loc[df_ema.index[-1],'close']))
            print (float(self.balance[self.tikcers[self.pair]]),self.pair,'to sell')
            print ('-------------------------------------------------')
            print ('closing price set to', 0.997 * df_ema.close[-1])
            
            if self.tickers[self.pair] in self.balance.keys:
                
                print (float(self.balance[self.tickers[self.pair]]),self.tickers[self.pair],'to sell')

                if float(self.balance[self.tickers[self.pair]]) * df_ema.close[-1] > self.minimum_fund:

                    response = k.query_private('AddOrder',
                                                    {'pair': self.tickers[self.pair],
                                                     'type': 'sell',
                                                     'ordertype': 'market',
                                                     'volume': float(self.balance[self.tikcers[self.pair]]),
                                                     'close[ordertype]': 'limit',
                                                     'close[price]': 0.997 * df_ema.close[-1]})

                else:
                    
                    print ('not enough fund to sell')
            else:
                
                print ('the asset is not available')
        else:
            
            print ('pivot pattern not met') 
                
                
        