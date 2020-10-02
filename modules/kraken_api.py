import krakenex
import datetime as datetime
import decimal
import time
import numpy as np
import pandas as pd
import talib
import warnings
from findiff import FinDiff
import talib
from sklearn.linear_model import LinearRegression
from itertools import compress

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
        self.tickers = {'BTC/EUR':'XXBT','ETH/EUR':'XETH','DASH/EUR':'XDASH',
                        'XRP/EUR':'XXRP','DOT/EUR':'XDOT'}
        
        self.fund = float(self.balance['ZEUR'])
        self.candle_rankings = {
                                "CDL3LINESTRIKE_Bull": 1,
                                "CDL3LINESTRIKE_Bear": 2,
                                "CDL3BLACKCROWS_Bull": 3,
                                "CDL3BLACKCROWS_Bear": 3,
                                "CDLEVENINGSTAR_Bull": 4,
                                "CDLEVENINGSTAR_Bear": 4,
                                "CDLTASUKIGAP_Bull": 5,
                                "CDLTASUKIGAP_Bear": 5,
                                "CDLINVERTEDHAMMER_Bull": 6,
                                "CDLINVERTEDHAMMER_Bear": 6,
                                "CDLMATCHINGLOW_Bull": 7,
                                "CDLMATCHINGLOW_Bear": 7,
                                "CDLABANDONEDBABY_Bull": 8,
                                "CDLABANDONEDBABY_Bear": 8,
                                "CDLBREAKAWAY_Bull": 10,
                                "CDLBREAKAWAY_Bear": 10,
                                "CDLMORNINGSTAR_Bull": 12,
                                "CDLMORNINGSTAR_Bear": 12,
                                "CDLPIERCING_Bull": 13,
                                "CDLPIERCING_Bear": 13,
                                "CDLSTICKSANDWICH_Bull": 14,
                                "CDLSTICKSANDWICH_Bear": 14,
                                "CDLTHRUSTING_Bull": 15,
                                "CDLTHRUSTING_Bear": 15,
                                "CDLINNECK_Bull": 17,
                                "CDLINNECK_Bear": 17,
                                "CDL3INSIDE_Bull": 20,
                                "CDL3INSIDE_Bear": 56,
                                "CDLHOMINGPIGEON_Bull": 21,
                                "CDLHOMINGPIGEON_Bear": 21,
                                "CDLDARKCLOUDCOVER_Bull": 22,
                                "CDLDARKCLOUDCOVER_Bear": 22,
                                "CDLIDENTICAL3CROWS_Bull": 24,
                                "CDLIDENTICAL3CROWS_Bear": 24,
                                "CDLMORNINGDOJISTAR_Bull": 25,
                                "CDLMORNINGDOJISTAR_Bear": 25,
                                "CDLXSIDEGAP3METHODS_Bull": 27,
                                "CDLXSIDEGAP3METHODS_Bear": 26,
                                "CDLTRISTAR_Bull": 28,
                                "CDLTRISTAR_Bear": 76,
                                "CDLGAPSIDESIDEWHITE_Bull": 46,
                                "CDLGAPSIDESIDEWHITE_Bear": 29,
                                "CDLEVENINGDOJISTAR_Bull": 30,
                                "CDLEVENINGDOJISTAR_Bear": 30,
                                "CDL3WHITESOLDIERS_Bull": 32,
                                "CDL3WHITESOLDIERS_Bear": 32,
                                "CDLONNECK_Bull": 33,
                                "CDLONNECK_Bear": 33,
                                "CDL3OUTSIDE_Bull": 34,
                                "CDL3OUTSIDE_Bear": 39,
                                "CDLRICKSHAWMAN_Bull": 35,
                                "CDLRICKSHAWMAN_Bear": 35,
                                "CDLSEPARATINGLINES_Bull": 36,
                                "CDLSEPARATINGLINES_Bear": 40,
                                "CDLLONGLEGGEDDOJI_Bull": 37,
                                "CDLLONGLEGGEDDOJI_Bear": 37,
                                "CDLHARAMI_Bull": 38,
                                "CDLHARAMI_Bear": 72,
                                "CDLLADDERBOTTOM_Bull": 41,
                                "CDLLADDERBOTTOM_Bear": 41,
                                "CDLCLOSINGMARUBOZU_Bull": 70,
                                "CDLCLOSINGMARUBOZU_Bear": 43,
                                "CDLTAKURI_Bull": 47,
                                "CDLTAKURI_Bear": 47,
                                "CDLDOJISTAR_Bull": 49,
                                "CDLDOJISTAR_Bear": 51,
                                "CDLHARAMICROSS_Bull": 50,
                                "CDLHARAMICROSS_Bear": 80,
                                "CDLADVANCEBLOCK_Bull": 54,
                                "CDLADVANCEBLOCK_Bear": 54,
                                "CDLSHOOTINGSTAR_Bull": 55,
                                "CDLSHOOTINGSTAR_Bear": 55,
                                "CDLMARUBOZU_Bull": 71,
                                "CDLMARUBOZU_Bear": 57,
                                "CDLUNIQUE3RIVER_Bull": 60,
                                "CDLUNIQUE3RIVER_Bear": 60,
                                "CDL2CROWS_Bull": 61,
                                "CDL2CROWS_Bear": 61,
                                "CDLBELTHOLD_Bull": 62,
                                "CDLBELTHOLD_Bear": 63,
                                "CDLHAMMER_Bull": 65,
                                "CDLHAMMER_Bear": 65,
                                "CDLHIGHWAVE_Bull": 67,
                                "CDLHIGHWAVE_Bear": 67,
                                "CDLSPINNINGTOP_Bull": 69,
                                "CDLSPINNINGTOP_Bear": 73,
                                "CDLUPSIDEGAP2CROWS_Bull": 74,
                                "CDLUPSIDEGAP2CROWS_Bear": 74,
                                "CDLGRAVESTONEDOJI_Bull": 77,
                                "CDLGRAVESTONEDOJI_Bear": 77,
                                "CDLHIKKAKEMOD_Bull": 82,
                                "CDLHIKKAKEMOD_Bear": 81,
                                "CDLHIKKAKE_Bull": 85,
                                "CDLHIKKAKE_Bear": 83,
                                "CDLENGULFING_Bull": 84,
                                "CDLENGULFING_Bear": 91,
                                "CDLMATHOLD_Bull": 86,
                                "CDLMATHOLD_Bear": 86,
                                "CDLHANGINGMAN_Bull": 87,
                                "CDLHANGINGMAN_Bear": 87,
                                "CDLRISEFALL3METHODS_Bull": 94,
                                "CDLRISEFALL3METHODS_Bear": 89,
                                "CDLKICKING_Bull": 96,
                                "CDLKICKING_Bear": 102,
                                "CDLDRAGONFLYDOJI_Bull": 98,
                                "CDLDRAGONFLYDOJI_Bear": 98,
                                "CDLCONCEALBABYSWALL_Bull": 101,
                                "CDLCONCEALBABYSWALL_Bear": 101,
                                "CDL3STARSINSOUTH_Bull": 103,
                                "CDL3STARSINSOUTH_Bear": 103,
                                "CDLDOJI_Bull": 104,
                                "CDLDOJI_Bear": 104
                            }
        
    
    def live_pattern(self, interval_5min, bars_5min, period_ema_5min, 
                           interval_hourly, bars_hourly, period_ssl):
        
        while True:
            
            print ('------------------------------------------------------------------------')

            df_5min = self.ohlcv(interval_5min, bars_5min)
            df_hourly = self.ohlcv(interval_hourly, bars_hourly)
            
            df_ema = self.ema(period_ema_5min, df_5min)
            df_ssl = self.ssl(period_ssl, df_hourly)
            
            #df_ema = self.average_bar(30, df_ema)
            
            print (df_ssl.index[-1],'hourly candle in progress')
            print ('starting',df_ema.index[-1],'5min candle')

            #################################################################################
            self.research_pattern(df_ema, df_ssl)
            #################################################################################

            last = datetime.datetime.strptime(df_ema.index[-1], '%Y-%m-%d %H:%M:%S').timestamp()
            next_time = last + (interval_5min*60)
            sleep = float(next_time) - time.time()

            #print (np.round(sleep/60,2), 'minutes sleep time')

            time.sleep(sleep)
            
    def daily_analysis(self, interval_daily, bars_daily, period_ema_daily, period_ssl_daily,
                             rolling_reg_period, zone_quantile):
        
        df_daily = self.ohlcv(interval_daily, bars_daily)
        df_daily = self.ema(period_ema_daily, df_daily)
        df_daily = self.ssl(period_ssl_daily, df_daily)
        
        multi_reg = self.rolling_regression(df_daily, rolling_reg_period)
        ema_to_close = df_daily['ema_to_close'].iloc[-1]
        bull,bear = self.tr_zones(df_daily, zone_quantile)
        if df_daily.green[-1] > df_daily.red[-1]:
            print ('daily bullish ssl for',period_ssl_daily,'period')
            score_ssl = +1
            
        else:
            print ('daily bearish ssl for',period_ssl_daily,'period')
            score_ssl = -1
            
        
        if df_daily.close[-1] < bull:
            print ('daily price in bullish for',zone_quantile,'qunatilize zone')
            score_zone = +1
            
        elif df_daily.close[-1] > bear:
            print ('daily price in bearish for',zone_quantile,'qunatilize zone')
            score_zone = -1
        
        else:
            print ('daily price in neutral for',zone_quantile,'qunatilize zone')
            score_zone = 0
            
        uptrend_date_daily, uptrend_mom_daily, uptrend_mom_acc_daily, downtrend_date_daily, downtrend_mom_daily, downtrend_mom_acc_daily, m_mid_daily = self.trend_analysis(df_daily)
        
        if datetime.datetime.strptime(uptrend_date_daily, '%Y-%m-%d %H:%M:%S')>datetime.datetime.strptime(downtrend_date_daily, '%Y-%m-%d %H:%M:%S'):
            last_date = uptrend_date_daily
            print ('daily uptrend pivot', last_date, 'Mom:', uptrend_mom_daily, 'Acc:', uptrend_mom_acc_daily)
            score_piv = +1
        else:
            last_date = downtrend_date_daily
            print ('daily downtrend pivot', last_date, 'Mom:', downtrend_mom_daily, 'Acc:', downtrend_mom_acc_daily)
            score_piv = -1
            
        if multi_reg[0][-1][0] > 0:
            print ('daily bullish momentum regression line slope for last', rolling_reg_period ,'days period:', multi_reg[0][-1][0])
            score_reg = +1
        else:
            print ('daily bearish momentum regression line slope for last', rolling_reg_period ,'days period:', multi_reg[0][-1][0])
            score_reg = -1
            
        if ema_to_close > 0:
            print ('daily bullish ema', period_ema_daily,'to close', ema_to_close)
            score_ema = +1
        else:
            print ('daily bearish ema', period_ema_daily,'to close', ema_to_close)
            score_ema = -1
            
        score = score_ema + score_reg + score_piv + score_zone + score_ssl
        
        print ('daily total score:',score)
        
        return score
        
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
    
    
    def recognize_candlestick(self,df):

        op = df['open'].astype(float)
        hi = df['high'].astype(float)
        lo = df['low'].astype(float)
        cl = df['close'].astype(float)

        candle_names = talib.get_function_groups()['Pattern Recognition']

        exclude_items = ('CDLCOUNTERATTACK',
                         'CDLLONGLINE',
                         'CDLSHORTLINE',
                         'CDLSTALLEDPATTERN',
                         'CDLKICKINGBYLENGTH')

        candle_names = [candle for candle in candle_names if candle not in exclude_items]


        for candle in candle_names:
            df[candle] = getattr(talib, candle)(op, hi, lo, cl)


        df['candlestick_pattern'] = np.nan
        df['candlestick_match_count'] = np.nan
        for index, row in df.iterrows():

            # no pattern found
            if len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
                df.loc[index,'candlestick_pattern'] = "NO_PATTERN"
                df.loc[index, 'candlestick_match_count'] = 0
            # single pattern found
            elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:
                # bull pattern 100 or 200
                if any(row[candle_names].values > 0):
                    pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bull'
                    df.loc[index, 'candlestick_pattern'] = pattern
                    df.loc[index, 'candlestick_match_count'] = 1
                # bear pattern -100 or -200
                else:
                    pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bear'
                    df.loc[index, 'candlestick_pattern'] = pattern
                    df.loc[index, 'candlestick_match_count'] = 1
            # multiple patterns matched -- select best performance
            else:
                # filter out pattern names from bool list of values
                patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))
                container = []
                for pattern in patterns:
                    if row[pattern] > 0:
                        container.append(pattern + '_Bull')
                    else:
                        container.append(pattern + '_Bear')
                rank_list = [self.candle_rankings[p] for p in container]
                if len(rank_list) == len(container):

                    rank_index_best = rank_list.index(min(rank_list))
                    df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
                    df.loc[index, 'candlestick_match_count'] = len(container)

        for t in df.index:
            if df.loc[t]['candlestick_pattern'].split('_')[-1] == 'Bear':
                df.loc[t,'candlestick_match_count']= df.loc[t]['candlestick_match_count']*-1

        return df

    
    def tr_zones(self, df, quantiles):
        
        bear = max(df.close) - ((max(df.close) - min(df.close))/quantiles) 
        bull = ((max(df.close) - min(df.close))/quantiles + min(df.close)) 
        
        return bull, bear
        
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

        df_ema['ema'] = ema
        df_ema['ema_to_close']  = (ema - raw)/raw 
        
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
    
    def trend_analysis(self, df):
        
        tr = TrendLine(df = df, errpct= 0.005, col = 'close')
        minimaIdxs, maximaIdxs = tr.get_extrema(True), tr.get_extrema(False)
        mintrend, maxtrend = tr.sorted_slope_trendln(minimaIdxs), tr.sorted_slope_trendln(maximaIdxs)
        df,m_mid,c_mid = tr.trendln(mintrend,maxtrend)
        
        uptrend_date = df.iloc[minimaIdxs].index[-1]
        uptrend_mom = df['Mom'].iloc[minimaIdxs][-1]
        uptrend_mom_acc = df['Mom Acc'].iloc[minimaIdxs][-1]
        
        downtrend_date = df.iloc[maximaIdxs].index[-1]
        downtrend_mom = df['Mom'].iloc[maximaIdxs][-1]
        downtrend_mom_acc = df['Mom Acc'].iloc[maximaIdxs][-1]
        
        
        return uptrend_date, uptrend_mom, uptrend_mom_acc, downtrend_date, downtrend_mom, downtrend_mom_acc, m_mid
    
    def rolling_regression(self, df, reg_range):
        
        multi_reg = []
        for i in range(0,len(df.index),reg_range): 

            j= i + reg_range

            if i == 0:
                x = np.array(range(len(df.close.index)))[-j:].reshape(-1,1)
                y = df.close.iloc[-j:].values.reshape(-1,1)

            else:
                x = np.array(range(len(df.close.index)))[-j:-i].reshape(-1,1)
                y = df.close.iloc[-j:-i].values.reshape(-1,1)


            reg = LinearRegression().fit(X = x, y = y)
            y_pred = reg.predict(x).flatten()
            multi_reg.append([y_pred,x,y,reg.coef_[0]])

        return multi_reg
    
    def research_pattern(self, df_ema, df_ssl):
        
        uptrend_date_ssl, uptrend_mom_ssl, uptrend_mom_acc_ssl, downtrend_date_ssl, downtrend_mom_ssl, downtrend_mom_acc_ssl, m_mid_ssl = self.trend_analysis(df_ssl)
        
        uptrend_date_ema, uptrend_mom_ema, uptrend_mom_acc_ema, downtrend_date_ema, downtrend_mom_ema, downtrend_mom_acc_ema, m_mid_ema = self.trend_analysis(df_ema)
        
        ######################################################################################
        
        ssl_trail_bull = ((df_ssl.green[-2] > df_ssl.red[-2]) and (df_ssl.green[-3] < df_ssl.red[-3]) and (df_ssl.green[-4] < df_ssl.red[-4]) and (df_ssl.green[-5] < df_ssl.red[-5]) and (df_ssl.green[-6] < df_ssl.red[-6]) and (df_ssl.green[-7] < df_ssl.red[-7]))
        
        ssl_trail_bear = ((df_ssl.green[-2] < df_ssl.red[-2]) and (df_ssl.green[-3] > df_ssl.red[-3]) and (df_ssl.green[-4] > df_ssl.red[-4]) and (df_ssl.green[-5] > df_ssl.red[-5]) and (df_ssl.green[-6] > df_ssl.red[-6]) and (df_ssl.green[-7] > df_ssl.red[-7]))
        
        ######################################################################################
        
        ema_to_close_bull = (df_ema['ema_to_close'][-1]>0)
        ema_to_close_bear = (df_ema['ema_to_close'][-1]<0)

        if datetime.datetime.strptime(uptrend_date_ssl, '%Y-%m-%d %H:%M:%S')>datetime.datetime.strptime(downtrend_date_ssl, '%Y-%m-%d %H:%M:%S'):
            last_date = uptrend_date_ssl
            print ('hourly uptrend pivot', last_date, 'Mom:', uptrend_mom_ssl, 'Acc:', uptrend_mom_acc_ssl)
            
        else:
            last_date = downtrend_date_ssl
            print ('hourly downtrend pivot', last_date, 'Mom:', downtrend_mom_ssl, 'Acc:', downtrend_mom_acc_ssl)

        ######################################################################################
        
        if ssl_trail_bull:
            
            print ('we are in bullish pattern...its a buy')
            print ('-------------------------------------------------')
            print ('price: {}'.format(df_ema.loc[df_ema.index[-1],'close']))
            print ('-------------------------------------------------')
            print (self.fund,'euros to buy')
            
            if self.fund > self.minimum_fund:
    
                response = k.query_private('AddOrder',
                                                {'pair': self.tickers[self.pair],
                                                 'type': 'buy',
                                                 'ordertype': 'market',
                                                 'volume': self.fund})
        
            
            else:
                print ('not enough fund to buy')
                
        elif ssl_trail_bear:
            
            print ('we are in bearish pattern...its a sell')
            print ('-------------------------------------------------')
            print ('price: {}'.format(df_ema.loc[df_ema.index[-1],'close']))
            print ('-------------------------------------------------')
            
            try:
                print (float(self.balance[self.tickers[self.pair]]),self.tickers[self.pair],'to sell')

                if float(self.balance[self.tickers[self.pair]]) * df_ema.close[-1] > self.minimum_fund:

                    response = k.query_private('AddOrder',
                                                    {'pair': self.tickers[self.pair],
                                                     'type': 'sell',
                                                     'ordertype': 'market',
                                                     'volume': float(self.balance[self.tickers[self.pair]])})

                
                else:
                    
                    print ('not enough fund to sell')
            except:
                print ('the asset is not in the balance')
        else:
            print ('pattern not met')    
                
        