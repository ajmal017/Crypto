from modules import Kraken

constructor = Kraken(pair = 'BTC/EUR', api_key_file = 'kraken.key', minimum_fund=1000, percent_alloc=1)

constructor.live_pattern(interval_5min = 5, bars_5min = 200, period_ema_5min = 50, 
                         interval_hourly = 60 , bars_hourly = 100, period_ssl = 10,quantiles =3)

