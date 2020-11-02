import sys

sys.path.append('/home/keyvan_tajbakhsh/.local/lib/python3.7/site-packages')
sys.path.append('')

from modules import KrakenLight

constructor = KrakenLight(pair = 'ADA/EUR', api_key_file = 'kraken.key', minimum_fund=1000, percent_alloc=1, sleepage =120)

constructor.live_pattern(interval_5min = 5, bars_5min = 1000, period_ema_5min = 50, 
                         interval_hourly = 60 , bars_hourly = 1000, period_ssl = 10,quantiles =3)

