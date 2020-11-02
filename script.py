import sys

sys.path.append('/home/keyvan_tajbakhsh/.local/lib/python3.7/site-packages')
sys.path.append('')

from modules import KrakenLight

pair = str(input("Enter your pair: "))
api = str(input("Enter your api key file: "))
sleepage = int(input("Enter your sleepage: "))
alloc = float(input("Enter your allocation: "))
minimum_fund = int(input("Enter your minimum fund: "))

constructor = KrakenLight(pair = pair, api_key_file = api, minimum_fund=minimum_fund, percent_alloc=alloc, sleepage =sleepage)

constructor.live_pattern(interval_5min = 5, bars_5min = 1000, period_ema_5min = 50, 
                         interval_hourly = 60 , bars_hourly = 1000, period_ssl = 10,quantiles =3)

