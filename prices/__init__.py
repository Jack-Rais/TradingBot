from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from pandas import Timestamp
from datetime import datetime, timedelta
import numpy as np

class PricesClient:

    def __init__(self, api_key, secret_key):

        self._client = StockHistoricalDataClient(
            api_key, 
            secret_key
        )

    def get_last_price(self, symbol:str, date:datetime):

        delta = None

        if not delta and date.minute != 0:
            delta = timedelta(hours=1)
            time_frame = '1m'

        if not delta and date.hour != 0:
            delta = timedelta(days=1)
            time_frame = '1d'

        if not delta and date.month != 0:
            delta = timedelta(days=30)
            time_frame = '1M'

        df = self.get_data_prices(
                symbol,
                date - delta,
                date + delta,
                time_frame
        )

        df_idx = Timestamp(datetime(date.year, 
                                         date.month,
                                         date.day,
                                         date.hour,
                                         date.minute,
                                         date.second,
                                         date.microsecond), tz='UTC')
        
        closest_date = df.index[np.argmin(np.abs(df.index - df_idx))]

        return df.loc[closest_date]
    

    def get_num_prices(self, symbol:str, date:datetime, num:int, interval:str='1h'):

        if interval[-1] == 'h':
            delta = timedelta(days=((int(interval[:-1]) % 24) + 1) * 3)
            
        elif interval[-1] == 'm':
            delta = timedelta(hours=((int(interval[:-1]) % 60) + 1) * 3)

        elif interval[-1] == 'd':
            delta = timedelta(days=int(interval[:-1]) * 7)

        elif interval[-1] == 'M':
            delta = timedelta(days=int(interval[:-1] * 60))

        else:
            raise ValueError(f'Interval: {interval}, not supported, only: %%d, %%m, %%s')


        while True:
            
            df = self.get_delta_prices(
                symbol,
                date,
                delta,
                interval
            )

            if len(df) > num:
                break

            else:
                delta += delta

        return df[(len(df) - num):]
    

    def get_data_prices(self, symbol:str, date:datetime, end:datetime, interval:str='1h'):

        if interval[-1] == 'h':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)
        
        elif interval[-1] == 'm':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)

        elif interval[-1] == 'd':

            if int(interval[:-1]) == 1:
                time_frame = TimeFrame.Day

            else:
                time_frame = TimeFrame(int(interval[:-1]) * 24, TimeFrameUnit.Hour)

        elif interval[-1] == 'M':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Month)

        else:
            raise ValueError(f'Interval: {interval}, not supported, only: %%d, %%m, %%s')

        
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            start = date,
            end = end,
            timeframe = time_frame
        )

        df = self._client.get_stock_bars(request).df

        if len(df.index.names) > 1:
            df_index = df.reset_index()
            df_index = df_index.drop(columns=['symbol'])

            df_new = df_index.set_index('timestamp')

            return df_new
        
        return df
    
    
    def get_delta_prices(self, symbol:str, date:datetime, delta:timedelta, interval:str='1h'):

        if interval[-1] == 'h':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)
        
        elif interval[-1] == 'm':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)

        elif interval[-1] == 'd':

            if int(interval[:-1]) == 1:
                time_frame = TimeFrame.Day

            else:
                time_frame = TimeFrame(int(interval[:-1]) * 24, TimeFrameUnit.Hour)

        elif interval[-1] == 'M':
            time_frame = TimeFrame(int(interval[:-1]), TimeFrameUnit.Month)

        else:
            raise ValueError(f'Interval: {interval}, not supported, only: %%d, %%m, %%s')

        
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            start = date - delta,
            end = date,
            timeframe = time_frame
        )

        df = self._client.get_stock_bars(request).df

        if len(df.index.names) > 1:
            df_index = df.reset_index()
            df_index = df_index.drop(columns=['symbol'])

            df_new = df_index.set_index('timestamp')

            return df_new
        
        return df
