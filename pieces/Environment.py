import logging
import yfinance as yf
import numpy as np

from prices import PricesClient
from Observation import Observer

from datetime import datetime, timedelta
from tf_agents.environments import py_environment
from tf_agents.specs import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import time_step



class TradingEnv(py_environment.PyEnvironment):

    def __init__(self, api_key_alpaca:str,
                       api_secret_alpaca:str,
                       start:datetime, 
                       end:datetime,
                       buying_simbol:str,
                       interval_buying_time:timedelta = timedelta(hours=1), 
                       interval_prices:str = '1h',
                       news_limit:int = 30,
                       limit_percent:int = 100,
                       limit_steps:int | None = 100,
                       use_neutrality:bool = False,
                       logger:logging.Logger | None = None):
        
        super().__init__(handle_auto_reset=True)

        supported = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if interval_prices not in supported:
            raise ValueError(f'Interval dev\'essere in {supported}')
        
        if yf.Ticker(buying_simbol).info.get('exchange', 0) != 'NMS':
            raise NotImplementedError(f'Simboli come {buying_simbol} non sono implementati')
        
        self.start = start
        self.date = start
        self.stop = end

        self.interval_buying_time = interval_buying_time
        self.symbol = buying_simbol
        self.news_limit = news_limit

        self.limit_percent = limit_percent
        self.percent = limit_percent
        self.limit_steps = limit_steps
        self.steps = 0

        self.neutrality_counter = 0
        self.neutrality = use_neutrality

        self.nasdaq_close = datetime(self.date.year, self.date.month, self.date.day + 1)
        self.nasdaq_open = datetime(self.date.year, self.date.month, self.date.day, 9)

        self.client = PricesClient(api_key_alpaca, api_secret_alpaca)

        self.observer = Observer(
            api_key_alpaca,
            api_secret_alpaca,
            news_limit,
            interval_prices
        )

        if isinstance(logger, logging.Logger):

            self.logger = logger.getChild('env')
            self.logger.setLevel(logger.level)

            for handler in logger.handlers:
                self.logger.addHandler(handler)

        else:
            self.logger = None
            self.logger = self.setup_logger(
                name = 'env',
                level = logging.WARNING,
                format_msg = r'[{asctime}] [{levelname}] {message}',
                format_date = r'%S:%M:%H %d-%m-%Y',
                style = r'{'
            )

        self.update_date()


    def setup_logger(self, *, 
                     name:str | None = None, 
                     level:int | str | None = None,
                     format_msg:str | None = None,
                     format_date:str = None,
                     style:str = None):
        
        logger = logging.getLogger(name if name else self.logger.name)
        logger.setLevel(level if level else self.logger.level)

        if hasattr(self.logger, 'hasHandlers'):
            run = self.logger.hasHandlers()
        else:
            run = False

        if run:
            handler = logging.StreamHandler()

            formatter = logging.Formatter(
                fmt = format_msg if format_msg else self.logger.handlers[0].formatter._fmt,
                datefmt = format_date if format_date else self.logger.handlers[0].formatter.datefmt,
                style = style if style else self.logger.handlers[0].formatter._style.asctime_search[0]
            )
            handler.setFormatter(formatter)

            logger.addHandler(handler)

        else:
            handler = logging.StreamHandler()

            formatter = logging.Formatter(
                fmt = format_msg,
                datefmt = format_date,
                style = style
            )
            handler.setFormatter(formatter)

            logger.addHandler(handler)

        return logger
    

    def update_date(self):

        self.logger.info('Updating date')
        self.date += self.interval_buying_time

        self.date = self.start if self.date > self.stop or self.date < self.start else self.date

        if self.date < self.nasdaq_open:

            self.date = datetime(
                self.date.year,
                self.date.month,
                self.date.day,
                9
            )

        if self.date > self.nasdaq_close:

            self.date = datetime(
                self.date.year,
                self.date.month,
                self.date.day,
                9
            )

            self.nasdaq_close = datetime(self.date.year, self.date.month, self.date.day + 1)
            self.nasdaq_open = datetime(self.date.year, self.date.month, self.date.day, 9)
        
        while True:

            dates = self.client.get_data_prices(
                'AAPL',
                self.date - self.interval_buying_time,
                self.date + self.interval_buying_time
            )

            if not dates.empty:
                break

            self.date += timedelta(1)


    def observation_spec(self):
        
        return {
            
            'simbolo': BoundedArraySpec(shape=(1, 5), 
                                        dtype=np.int32,
                                        minimum=0,
                                        maximum=self.observer.tokenizer.vocab_size - 1),
                                        
            'titolo': BoundedArraySpec(shape=(self.news_limit, 100), 
                                        dtype=np.int32,
                                        minimum=0,
                                        maximum=self.observer.tokenizer.vocab_size - 1),

            'paragrafi': BoundedArraySpec(shape=(self.news_limit, 512), 
                                            dtype=np.int32,
                                            minimum=0,
                                            maximum=self.tokenizer.vocab_size - 1),

            'prezzi': ArraySpec(shape=(50, 7), dtype=np.float32)
        }
    

    def action_spec(self):

        return BoundedArraySpec(
            shape = (),
            dtype = np.int32,
            minimum = 0,
            maximum = 2 if self.neutrality else 1
        )
    
    def get_tokenizer(self):
        return self.observer.tokenizer

    def get_observation(self):
        
        return self.observer(self.symbol, self.date)
    

    def _reset(self):
        self.steps = 0
        self.percent = self.limit_percent
        
        return time_step.restart(self.get_observation())


    def _step(self, action):

        last_price = self.client.get_last_price(self.symbol, self.date)['close']
        self.update_date()

        current_price = self.client.get_last_price(self.symbol, self.date)['close']

        if action == 0:

            premio = ((last_price - current_price) / last_price) * 100
            self.neutrality_counter = 0

            if self.neutrality_counter != 0 and self.neutrality:

                time_whithout = timedelta(1) * self.neutrality_counter

                prices = self.client.get_delta_prices(self.symbol, 
                                                      self.date, 
                                                      time_whithout, 
                                                      self.interval_prices)['close']
                val_min = min(prices.values[len(prices)-self.neutrality_counter-1:-1])


                if val_min <= current_price:
                    premio *= 1.2 if premio < 0 else 0.8

                else:
                    premio *= 1.2 if premio > 0 else 0.8


        elif action == 1:

            premio = ((current_price - last_price) / last_price) * 100
            self.neutrality_counter = 0

            if self.neutrality_counter != 0 and self.neutrality:

                time_whithout = timedelta(1) * self.neutrality_counter

                prices = self.client.get_delta_prices(self.symbol, 
                                                      self.date, 
                                                      time_whithout, 
                                                      self.interval_prices)['close']
                val_max = max(prices.values[len(prices)-self.neutrality_counter-1:-1])

                if val_max <= current_price:
                    premio *= 1.2 if premio < 0 else 0.8

                else:
                    premio *= 1.2 if premio > 0 else 0.8

        else:

            if self.neutrality:
                self.neutrality_counter += 1
            
            premio = 0

        self.percent += premio
        
        self.logger.info(f'Updating observation and reward: {{{premio}}}')

        if self.limit_steps:
            self.steps += 1

            if self.limit_steps < self.steps:

                if self.date + self.interval_buying_time * self.limit_steps > self.stop:
                    self.date = self.start

                return time_step.termination(self.get_observation(), np.array(premio, dtype=np.float32))
            
        if self.percent < 0:
            self.percent = self.limit_percent

            return time_step.termination(self.get_observation(), np.array(premio, dtype=np.float32))
        
        if (self.date + self.interval_buying_time) > self.stop:
            self.date = self.start

            return time_step.termination(self.get_observation(), np.array(premio, dtype=np.float32))


        return time_step.transition(self.get_observation(), np.array(premio, dtype=np.float32))