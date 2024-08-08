import tensorflow as tf

from typing import Iterable
from tf_agents.networks.network import Network


class TextProcessor(tf.keras.layers.Layer):

    def __init__(self, vocab_size, 
                       embed_dim, 
                       lstm_units:Iterable = (128, 64),
                       dense_units:Iterable = (128, 64, 32),
                       last_units:int | None = None):
        super().__init__()

        self._embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self._model = tf.keras.models.Sequential(name = 'text_prep_model')

        for units in lstm_units[:-1]:
            self._model.add(tf.keras.layers.LSTM(units, activation='relu', return_sequences=True))
        self._model.add(tf.keras.layers.LSTM(lstm_units[-1], activation='relu'))

        for units in dense_units[:(-1 if not last_units else len(dense_units))]:
            self._model.add(tf.keras.layers.Dense(units, activation='relu'))

        if last_units:
            self._model.add(tf.keras.layers.Dense(last_units))
        else:
            self._model.add(tf.keras.layers.Dense(dense_units[-1]))

    def call(self, x):

        if x.shape.rank == 2:
            x = self._embed(tf.expand_dims(x, axis=0))
        
        else:
            x = self._embed(x)
        
        def process_batch(batch):
            batch = self._model(batch)
            return batch

        x = tf.map_fn(process_batch, x, dtype=tf.float32)

        return x
    

class TextSentimentAnalysis(tf.keras.layers.Layer):

    def __init__(self, vocab_size:int, 
                       embed_dim_title:int,
                       embed_dim_parag:int, 
                       lstm_units_title:Iterable = (128, 64),
                       dense_units_title:Iterable = (128, 64, 32),
                       lstm_units_parag:Iterable = (256, 128, 64),
                       dense_units_parag:Iterable = (128, 64, 32),
                       last_layer_units:Iterable = (256, 128, 64, 32),
                       last_units_title:int | None = None,
                       last_units_parag:int | None = None):
        super().__init__()
        
        self._process_title = TextProcessor(
            vocab_size = vocab_size,
            embed_dim = embed_dim_title,
            lstm_units = lstm_units_title,
            dense_units = dense_units_title,
            last_units = last_units_title
        )

        self._process_parag = TextProcessor(
            vocab_size = vocab_size,
            embed_dim = embed_dim_parag,
            lstm_units = lstm_units_parag,
            dense_units = dense_units_parag,
            last_units = last_units_parag
        )

        self._last_layer = tf.keras.models.Sequential(name = 'text_last_model')
        for unit in last_layer_units[:-1]:
            self._last_layer.add(tf.keras.layers.Dense(unit, activation='relu'))

        self._last_layer.add(tf.keras.layers.Dense(last_layer_units[-1], activation='sigmoid'))


    def call(self, x:Iterable):

        t, p = x

        if t.shape.rank == 2 and p.shape.rank == 2:

            t = tf.expand_dims(t, axis=0)
            p = tf.expand_dims(p, axis=0)
        

        result = []

        for batch in range(t.shape[0]):

            t_proces = self._process_title(t[batch])
            p_proces = self._process_parag(p[batch])

            x = tf.concat([t_proces, p_proces], axis=-1)
            x = tf.reduce_mean(x, axis=-2)

            x = self._last_layer(x)

            result.append(x)

        return tf.concat(result, axis=0)
    

class PricesSentimentAnalysis(tf.keras.layers.Layer):

    def __init__(self, lstm_units:Iterable = (256, 128, 64),
                       dense_units:Iterable = (256, 128, 64, 32),
                       last_layer_dense:int | None = None,
                       activation:str = 'relu'):
        super().__init__()

        self._model = tf.keras.models.Sequential(name = 'prices_model')

        for units in lstm_units[:-1]:
            self._model.add(tf.keras.layers.LSTM(units, return_sequences=True))
        
        self._model.add(tf.keras.layers.LSTM(lstm_units[-1]))


        for units in dense_units[:(len(dense_units) if last_layer_dense else -1)]:
            self._model.add(tf.keras.layers.Dense(units, activation=activation))
        
        self._model.add(tf.keras.layers.Dense(last_layer_dense or dense_units[-1]))


    def call(self, x):
        
        if x.shape.rank == 2:
            x = tf.expand_dims(x, axis=0)

        return self._model(x)



class TradingNet(Network):

    def __init__(self, action_spec,
                       vocab_size:int, 
                       embed_dim_title:int = 128,
                       embed_dim_parag:int = 256,
                       last_model_params:Iterable = (256, 128, 64, 32),
                       lstm_units_title:Iterable = (128, 64),
                       dense_units_title:Iterable = (128, 64, 32),
                       lstm_units_prices: Iterable = (256, 128, 64),
                       lstm_units_parag:Iterable = (256, 128, 64),
                       dense_units_parag:Iterable = (128, 64, 32),
                       dense_units_prices: Iterable = (256, 128, 64, 32),
                       last_layer_units_news:Iterable = (256, 128, 64, 32, 2),
                       last_units_title:int | None = None,
                       last_layer_prices: int | None = None,
                       last_units_parag:int | None = None,
                       activation_prices: str = 'relu',
                       name:str = 'CustomQNetwork'):
        
        num_actions = action_spec.maximum - action_spec.minimum + 1

        self._news = TextSentimentAnalysis(
            vocab_size = vocab_size,
            embed_dim_title = embed_dim_title,
            embed_dim_parag = embed_dim_parag,
            lstm_units_title = lstm_units_title,
            lstm_units_parag = lstm_units_parag,
            dense_units_title = dense_units_title,
            dense_units_parag = dense_units_parag,
            last_layer_units = last_layer_units_news,
            last_units_title = last_units_title,
            last_units_parag = last_units_parag
        )

        self._prices = PricesSentimentAnalysis(
            lstm_units = lstm_units_prices,
            dense_units = dense_units_prices,
            last_layer_dense = last_layer_prices,
            activation = activation_prices
        )

        self._last_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(x, activation='relu') for x in last_model_params
        ] + [tf.keras.layers.Dense(num_actions, activation='sigmoid')],
        name = 'output_model')

        super().__init__()


    def call(self, step, step_type=None, network_state=(), training=False):

        news_part = self._news((step['titolo'], 
                                step['paragrafi']), 
                               training = training)
        
        prices_part = self._prices(step['prezzi'], 
                                   training = training)

        x = tf.concat([news_part, prices_part], axis=1)

        x = self._last_model(x, training = training)

        return x, network_state