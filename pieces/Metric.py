import csv
import os
import tensorflow as tf

from tf_agents.metrics.tf_metric import TFStepMetric
from tf_agents.trajectories import Trajectory

class TradingMetric(TFStepMetric):

    def __init__(self, save_in_file = False,
                       file_output = 'metrics.csv',
                       name = 'TradingMetric'):
        super().__init__(name = name)

        self._rewards = tf.Variable(0, dtype=tf.float32, trainable = False)
        self._num_rewards = tf.Variable(0, dtype=tf.float32, trainable=False)
        self._min_reward = tf.Variable(float('inf'), trainable=False, dtype=tf.float32)
        self._max_reward = tf.Variable(-float('inf'), trainable=False, dtype=tf.float32)

        self.save_in_file = save_in_file
        self.file_output = file_output

        if self.save_in_file and not os.path.exists(self.file_output):
            
            with open(self.file_output, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['tot_reward', 'avg_reward', 'min_reward', 'max_reward'])
    
    @tf.function
    def call(self, traj:Trajectory):

        self._rewards.assign_add(tf.reduce_sum(traj.reward))
        self._num_rewards.assign_add(tf.constant(1, dtype=tf.float32))

        self._min_reward.assign(tf.math.minimum(self._min_reward, tf.reduce_min(traj.reward)))
        self._max_reward.assign(tf.math.maximum(self._max_reward, tf.reduce_max(traj.reward)))

    
    def result(self):

        result = {
            'tot_reward': self._rewards.numpy(),
            'avg_reward': tf.Variable(self._rewards / self._num_rewards, trainable=False, dtype=tf.float32).numpy()\
                                if self._num_rewards != 0 else \
                                tf.Variable(0, trainable=False, dtype=tf.float32).numpy(),
            'min_reward': self._min_reward.numpy(),
            'max_reward': self._max_reward.numpy()
        }

        if self.save_in_file:
            with open(self.file_output, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    result['tot_reward'],
                    result['avg_reward'],
                    result['min_reward'],
                    result['max_reward']
                ])

        return result
    

    def reset(self):

        self._rewards = tf.Variable(0, dtype=tf.float32, trainable = False)
        self._num_rewards = tf.Variable(0, dtype=tf.float32, trainable=False)
        self._min_reward = tf.Variable(float('inf'), trainable=False, dtype=tf.float32)
        self._max_reward = tf.Variable(-float('inf'), trainable=False, dtype=tf.float32)