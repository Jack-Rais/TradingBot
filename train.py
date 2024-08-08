import tensorflow as tf

from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.utils import common
from tf_agents.policies.tf_policy import TFPolicy

from datetime import datetime
from GymEnvironment import TradingEnv
from Net import TradingNet
from Metric import TradingMetric


initial_collect_steps = 50
batch_size = 64
num_eval_episodes = 1
num_iterations = 20
log_interval = 5
eval_interval = 10
name = 'model'


API_KEY = 'API_KEY'
API_SECRET = 'API_SECRET'

env = TradingEnv(
    API_KEY,
    API_SECRET,
    datetime(2017, 1, 1),
    datetime(2024, 1, 1),
    'AAPL'
)
vocab_size = env.get_tokenizer().vocab_size
env = GymWrapper(env)
env = TFPyEnvironment(env)

q_net = TradingNet(
    env.action_spec(),
    vocab_size,
    64,
    64,
    (128, 32),
    (64, 32),
    (64, 32),
    (64, 32),
    (64, 32),
    (64, 32),
    (64, 32),
    (64, 32)
)
q_net.create_variables(env.observation_spec())

agent = DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_net,
    tf.keras.optimizers.Adam()
)
agent.initialize()

replay_buffer = TFUniformReplayBuffer(
    agent.collect_data_spec,
    env.batch_size,
    100
)

train_metrics = [
    TradingMetric(save_in_file = True)
]

collect_steps_per_iteration = 1
collect_driver = DynamicStepDriver(
    env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch] + train_metrics,
    num_steps=collect_steps_per_iteration)

# Initial data collection
collect_driver.run = common.function(collect_driver.run)
collect_driver.run()

dataset = replay_buffer.as_dataset(
    num_parallel_calls=4,
    sample_batch_size=batch_size,
    num_steps=2
).prefetch(4)

iterator = iter(dataset)

agent.train_step_counter.assign(0)

def compute_avg_return(environment:TFPyEnvironment, 
                       policy:TFPolicy,
                       num_episodes:int=10):
    
    total_return = 0.0
    total_count = 0.0

    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        episode_count = 0

        while not time_step.is_last():

            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            episode_count += 1

        total_count += episode_count
        total_return += episode_return

    avg_count = total_count / num_episodes
    avg_return = total_return / num_episodes

    return avg_return.numpy()[0], avg_count

avg_return, avg_step = compute_avg_return(env, agent.policy, num_eval_episodes)
returns = [avg_return]
steps = [avg_step]


for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_driver.run()

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}, Agv Return = {2}'.format(step, train_loss, train_metrics[0].result().numpy))

    if step % eval_interval == 0:
        avg_return, avg_step = compute_avg_return(env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}, Average Steps = {2}'.format(step, avg_return, avg_step))

        returns.append(avg_return)
        steps.append(avg_step)

