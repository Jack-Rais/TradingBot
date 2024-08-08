import tensorflow as tf

from typing import Sequence, Any
from tf_agents.drivers.driver import Driver
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import Trajectory, Transition, trajectory
from tf_agents.policies.py_policy import PyPolicy

class TradingDriver(Driver):

    def __init__(self, env:PyEnvironment,
                       policy: PyPolicy, 
                       observers: Sequence[(Trajectory)], 
                       num_steps:int,
                       transition_observers: Sequence[(Transition)] | None = None, 
                       info_observers: Sequence[(Any)] | None = None):
        
        super().__init__(
            env,
            policy, 
            observers, 
            transition_observers, 
            info_observers
        )

        self._env = env
        self._policy = policy
        self._observers = observers if observers else []
        self._transition_observers = transition_observers if transition_observers else []
        self._steps = num_steps

    
    @tf.function
    def run(self, time_step: Any | None = None, policy_state: Any | None = None ):

        def condition(time_step, policy_state):
            return not time_step.is_last()

        def body(time_step, policy_state):

            action_step = self._policy.action(time_step, policy_state)
            next_time_step = self._env.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            for observer in self._observers:
                observer(traj)

            for transition_observer in self._transition_observers:
                transition_observer(time_step, action_step, next_time_step)

            time_step = next_time_step
            policy_state = action_step.state

            return time_step, policy_state

        for _ in range(self._steps):

            time_step = time_step or self._env.reset()
            policy_state = policy_state or {}

            time_step, policy_state = tf.while_loop(
                                                    condition,
                                                    body,
                                                    loop_vars = [time_step, policy_state]
                                                )

        return time_step, policy_state

