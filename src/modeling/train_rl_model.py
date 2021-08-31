from collections import defaultdict
import constant as const
import functools
import json
import logging
import matplotlib.pyplot as plt 
import numpy as np
import os
import time
from typing import Callable, Dict, List, Any, Optional, TypeVar
from plotting_utils import plot 

import tensorflow as tf 
import tf_agents
from tf_agents.agents import TFAgent
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.environments import environment_utilities
from tf_agents.bandits.environments import movielens_py_environment
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import TFEnvironment
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.metrics.tf_metric import TFStepMetric
from tf_agents.policies import policy_saver

if tf.__version__[0] != "2":
  raise Exception("The trainer only runs with TensorFlow version 2.")

T = TypeVar("T")


def define_rl_envirioment() -> TFEnvironment:
    env = movielens_py_environment.MovieLensPyEnvironment(
    const.DATA_PATH, const.RANK_K, const.BATCH_SIZE, num_movies=const.NUM_ACTIONS, csv_delimiter="\t")
    environment = tf_py_environment.TFPyEnvironment(env)
    return environment

def define_rl_agent(environment: TFEnvironment) -> TFAgent:
    # Define RL agent/algorithm.
    agent = lin_ucb_agent.LinearUCBAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        tikhonov_weight=const.TIKHONOV_WEIGHT,
        alpha=const.AGENT_ALPHA,
        dtype=tf.float32,
        accepts_per_arm_features=const.PER_ARM)
    return agent

def define_rl_metric(environment: TFEnvironment) -> List[Any]:
    optimal_reward_fn = functools.partial(
    environment_utilities.compute_optimal_reward_with_movielens_environment,
    environment=environment)
    regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward_fn)
    metrics = [regret_metric]
    return metrics

def train(
    root_dir: str,
    agent: TFAgent,
    environment: TFEnvironment,
    training_loops: int,
    steps_per_loop: int,
    additional_metrics: Optional[List[TFStepMetric]] = None,
    training_data_spec_transformation_fn: Optional[Callable[[T], T]] = None,
) -> Dict[str, List[float]]:
  """Performs `training_loops` iterations of training on the agent's policy.

  Uses the `environment` as the problem formulation and source of immediate
  feedback and the agent's algorithm, to perform `training-loops` iterations
  of on-policy training on the policy.
  If one or more baseline_reward_fns are provided, the regret is computed
  against each one of them. Here is example baseline_reward_fn:
  def baseline_reward_fn(observation, per_action_reward_fns):
   rewards = ... # compute reward for each arm
   optimal_action_reward = ... # take the maximum reward
   return optimal_action_reward

  Args:
    root_dir: Path to the directory where training artifacts are written.
    agent: An instance of `TFAgent`.
    environment: An instance of `TFEnvironment`.
    training_loops: An integer indicating how many training loops should be run.
    steps_per_loop: An integer indicating how many driver steps should be
      executed and presented to the trainer during each training loop.
    additional_metrics: Optional; list of metric objects to log, in addition to
      default metrics `NumberOfEpisodes`, `AverageReturnMetric`, and
      `AverageEpisodeLengthMetric`.
    training_data_spec_transformation_fn: Optional; function that transforms
      the data items before they get to the replay buffer.

  Returns:
    A dict mapping metric names (eg. "AverageReturnMetric") to a list of
    intermediate metric values over `training_loops` iterations of training.
  """
  if training_data_spec_transformation_fn is None:
    data_spec = agent.policy.trajectory_spec
  else:
    data_spec = training_data_spec_transformation_fn(
        agent.policy.trajectory_spec)
  replay_buffer = trainer.get_replay_buffer(data_spec, environment.batch_size,
                                            steps_per_loop)

  # `step_metric` records the number of individual rounds of bandit interaction;
  # that is, (number of trajectories) * batch_size.
  step_metric = tf_metrics.EnvironmentSteps()
  metrics = [
      tf_metrics.NumberOfEpisodes(),
      tf_metrics.AverageEpisodeLengthMetric(batch_size=environment.batch_size)
  ]
  if additional_metrics:
    metrics += additional_metrics

  if isinstance(environment.reward_spec(), dict):
    metrics += [tf_metrics.AverageReturnMultiMetric(
        reward_spec=environment.reward_spec(),
        batch_size=environment.batch_size)]
  else:
    metrics += [
        tf_metrics.AverageReturnMetric(batch_size=environment.batch_size)]

  # Store intermediate metric results, indexed by metric names.
  metric_results = defaultdict(list)

  if training_data_spec_transformation_fn is not None:
    add_batch_fn = lambda data: replay_buffer.add_batch(
        training_data_spec_transformation_fn(data))
  else:
    add_batch_fn = replay_buffer.add_batch

  observers = [add_batch_fn, step_metric] + metrics

  driver = dynamic_step_driver.DynamicStepDriver(
      env=environment,
      policy=agent.collect_policy,
      num_steps=steps_per_loop * environment.batch_size,
      observers=observers)

  training_loop = trainer.get_training_loop_fn(
      driver, replay_buffer, agent, steps_per_loop)
  saver = policy_saver.PolicySaver(agent.policy)

  for _ in range(training_loops):
    training_loop()
    metric_utils.log_metrics(metrics)
    for metric in metrics:
      metric.tf_summaries(train_step=step_metric.result())
      metric_results[type(metric).__name__].append(metric.result().numpy())
  saver.save(root_dir)
  return metric_results