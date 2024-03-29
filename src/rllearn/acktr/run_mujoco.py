#!/usr/bin/env python3

import tensorflow as tf

import logger
from rllearn.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from rllearn.acktr.acktr_cont import learn
from rllearn.acktr.policies import GaussianMlpPolicy
from rllearn.acktr.value_functions import NeuralNetValueFunction


def train(env_id, num_timesteps, seed):
    """
    train an ACKTR model on atari

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    env = make_mujoco_env(env_id, seed)

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.compat.v1.variable_scope("vf"):
            value_fn = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.compat.v1.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(
            env,
            policy=policy,
            value_fn=value_fn,
            gamma=0.99,
            lam=0.97,
            timesteps_per_batch=2500,
            desired_kl=0.002,
            num_timesteps=num_timesteps,
            animate=False,
        )

        env.close()


def main():
    """
    Runs the test
    """
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == "__main__":
    main()
