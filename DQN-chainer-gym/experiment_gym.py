# -*- coding: utf-8 -*-
"""
Simple gym experiment setup
"""


import gym
import dqn_agent as ag
import matplotlib.pyplot as plt
import numpy as np
import time

# Generate an environment
env = gym.make('Pong-v0')

# Generate an agent
agent = ag.DQN_Agent(env)

eval_interval = 5
num_episode = 1000
total_score = []
eval_steps = []
for i_episode in xrange(num_episode):
    observation  = env.reset()
    terminal = False
    total_score_ = 0
    reward = 0.0  # initial reward is assumed to be zero
    step_in_episode = 0

    if np.mod(i_episode, eval_interval) == 0:
        # Learnin OFF evaluation
        agent.policyFrozen = True
    else:
        # Learning ON
        agent.policyFrozen = False

    while True:
        print(str(i_episode) + "-th episode")
        #env.render() # Render the game

        if step_in_episode == 0:
            observation, reward, terminal, info = env.step(agent.start(observation, evaluation=False)) # take an action
        else:
            observation, reward, terminal, info = env.step(agent.act(observation, reward, evaluation=False)) # take an action

        total_score_ += reward
        step_in_episode += 1

        if terminal is True:
            agent.end(reward)
            break

    if np.mod(i_episode, eval_interval) == 0:
        total_score.append(range(0,i_episode+1, eval_interval), total_score_)
        print("REWARD@" + str(i_episode) + "-th episode : " + str(total_score_))

        plt.clf()
        plt.plot(total_score)
        plt.legend(["Total Score"])
        plt.draw()
        plt.pause(0.001)

        # Save the current agent parameters
        agent.save()
