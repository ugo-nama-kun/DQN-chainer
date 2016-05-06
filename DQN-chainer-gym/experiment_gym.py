# -*- coding: utf-8 -*-
"""
Simple gym experiment setup
"""


import gym
import dqn_agent as ag
import matplotlib.pyplot as plt
import numpy as np

# Generate an environment
env = gym.make('Pong-v0')

# Generate an agent
agent = ag.DQN_Agent(env)

num_episode = 1000
total_score = np.zeros(num_episode)

for i_episode in xrange(num_episode):
    observation  = env.reset()
    terminal = False
    total_score_ = 0
    reward = 0.0  # initial reward is assumed to be zero
    step_in_episode = 0
    while True:
        env.render()
        if step_in_episode == 0:
            observation, reward, terminal, info = env.step(agent.start(observation)) # take an action
        else:
            observation, reward, terminal, info = env.step(agent.act(observation, reward)) # take an action

        total_score_ += reward
        step_in_episode += 1

        if terminal is True:
            agent.end(reward)
            break

    total_score[i_episode] = total_score_
    print("REWARD@" + str(i_episode) + "-th episode : " + str(total_score_))
    if np.mod(i_episode, 50) == 0:
        plt.figure(0)
        plt.clf()
        plt.plot(total_reward)
        plt.legend(["Total Reward"])
        plt.draw()
        plt.pause(0.001)

        # Save the current agent parameters
        agent.save()
