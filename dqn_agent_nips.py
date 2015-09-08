# -*- coding: utf-8 -*-
"""
Deep Q-network implementation with chainer and rlglue
Copyright (c) 2015 Naoto Yoshida All Right Reserved.
"""

import copy

import pickle
import numpy as np
import scipy.misc as spm

from chainer import cuda, FunctionSet, Variable, optimizers
import chainer.functions as F

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action


class DQN_class:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 10**4  # Initial exploratoin. original: 5x10^4
    replay_size = 32  # Replay (batch) size
    target_model_update_freq = 10**4  # Target update frequancy. original: 10^4
    data_size = 10**5  # Data size of history. original: 10^6

    def __init__(self, enable_controller=[0, 3, 4]):
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller  # Default setting : "Pong"

        print "Initializing DQN..."
#	Initialization for Chainer 1.1.0 or older.
#        print "CUDA init"
#        cuda.init()

        print "Model Building"
        self.model = FunctionSet(
            l1=F.Convolution2D(4, 16, ksize=8, stride=4, wscale=np.sqrt(2)),
            l2=F.Convolution2D(16, 32, ksize=4, stride=2, wscale=np.sqrt(2)),
            l3=F.Linear(2592, 256),
            q_value=F.Linear(256, self.num_of_actions,
                             initialW=np.zeros((self.num_of_actions, 256),
                                               dtype=np.float32))
        ).to_gpu()

        print "Initizlizing Optimizer"
        self.optimizer = optimizers.RMSpropGraves(lr=0.0002, alpha=0.3, momentum=0.2)
        self.optimizer.setup(self.model.collect_parameters())

        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        self.D = [np.zeros((self.data_size, 4, 84, 84), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.int8),
                  np.zeros((self.data_size, 4, 84, 84), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

    def forward(self, state, action, Reward, state_dash, episode_end):
        num_of_batch = state.shape[0]
        s = Variable(state)
        s_dash = Variable(state_dash)

        Q = self.Q_func(s)  # Get Q-value

        # Generate Target Signals
        max_Q_dash_ = self.Q_func(s_dash)
        tmp = list(map(np.max, max_Q_dash_.data.get()))
        max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(Q.data.get(), dtype=np.float32)

        for i in xrange(num_of_batch):
            if not episode_end[i][0]:
                tmp_ = np.sign(Reward[i]) + self.gamma * max_Q_dash[i]
            else:
                tmp_ = np.sign(Reward[i])
            target[i, self.action_to_index(action[i])] = tmp_

        loss = F.mean_squared_error(Variable(cuda.to_gpu(target)), Q)
        return loss, Q

    def stockExperience(self, time,
                        state, action, reward, state_dash,
                        episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.D[0][data_index] = state
            self.D[1][data_index] = action
            self.D[2][data_index] = reward
        else:
            self.D[0][data_index] = state
            self.D[1][data_index] = action
            self.D[2][data_index] = reward
            self.D[3][data_index] = state_dash
        self.D[4][data_index] = episode_end_flag

    def experienceReplay(self, time):

        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            s_replay = np.ndarray(shape=(self.replay_size, 4, 84, 84), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, 4, 84, 84), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in xrange(self.replay_size):
                s_replay[i] = np.asarray(self.D[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.D[1][replay_index[i]]
                r_replay[i] = self.D[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.D[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.D[4][replay_index[i]]

            s_replay = cuda.to_gpu(s_replay)
            s_dash_replay = cuda.to_gpu(s_dash_replay)

            # Gradient-based update
            self.optimizer.zero_grads()
            loss, _ = self.forward(s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay)
            loss.backward()
            self.optimizer.update()

    def Q_func(self, state):
        h1 = F.relu(self.model.l1(state / 254.0))  # scale inputs in [0.0, 1.0]
        h2 = F.relu(self.model.l2(h1))
        h3 = F.relu(self.model.l3(h2))
        Q = self.model.q_value(h3)
        return Q

    def e_greedy(self, state, epsilon):
        s = Variable(state)
        Q = self.Q_func(s)
        Q = Q.data

        if np.random.rand() < epsilon:
            index_action = np.random.randint(0, self.num_of_actions)
            print "RANDOM"
        else:
            index_action = np.argmax(Q.get())
            print "GREEDY"

        return self.index_to_action(index_action), Q

    def index_to_action(self, index_of_action):
        return self.enable_controller[index_of_action]

    def action_to_index(self, action):
        return self.enable_controller.index(action)


class dqn_agent(Agent):  # RL-glue Process
    lastAction = Action()
    policyFrozen = False

    def agent_init(self, taskSpec):
        # Some initializations for rlglue
        self.lastAction = Action()

        self.time = 0
        self.epsilon = 1.0  # Initial exploratoin rate

        # Pick a DQN from DQN_class
        self.DQN = DQN_class()  # Default is for "Pong".

    def agent_start(self, observation):

        # Get intensity from current observation array
        tmp = np.bitwise_and(np.asarray(observation.intArray[128:]).reshape([210, 160]), 0b0001111)  # Get Intensity from the observation
        obs_array = (spm.imresize(tmp, (110, 84)))[110-84-8:110-8, :]  # Scaling

        # Initialize State
        self.state = np.zeros((4, 84, 84), dtype=np.uint8)
        self.state[0] = obs_array
        state_ = cuda.to_gpu(np.asanyarray(self.state.reshape(1, 4, 84, 84), dtype=np.float32))

        # Generate an Action e-greedy
        returnAction = Action()
        action, Q_now = self.DQN.e_greedy(state_, self.epsilon)
        returnAction.intArray = [action]

        # Update for next step
        self.lastAction = copy.deepcopy(returnAction)
        self.last_state = self.state.copy()
        self.last_observation = obs_array

        return returnAction

    def agent_step(self, reward, observation):

        # Preproces
        tmp = np.bitwise_and(np.asarray(observation.intArray[128:]).reshape([210, 160]), 0b0001111)  # Get Intensity from the observation
        obs_array = (spm.imresize(tmp, (110, 84)))[110-84-8:110-8, :]  # Scaling
        obs_processed = np.maximum(obs_array, self.last_observation)  # Take maximum from two frames

        # Compose State : 4-step sequential observation
        self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], obs_processed], dtype=np.uint8)
        state_ = cuda.to_gpu(np.asanyarray(self.state.reshape(1, 4, 84, 84), dtype=np.float32))

        # Exploration decays along the time sequence
        if self.policyFrozen is False:  # Learning ON/OFF
            if self.DQN.initial_exploration < self.time:
                self.epsilon -= 1.0/10**6
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
                eps = self.epsilon
            else:  # Initial Exploation Phase
                print "Initial Exploration : %d/%d steps" % (self.time, self.DQN.initial_exploration)
                eps = 1.0
        else:  # Evaluation
                print "Policy is Frozen"
                eps = 0.05

        # Generate an Action from e-greedy action selection
        returnAction = Action()
        action, Q_now = self.DQN.e_greedy(state_, eps)
        returnAction.intArray = [action]

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.DQN.stockExperience(self.time, self.last_state, self.lastAction.intArray[0], reward, self.state, False)
            self.DQN.experienceReplay(self.time)

        # Simple text based visualization
        print ' Time Step %d /   ACTION  %d  /   REWARD %.1f   / EPSILON  %.6f  /   Q_max  %3f' % (self.time, self.DQN.action_to_index(action), np.sign(reward), eps, np.max(Q_now.get()))

        # Updates for next step
        self.last_observation = obs_array
        
        # Update for next step
        if self.policyFrozen is False:
            self.lastAction = copy.deepcopy(returnAction)
            self.last_state = self.state.copy()
            self.time += 1

        return returnAction

    def agent_end(self, reward):  # Episode Terminated

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.DQN.stockExperience(self.time, self.last_state, self.lastAction.intArray[0], reward, self.last_state, True)
            self.DQN.experienceReplay(self.time)

        # Simple text based visualization
        print '  REWARD %.1f   / EPSILON  %.5f' % (np.sign(reward), self.epsilon)

        # Time count
        if not self.policyFrozen:
            self.time += 1

    def agent_cleanup(self):
        pass

    def agent_message(self, inMessage):
        if inMessage.startswith("freeze learning"):
            self.policyFrozen = True
            return "message understood, policy frozen"

        if inMessage.startswith("unfreeze learning"):
            self.policyFrozen = False
            return "message understood, policy unfrozen"

        if inMessage.startswith("save model"):
            with open('dqn_model.dat', 'w') as f:
                pickle.dump(self.DQN.model, f)
            return "message understood, model saved"

if __name__ == "__main__":
    AgentLoader.loadAgent(dqn_agent())
