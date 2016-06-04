# -*- coding: utf-8 -*-
"""
Deep Q-network implementation with chainer for gym environment
Copyright (c) 2016  Naoto Yoshida All Right Reserved.
"""

import copy

import pickle
import numpy as np
import scipy.misc as spm

from chainer import cuda, Function, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class ActionValue(Chain):
    def __init__(self, n_history, n_act):
        super(ActionValue, self).__init__(
            l1=F.Convolution2D(n_history, 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
            l2=F.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=np.sqrt(2)),
            l3=F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2)),
            l4=F.Linear(3136, 512, wscale=np.sqrt(2)),
            q_value=F.Linear(512, n_act,
                             initialW=np.zeros((n_act, 512),
                             dtype=np.float32))
        )

    def q_function(self, state):
        h1 = F.relu(self.l1(state/255.))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        return self.q_value(h4)


class DQN:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 10**4  # Initial exploratoin. original: 5x10^4
    replay_size = 32  # Replay (batch) size
    target_model_update_freq = 10**4  # Target update frequancy. original: 10^4
    data_size = 10**5  # Data size of history. original: 10^6
    img_size = 84  # 84x84 image input (fixed)

    def __init__(self, n_history, n_act):
        print("Initializing DQN...")
        self.step = 0  # number of steps that DQN is updated
        self.n_act = n_act
        self.n_history = n_history  # Number of obervations used to construct the single state

        print("Model Building")
        self.model = ActionValue(n_history, n_act).to_gpu()
        self.model_target = copy.deepcopy(self.model)

        print("Initizlizing Optimizer")
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.setup(self.model)

        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        hs = self.n_history
        ims = self.img_size
        self.replay_buffer = [np.zeros((self.data_size, hs, ims, ims), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.float32),
                  np.zeros((self.data_size, hs, ims, ims), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

    def get_loss(self, state, action, reward, state_prime, episode_end):
        s = Variable(cuda.to_gpu(state))
        s_dash = Variable(cuda.to_gpu(state_prime))

        q = self.model.q_function(s)  # Get Q-value

        # Generate Target Signals
        tmp = self.model_target.q_function(s_dash)  # Q(s',*)
        tmp = list(map(np.max, tmp.data))  # max_a Q(s',a)
        max_q_prime = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(copy.deepcopy(q.data.get()), dtype=np.float32)

        for i in range(self.replay_size):
            if episode_end[i][0] is True:
                tmp_ = np.sign(reward[i])
            else:
                #  The sign of reward is used as the reward of DQN!
                tmp_ = np.sign(reward[i]) + self.gamma * max_q_prime[i]

            target[i, action[i]] = tmp_

        # TD-error clipping
        td = Variable(cuda.to_gpu(target)) - q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero_val = Variable(cuda.to_gpu(np.zeros((self.replay_size, self.n_act), dtype=np.float32)))
        loss = F.mean_squared_error(td_clip, zero_val)
        return loss, q

    def stock_experience(self, time,
                        state, action, reward, state_prime,
                        episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.replay_buffer[0][data_index] = state
            self.replay_buffer[1][data_index] = action
            self.replay_buffer[2][data_index] = reward
        else:
            self.replay_buffer[0][data_index] = state
            self.replay_buffer[1][data_index] = action
            self.replay_buffer[2][data_index] = reward
            self.replay_buffer[3][data_index] = state_prime
        self.replay_buffer[4][data_index] = episode_end_flag

    def experience_replay(self, time):

        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            hs = self.n_history
            ims = self.img_size
            rs = self.replay_size

            s_replay = np.ndarray(shape=(rs, hs, ims, ims), dtype=np.float32)
            a_replay = np.ndarray(shape=(rs, 1), dtype=np.int8)
            r_replay = np.ndarray(shape=(rs, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(rs, hs, ims, ims), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(rs, 1), dtype=np.bool)
            for i in range(self.replay_size):
                s_replay[i] = np.asarray(self.replay_buffer[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.replay_buffer[1][replay_index[i]]
                r_replay[i] = self.replay_buffer[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.replay_buffer[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.replay_buffer[4][replay_index[i]]

            # Gradient-based update
            self.optimizer.zero_grads()
            loss, _ = self.get_loss(s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay)
            loss.backward()
            self.optimizer.update()


    def action_sample_e_greedy(self, state, epsilon):
        s = Variable(cuda.to_gpu(state))
        q = self.model.q_function(s)
        q = q.data.get()[0]

        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.n_act)
            print("RANDOM : " + str(action))
        else:
            a = np.argmax(q)
            print("GREEDY  : " + str(a))
            action = np.asarray(a, dtype=np.int8)
            print(q)
        return action, q

    def target_model_update(self, soft_update):
        if soft_update is True:
            tau = self.target_update_rate

            # Target preference Update
            model_params = dict(self.model.namedparams())
            model_target_params = dict(self.model_target.namedparams())
            for name in model_target_params:
                model_target_params[name].data = tau*model_params[name].data\
                                        + (1 - tau)*model_target_params[name].data
        else:
            if np.mod(self.step, self.target_model_update_freq) == 0:
                self.model_target = copy.deepcopy(self.model)

    def learn(self, state, action, reward, state_prime, terminal):
        self.stock_experience(self.step,
                         state, action, reward, state_prime,
                         terminal)

        self.experience_replay(self.step)
        self.target_model_update(soft_update=False)

        self.step += 1


class DQN_Agent:  # RL-glue Process
    policyFrozen = False

    def __init__(self, env):

        self.epsilon = 1.0  # Initial exploratoin rate

        # Pick a DQN from DQN_class
        self.dqn = DQN(n_history=4, n_act=env.action_space.n)

    def start(self, observation):

        self.reset_state(observation)
        state_ = np.asanyarray(self.state.reshape(1, 4, 84, 84), dtype=np.float32)

        # Generate an Action e-greedy
        action, Q_now = self.dqn.action_sample_e_greedy(state_, self.epsilon)

        # Update for next step
        self.last_action = action
        self.last_state = copy.deepcopy(self.state)

        return action

    def act(self, observation, reward):

        self.set_state(observation)
        state_ = np.asanyarray(self.state.reshape(1, self.dqn.n_history, 84, 84), dtype=np.float32)

        # Exploration decays along the time sequence
        if self.policyFrozen is False:  # Learning ON/OFF
            if self.dqn.initial_exploration < self.dqn.step:
                self.epsilon -= 1.0/10**6
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
                eps = self.epsilon
            else:  # Initial Exploation Phase
                print("Initial Exploration : %d/%d steps" % (self.dqn.step, self.dqn.initial_exploration))
                eps = 1.0
        else:  # Evaluation
                print("Policy is Frozen")
                eps = 0.05

        # Generate an Action by e-greedy action selection
        action, Q_now = self.dqn.action_sample_e_greedy(state_, eps)

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.dqn.learn(self.last_state, self.last_action, reward, self.state, False)
            self.last_action = copy.deepcopy(action)
            self.last_state = self.state.copy()

        # Simple text based visualization
        print(' Time Step %d /   ACTION  %d  /   REWARD %.1f   / EPSILON  %.6f  /   Q_max  %3f' % (self.dqn.step, action, np.sign(reward), eps, np.max(Q_now)))

        return action

    def end(self, reward):  # Episode Terminated

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.dqn.learn(self.last_state, self.last_action, reward, self.last_state, True)

        # Simple text based visualization
        print('  REWARD %.1f   / EPSILON  %.5f' % (np.sign(reward), self.epsilon))


    def reset_state(self, observation):
        # Preprocess
        obs_array = self.scale_image(observation)
        # Updates for next step
        self.last_observation = obs_array

        # Initialize State
        self.state = np.zeros((self.dqn.n_history, 84, 84), dtype=np.uint8)
        self.state[0] = obs_array

    def set_state(self, observation):
        # Preproces
        obs_array = self.scale_image(observation)
        obs_processed = np.maximum(obs_array, self.last_observation)  # Take maximum from two frames

        # Updates for the next step
        self.last_observation = obs_array

        # Compose State : 4-step sequential observation
        for i in range(self.dqn.n_history - 1):
            self.state[i] = self.state[i + 1].astype(np.uint8)
        self.state[self.dqn.n_history - 1] = obs_processed.astype(np.uint8)

    def scale_image(self, observation):
        img = self.rgb2gray(observation)  # Convert RGB to Grayscale
        return (spm.imresize(img, (110, 84)))[110-84-8:110-8, :]  # Scaling

    def rgb2gray(self, image):
        return np.dot(image[...,:3], [0.299, 0.587, 0.114])

    def save(self):
        serializers.save_npz('network/model.model', self.dqn.model)
        serializers.save_npz('network/model_target.model',
                             self.dqn.model_target)

        print("------------ Networks were SAVED ---------------")
