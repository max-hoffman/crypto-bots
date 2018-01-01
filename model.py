import random as rd
import matplotlib.pyplot as plt
import pylab
import numpy as np
from collections import deque
import tensorflow as tf
import tensorflow.contrib.layers as layers
from gru import GRUStack
import sys

class RLAgent:
    def __init__(self, state_size, action_size, hidden_size, cell_layers, session, logs_path):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = .1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.5
        self.session = session
        self.training_step = 0
        self.trade_amount = .01
        self.hidden_state = np.zeros([1,hidden_size])
        rd.seed()

        NON_ZERO_PENALTY = 1
        TINY          = 1e-6    # to avoid NaNs in logs
        OUTPUT_SIZE = 1
        TIMESTEPS = 1

        with tf.name_scope('input'):
            self.x_in = tf.placeholder(tf.float32, shape=[state_size])
            self.s_start = tf.placeholder(tf.float32, shape=[1, hidden_size])
            self.y_target = tf.placeholder(tf.float32, shape=[])
            self.step_reward = tf.placeholder(tf.float32, shape=[])

        with tf.name_scope('model'):
            self.y_out = tf.placeholder(tf.float32, shape=[1, action_size])
            self.y_out, self.s_new = GRUStack(self.x_in, self.s_start, cell_layers)

        with tf.name_scope('error'):
            self.error = tf.reduce_sum(tf.squared_difference(self.y_out, tf.expand_dims(self.y_target,0))) - self.step_reward

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.error)
        # tensorboard tracking
        tf.summary.scalar("error", self.error)
        tf.summary.histogram("expected reward", self.y_out)
        tf.summary.histogram("state", self.x_in)
        self.summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        self.writer = tf.summary.FileWriter(logs_path , session.graph)
    
    def train(self, state, target, reward, iteration):
        self.training_step += 1
        state = np.array(state)
        target = np.array(target)
        _, s_current, summary = self.session.run([self.train_op, self.s_new, self.summary_op],
                                                 feed_dict={ self.x_in: state,
                                                             self.s_start: self.hidden_state,
                                                             self.y_target: target,
                                                             self.step_reward: reward })
        self.writer.add_summary(summary, iteration)
        self.training_step += 1
        self.hidden_state = s_current
        return

    def action(self, state): 
        # state = [usd, eth, eth_price]
        if self.epsilon > self.epsilon_min:
            if self.training_step % 50 == 0:
                self.epsilon *= self.epsilon_decay
        if np.random.rand() <= self.epsilon:
            return rd.randrange(0, 3)

        hidden_state = self.hidden_state # we want same hidden state for each
        q = np.zeros([3,1]) # hold, buy, sell
        states_actions = np.zeros([3,4])

        states_actions[0] = np.array([state[0], state[1], state[2], 0])
        states_actions[1] = np.array([state[0], state[1], state[2], 1])
        states_actions[2] = np.array([state[0], state[1], state[2], 2])

        q[0] = self.session.run(self.y_out, feed_dict={ self.x_in: states_actions[0], self.s_start: hidden_state })
        q[1] = self.session.run(self.y_out, feed_dict={ self.x_in: states_actions[1], self.s_start: hidden_state })
        q[2] = self.session.run(self.y_out, feed_dict={ self.x_in: states_actions[2], self.s_start: hidden_state })

        self.hidden_state = hidden_state
        # print(q)
        return np.argmax(q)

    def step(self, state, action):
        usd, eth, eth_price = state

        if action == 1:                                   # buy eth
            usd_sell = .05 * usd                            # buy with 5% of remaining usd
            buy_price = eth_price * (1 + .005)              # .5% markup
            eth_buy = (usd_sell / buy_price) * (1 - .0025)  # .25% trade fee
            usd = usd - usd_sell
            eth = eth + eth_buy
        elif action == 2:                                   # sell eth
            eth_sell = .05 * eth                            # sell 5% of remaining eth
            sell_price = eth_price * (1 - .005)             # .5% markdown
            usd_buy = eth_sell * sell_price * (1 - .0025)   # .25% trade fee
            usd = usd + usd_buy
            eth = eth - eth_sell
        return usd, eth
        
    def kill(self):
        self.writer.close()
        sys.exit()
