import random as rd
import matplotlib.pyplot as plt
import pylab
import numpy as np
from collections import deque
import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys

class RLAgent:
    def __init__(self, state_size, action_size, session, logs_path):
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
        self.hidden_state = np.zeros([1,32])
        rd.seed()

        RNN_HIDDEN = 20
        hidden_size = 32
        NON_ZERO_PENALTY = 1
        TINY          = 1e-6    # to avoid NaNs in logs
        OUTPUT_SIZE = 1
        TIMESTEPS = 1

        with tf.name_scope('input'):
            self.x_in = tf.placeholder(tf.float32, shape=[state_size]) # (batch, time, in)
            self.h_start = tf.placeholder(tf.float32, shape=[1, hidden_size]) # (batch, time, in)
            self.y_out = tf.placeholder(tf.float32, shape=[1, action_size])
            self.y_target = tf.placeholder(tf.float32, shape=[])
            self.step_reward = tf.placeholder(tf.float32, shape=[])

        # with tf.name_scope('forward-pass'):
        #     cell = tf.nn.rnn_cell.LSTMCell(num_units=RNN_HIDDEN)

        #     batch_size    = tf.shape(self.x_in)[0]
        #     initial_state = cell.zero_state(batch_size, tf.float32)

        #     # rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, self.x_in, initial_state=initial_state)
        #     rnn_outputs, rnn_states = tf.nn.static_rnn(cell, [self.x_in], initial_state=initial_state)

        #     self.y  = layers.linear(rnn_outputs, num_outputs=OUTPUT_SIZE, activation_fn=None)

        initializer = tf.random_normal_initializer(stddev=0.1)
        with tf.variable_scope('rnn') as rnn_scope:
            # assert rnn_scope.reuse == True
            h_state = self.h_start

            w_xh = tf.get_variable("w_xh", [state_size,hidden_size], initializer=initializer)
            w_hh = tf.get_variable("w_hh", [hidden_size,hidden_size], initializer=initializer)
            w_hy = tf.get_variable("w_hy",[hidden_size,action_size], initializer=initializer)
            b_h = tf.get_variable("b_h", [hidden_size],initializer=initializer)
            b_y = tf.get_variable("b_y",[action_size], initializer=initializer)

            h_state = tf.tanh(tf.matmul(tf.expand_dims(self.x_in,0), w_xh) + tf.matmul(h_state, w_hh) + b_h)
            self.y_out = tf.matmul(h_state, w_hy) + b_y

            self.h_last = h_state

        with tf.name_scope('error'):
            self.error = tf.reduce_sum(tf.squared_difference(self.y_out, tf.expand_dims(self.y_target,0))) - self.step_reward

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.error)

        # track cost and accuracy
        tf.summary.scalar("error", self.error)
        tf.summary.histogram("expected reward", self.y_out)
        self.summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        self.writer = tf.summary.FileWriter(logs_path , session.graph)
    
    def train(self, state, target, reward, iteration):
        self.training_step += 1
        state = np.array(state)
        target = np.array(target)
        _, h_state, summary = self.session.run([self.train_op, self.h_last, self.summary_op],
                                               feed_dict={ self.x_in: state,
                                                           self.h_start: self.hidden_state,
                                                           self.y_target: target,
                                                           self.step_reward: reward })
        self.writer.add_summary(summary, iteration)
        self.training_step += 1
        self.hidden_state = h_state
        return h_state

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

        q[0] = self.session.run(self.y_out, feed_dict={ self.x_in: states_actions[0], self.h_start: hidden_state })
        q[1] = self.session.run(self.y_out, feed_dict={ self.x_in: states_actions[1], self.h_start: hidden_state })
        q[2] = self.session.run(self.y_out, feed_dict={ self.x_in: states_actions[2], self.h_start: hidden_state })

        self.hidden_state = hidden_state
        # print(q)
        return np.argmax(q)

    def step(self, state, action):
        usd, eth, eth_price = state

        # if isInactive(state, action):
        #     return usd, eth
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
        # change the state according to the action
        
    def kill(self):
        self.writer.close()
        sys.exit()

def isInactive(state, action):
    usd, eth, eth_price = state
    if action == 0:
        return True
    elif action == 2 and eth == 0:
        return True
    elif action == 1 and usd == 0:
        return True
    return False