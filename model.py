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
        self.trade_amount = .03
        self.trade_max = .10
        self.trade_grow = 1.10
        self.hidden = np.zeros([8, 2, action_size, hidden_size])
        rd.seed()

        NON_ZERO_PENALTY = 1
        TINY          = 1e-6    # to avoid NaNs in logs
        OUTPUT_SIZE = 1
        TIMESTEPS = 1

        with tf.name_scope('input'):
            # self.x_in = tf.placeholder(tf.float32, shape=[state_size])
            # self.s_start = tf.placeholder(tf.float32, shape=[cell_layers, action_size, hidden_size])
            self.y_target = tf.placeholder(tf.float32, shape=[])
            self.step_reward = tf.placeholder(tf.float32, shape=[])
            self.x_in = tf.placeholder(tf.float32, [1, 1, state_size])
            self.y_out = tf.placeholder(tf.float32, [1, action_size])
            self.hidden_start = tf.placeholder(tf.float32, [cell_layers, 2, action_size, hidden_size])

        with tf.name_scope('model'):
            # self.y_out, self.s_new = GRUStack(self.x_in, self.s_start, cell_layers)

            # organize cells in layers, side-step tf bugs
            layers = tf.unstack(self.hidden_start, axis=0)
            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(layers[idx][0], layers[idx][1])
                for idx in range(cell_layers)]
            )
            def lstm_cell():
                return tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(cell_layers)], state_is_tuple=True)

            # actual transformations
            outputs, self.hidden_new = tf.nn.dynamic_rnn(stacked_lstm, self.x_in, initial_state=rnn_tuple_state)
            w_sy = tf.Variable(tf.random_normal([hidden_size,action_size], stddev=.1))
            b_y = tf.Variable(tf.random_normal([action_size], stddev=.1))
            self.y_out = tf.matmul(outputs[-1], w_sy) + b_y

        with tf.name_scope('error'):
            self.error = tf.reduce_sum(tf.squared_difference(self.y_out, tf.expand_dims(self.y_target,0))) - self.step_reward

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.error)

        # tensorboard tracking
        tf.summary.scalar("error", self.error)
        tf.summary.histogram("expected reward", self.y_out)
        tf.summary.histogram("state", self.x_in)
        tf.summary.histogram("final layer weight", w_sy)
        tf.summary.histogram("final layer bias", b_y)
        self.summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        self.writer = tf.summary.FileWriter(logs_path , session.graph)
    
    def train(self, state, target, reward, iteration):
        self.training_step += 1
        state = np.array(state)
        state = state.reshape([1,1,4])
        target = np.array(target)
        _, hidden_new, summary = self.session.run([self.train_op, self.hidden_new, self.summary_op],
                                                 feed_dict={ self.x_in: state,
                                                             self.hidden_start: self.hidden,
                                                             self.y_target: target,
                                                             self.step_reward: reward })
        self.writer.add_summary(summary, iteration)
        self.training_step += 1
        self.hidden = np.array(hidden_new)
        return

    def action(self, state): 
        # state = [usd, eth, eth_price]
        if self.epsilon > self.epsilon_min:
            if self.training_step % 50 == 0:
                self.epsilon *= self.epsilon_decay
        if rd.random() <= self.epsilon:
            return rd.randrange(0, 3)

        hidden_state = self.hidden # we want same hidden state for each
        q = np.zeros([3,1]) # hold, buy, sell
        states_actions = np.zeros([3, 1, 1, 4])

        states_actions[0][0][0] = np.array([state[0], state[1], state[2], 0])
        states_actions[1][0][0] = np.array([state[0], state[1], state[2], 1])
        states_actions[2][0][0] = np.array([state[0], state[1], state[2], 2])

        q[0] = self.session.run(self.y_out, feed_dict={ self.x_in: states_actions[0], self.hidden_start: hidden_state })
        q[1] = self.session.run(self.y_out, feed_dict={ self.x_in: states_actions[1], self.hidden_start: hidden_state })
        q[2] = self.session.run(self.y_out, feed_dict={ self.x_in: states_actions[2], self.hidden_start: hidden_state })

        self.hidden_state = hidden_state
        return np.argmax(q)

    def step(self, state, action):
        usd, eth, eth_price = state
        # if self.training_step % 50 == 0:
        #     if self.trade_amount < self.trade_max:
        #         self.trade_amount *= self.trade_grow

        if action == 1:                                     # buy eth
            usd_sell = self.trade_amount * usd              # buy with 5% of remaining usd
            buy_price = eth_price * (1 + .005)              # .5% markup
            eth_buy = (usd_sell / buy_price) * (1 - .0025)  # .25% trade fee
            usd = usd - usd_sell
            eth = eth + eth_buy
        elif action == 2:                                   # sell eth
            eth_sell = self.trade_amount * eth              # sell 5% of remaining eth
            sell_price = eth_price * (1 - .005)             # .5% markdown
            usd_buy = eth_sell * sell_price * (1 - .0025)   # .25% trade fee
            usd = usd + usd_buy
            eth = eth - eth_sell
        return usd, eth
        
    def kill(self):
        self.writer.close()
        sys.exit()
