import random as rd
import matplotlib.pyplot as plt
import pylab
import numpy as np
from collections import deque
import tensorflow as tf
import sys

class RLAgent:
    def __init__(self, state_size, action_size, sesh, logs_path):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = .1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.sesh = sesh
        self.training_step = 0

        RNN_HIDDEN = 20
        NON_ZERO_PENALTY = 1
        TINY          = 1e-6    # to avoid NaNs in logs

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=(None, None, self.state_size)) # (batch, time, in)
            self.y_ = tf.placeholder(tf.float32, shape=(None, None, self.action_size))

        with tf.name_scope('forward-pass'):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=RNN_HIDDEN)

            batch_size    = tf.shape(self.x)[0]
            initial_state = cell.zero_state(batch_size, tf.float32)

            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)

            self.y  = layers.linear(rnn_outputs, num_outputs=OUTPUT_SIZE, activation_fn=None)

        with tf.name_scope('squared-error'):
            self.squared_error = tf.reduce_sum(tf.squared_difference(self.y, self.y_))

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.squared_error)

        # track cost and accuracy
        tf.summary.scalar("error", self.squared_error)
        tf.summary.scalar("expected reward", self.y)
        self.summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        self.writer = tf.summary.FileWriter(logs_path , sesh.graph)
    
    def train(self, state, target, iteration):
        _, summary = self.sesh.run([self.train_op, self.summary_op], feed_dict={ self.x: [state], self.y_: target })
        # print("summary", summary)
        # print(self.writer)
        self.writer.add_summary(summary, iteration)
        self.training_step += 1
        if self.training_step % 750 == 0:
            print("Epoch, step: ", epoch, self.training_step)
        return

    def _predict(self, state):
        return self.sesh.run(self.y, feed_dict={ self.x: [state] })

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            # TODO: change this to pick the lowest one sometimes
            if np.random.rand() <= .5:
                return nonzero_argmin(state)
            return nonzero_argmin(state) + state.shape[0] / 2 + 1
            # return rd.randrange(self.action_size)
        if np.random.rand() <= self.epsilon:
            return rd.randrange(self.action_size)
        q = self._predict(state)
        return np.argmax(q)

    def step(self, state, action):
        # change the state according to the action
        sparsify = True
        next_state = state
        if action == 52:
            return next_state, -1, False
        elif action > self.state_size - 1:
            action = action % self.state_size
            sparsify = False

        next_state[action] = 0 if sparsify else 1
        next_state = self._lsqr(next_state, theta, dX)
        reward, done = self._reward(next_state, oracle)

        return next_state, reward, done
        
    def kill(self):
        self.writer.close()
        sys.exit()
