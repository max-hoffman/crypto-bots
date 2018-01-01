import tensorflow as tf
import numpy as np

def GRUStack(x_in, s_start, cell_layers=8, state_size=4, hidden_size=32, action_size=1):
    with tf.name_scope('gru'):
        s_new = []
        h, s = GRUCell(tf.expand_dims(x_in,0), state_size, s_start[0], hidden_size)
        s_new.append(s)
        if cell_layers > 1:
            for i in range(1, cell_layers):
                h, s = GRUCell(h, hidden_size, s_start[i], hidden_size)
                s_new.append(s)
        
        w_sy = tf.Variable(tf.random_normal([hidden_size,action_size], stddev=.1))
        b_y = tf.Variable(tf.random_normal([action_size], stddev=.1))
        y_out = tf.matmul(s_new[-1], w_sy) + b_y

        return y_out, s_new

def GRUCell(x_in, input_size, s_start, hidden_size):
    with tf.name_scope('gru-cell'):

        u_xz = tf.Variable(tf.random_normal([input_size,hidden_size], stddev=.1))
        u_xr = tf.Variable(tf.random_normal([input_size,hidden_size], stddev=.1))
        u_xh = tf.Variable(tf.random_normal([input_size,hidden_size], stddev=.1))

        w_sz = tf.Variable(tf.random_normal([hidden_size,hidden_size], stddev=.1))
        w_sh = tf.Variable(tf.random_normal([hidden_size,hidden_size], stddev=.1))
        w_sr = tf.Variable(tf.random_normal([hidden_size,hidden_size], stddev=.1))

        b_z = tf.Variable(tf.random_normal([hidden_size], stddev=.1))
        b_r = tf.Variable(tf.random_normal([hidden_size], stddev=.1))
        b_h = tf.Variable(tf.random_normal([hidden_size], stddev=.1))

        z = tf.sigmoid(tf.matmul(x_in, u_xz) + tf.matmul(s_start, w_sz) + b_z)
        r = tf.sigmoid(tf.matmul(x_in, u_xr) + tf.matmul(s_start, w_sr) + b_r)
        h = tf.tanh(tf.matmul(x_in, u_xh) + tf.matmul(tf.multiply(s_start, r), w_sh) + b_h)
        s_new = tf.add(tf.multiply((1 - z), h), tf.multiply(z, s_start))
    return h, s_new
   