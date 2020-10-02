import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time


class AccumulatingOptimizer(object):
    def __init__(own, opt, var_list):
        own.opt = opt
        own.var_list = var_list
        own.accum_vars = {tv : tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
                           for tv in var_list}
        own.total_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))
        own.count_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))

    def reset(own):
        updates = [tv.assign(tf.zeros_like(tv)) for tv in own.accum_vars.values()]
        updates.append(own.total_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        updates.append(own.count_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def compute_gradients(own, loss):
        grads = own.opt.compute_gradients(loss, own.var_list)
        updates = [own.accum_vars[v].assign_add(g) for (g,v) in grads]
        updates.append(own.total_loss.assign_add(loss))
        updates.append(own.count_loss.assign_add(1.0))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def apply_gradients(own):
        grads = [(g,v) for (v,g) in own.accum_vars.items()]
        with tf.control_dependencies([own.opt.apply_gradients(grads)]):
            return own.total_loss / own.count_loss
