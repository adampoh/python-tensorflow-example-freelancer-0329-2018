#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:23:32 2018

@author: cameron
"""

#code much from EZGAN github

from __future__ import absolute_import, division, print_function
import numbers
import numpy as np
import scipy as sp
from scipy import stats
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
from sklearn.mixture import BayesianGaussianMixture
import tensorflow as tf
import csv
np.random.seed(112)
tf.set_random_seed(232)



def selu(x):
    with ops.name_scope('elu') as scope1:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
    
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        alpha.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = math_ops.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * math_ops.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))

data = []
            
# with open("/home/cameron/AnacondaProjects/Bayesian_DSGE/dataforblp.csv", "r") as csvfile2:
with open("dataforblp.csv", "r") as csvfile2:
        reader2 = csv.reader(csvfile2)
        data = np.asarray(list(reader2), dtype = 'float32')
        
datalist = [[] for each in range(20)]
datasum = [0.0 for each in range(20)]


for each in data:
    flag1 = True
    for i in range(6,25):
        if each[i]==1:
            datasum[i-6] = datasum[i-6] + float(each[25])
            datalist[i-6].append(each.tolist())
            flag1 = False
    if flag1:
        datalist[19].append(each.tolist())
        datasum[19] = datasum[19] + float(each[25])

for i,each in enumerate(datalist):
    vec = [0.0 for each in range(26)]
    vec[i+6] = 1.0
    vec[25] = 1.0-datasum[i]
    datalist[i].append(vec)
    
    


def discriminator(x_data, y_data, drate, is_training, market, reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()
    
    hidden = 64
    
    input111 = tf.concat([x_data, y_data, market], 1)
    
    features = 26
    
    
    
    sdev = np.sqrt(float(1/features))
    
    w1_d = tf.get_variable("w1_d",[features, hidden],initializer = tf.random_normal_initializer(stddev=sdev))
    sdev = np.sqrt(float(1/hidden))
    
    weights2_d = tf.get_variable("w2_d", [hidden, hidden], initializer = tf.random_normal_initializer(stddev=sdev))
    
#    weights3 = tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights4 = tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights5 = tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights6 = tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights7 = tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights8 = tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights9 = tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights10 = tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights11 = tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights12= tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights13= tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights14= tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights15= tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
#    weights16= tf.Variable(tf.random_normal([hidden, hidden],stddev=sdev))
#    
    
    weights17_d = tf.get_variable("w3_d", [hidden, 1], initializer = tf.random_normal_initializer(stddev=sdev))
    
    bias1_d = tf.get_variable("b1_d", [hidden], initializer = tf.random_normal_initializer(stddev=0))
    bias2_d = tf.get_variable("b2_d", [hidden],initializer = tf.random_normal_initializer(stddev=0))
#    bias3 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias4 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias5 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias6 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias7 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias8 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias9 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias10 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias11= tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias12 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias13 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias14 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias15 = tf.Variable(tf.random_normal([hidden],stddev=0))
#    bias16 = tf.Variable(tf.random_normal([hidden],stddev=0))
    bias17_d = tf.get_variable("b3_d", [], initializer = tf.random_normal_initializer(stddev=0))
    
    
    mul_d = tf.matmul(input111, w1_d) + bias1_d
    layer1_d = dropout_selu(selu(mul_d), rate = drate, training = is_training)
    layer16_d = dropout_selu(selu(tf.matmul(layer1_d, weights2_d) + bias2_d + layer1_d), rate = drate, training = is_training)
#    layer3 = dropout_selu(selu(tf.matmul(layer2, weights3) + bias3 + layer2), rate = drate, training = is_training)
#    layer4 = dropout_selu(selu(tf.matmul(layer3, weights4) + bias4 + layer3), rate = drate, training = is_training)
#    layer5 = dropout_selu(selu(tf.matmul(layer4, weights5) + bias5 + layer4), rate = drate, training = is_training)
#    layer6 = dropout_selu(selu(tf.matmul(layer5, weights6) + bias6 + layer5), rate = drate, training = is_training)
#    layer7 = dropout_selu(selu(tf.matmul(layer6, weights7) + bias7 + layer6), rate = drate, training = is_training)
#    layer8 = dropout_selu(selu(tf.matmul(layer7, weights8) + bias8 + layer7), rate = drate, training = is_training)
#    layer9 = dropout_selu(selu(tf.matmul(layer8, weights9) + bias9 + layer8), rate = drate, training = is_training)
#    layer10 = dropout_selu(selu(tf.matmul(layer9, weights10) + bias10 + layer9), rate = drate, training = is_training)
#    layer11 = dropout_selu(selu(tf.matmul(layer10, weights11) + bias11 + layer10), rate = drate, training = is_training)
#    layer12 = dropout_selu(selu(tf.matmul(layer11, weights12) + bias12 + layer11), rate = drate, training = is_training)
#    layer13 = dropout_selu(selu(tf.matmul(layer12, weights13) + bias13 + layer12), rate = drate, training = is_training)
#    layer14 = dropout_selu(selu(tf.matmul(layer13, weights14) + bias14 + layer13), rate = drate, training = is_training)
#    layer15 = dropout_selu(selu(tf.matmul(layer14, weights15) + bias15 + layer14), rate = drate, training = is_training)
#    layer16 = dropout_selu(selu(tf.matmul(layer15, weights16) + bias16 + layer15), rate = drate, training = is_training)
    out_d = tf.matmul(layer16_d, weights17_d) + bias17_d
    sigout = tf.sigmoid(out_d)*.9999998+.0000001               
                       
                       
    return sigout
    
def generator(x_in, drate, is_training, samples):
    
    demlen = 4
    normdraws = np.random.randn()
    demodraws = np.random.rand(demlen,1)
    
    #normdraws =  normdraws.astype(np.float32)
    demodraws = demodraws.astype(np.float32)
    
    
    features = 6
    
    #tf.gather(samples,0)
    
    hidden = 64
    
    
    sdev = tf.sqrt(1.0/features)
    
    weights1_g = tf.get_variable("w1_g", [features, hidden], initializer = tf.random_normal_initializer(stddev=sdev))
    delta_g= tf.get_variable("delta_g", [features, 1], initializer = tf.random_normal_initializer(stddev=sdev))
    demweights_g = tf.get_variable("dw_g", [features, demlen], initializer = tf.random_normal_initializer(stddev=sdev))
    sdev = tf.sqrt(1/hidden)
      
    weights2_g = tf.get_variable("w2_g", [hidden,1], initializer = tf.random_normal_initializer(stddev=sdev))
    
    
    bias1_g = tf.get_variable("b1_g", [hidden], initializer = tf.random_normal_initializer(stddev=0))
    bias2_g = tf.get_variable("b2_g", [], initializer=tf.random_normal_initializer(stddev=0))
    biasdelt_g = tf.get_variable("bdelt2_g", [], initializer=tf.random_normal_initializer(stddev=0))
    
    mul_g = tf.matmul(x_in, weights1_g) + bias1_g
    layer1_g = dropout_selu(selu(mul_g), rate = drate, training = is_training)
    out_g = tf.matmul(layer1_g, weights2_g) + bias2_g
    

    
    utility_g = tf.matmul(x_in,delta_g) + biasdelt_g + tf.exp(out_g)*normdraws+tf.matmul(x_in,tf.matmul(demweights_g,demodraws))
    
    output_g = tf.nn.softmax(utility_g, 0)#*samples
    
    return (x_in, output_g)



sess = tf.Session()

x_placeholder = tf.placeholder("float", shape = [None, 6])
y_placeholder = tf.placeholder("float", shape = [None, 1])
drate1 = tf.placeholder("float")
is_training1= tf.placeholder("bool")
market_place = tf.placeholder("float", shape = [None, 19])
shape1 = tf.placeholder("float", shape = [])

Gz = generator(x_placeholder, drate1, is_training1, shape1)

Dx = discriminator(x_placeholder, y_placeholder, drate1, is_training1, market_place)

Dg = discriminator(Gz[0], Gz[1], drate1, is_training1, market_place, reuse=True)

g_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

d_loss_real = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([tf.shape(x_placeholder)[0], 1], 1.0)))
d_loss_fake = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if '_d' in var.name]
g_vars = [var for var in tvars if '_g' in var.name]
with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
    d_trainer_fake = tf.train.AdamOptimizer(0.0001).minimize(d_loss_fake, var_list=d_vars)
    d_trainer_real = tf.train.AdamOptimizer(0.0001).minimize(d_loss_real, var_list=d_vars)
    # Train the generator
    # Decreasing from 0.004 in GitHub version
    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)


# Sanity check to see how the discriminator evaluates
# generated and real MNIST images
#d_on_generated = tf.reduce_mean(discriminator(generator(x_placeholder, drate1, is_training1, shape12)))
#d_on_real = tf.reduce_mean(discriminator(x_placeholder))

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

xin = datalist[:,:,:6]

market = datalist[:,:, 6:26]

yin = datalist[:,:,26]

#for i,each in enumerate(datalist):
#    for j,each2 in enumerate(each):
#        datalist[i,j,25] = len(each)*datalist[i,j,25]
#        
#yin2 = datalist[:,:,25]

gLoss = 0
dLossFake, dLossReal = 1, 1
d_real_count, d_fake_count, g_count = 0, 0, 0
for i in range(100):
    if dLossFake > 0.6:
        # Train discriminator on generated images
        #TODO change input!!!
        sumlossdr = 0.0
        sumlossdf = 0.0
        sumlossg = 0.0
        for i in range(20):
            _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],
                                                      {shape1 : np.shape(xin[i,:,:])[0], x_placeholder: xin[i,:,:], y_placeholder : yin[i,:,:], market_place : market[i,:,:], drate1 : .3, is_training1: True})
            sumlossdr,sumlossdf,sumlossg = sumlossdr + dLossReal,sumlossdf+dLossFake,sumlossg+gLoss                                        
        d_fake_count += 1

    if gLoss > 0.5:
        # Train the generator
        for i in range(20):
            _, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],
                                                      {shape1 : np.shape(xin[i,:,:])[0], x_placeholder: xin[i,:,:], y_placeholder : yin[i,:,:], market_place : market[i,:,:], drate1 : .3, is_training1: True})
            sumlossdr,sumlossdf,sumlossg = sumlossdr + dLossReal,sumlossdf+dLossFake,sumlossg+gLoss                                        

        g_count += 1

    if dLossReal > 0.45:
        # If the discriminator classifies real images as fake,
        # train discriminator on real values
        for i in range(20):
            _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
                                                      {shape1 : np.shape(xin[i,:,:])[0], x_placeholder: xin[i,:,:], y_placeholder : yin[i,:,:], market_place : market[i,:,:], drate1 : .3, is_training1: True})
            sumlossdr,sumlossdf,sumlossg = sumlossdr + dLossReal,sumlossdf+dLossFake,sumlossg+gLoss                                        

        
        d_real_count += 1

    if i % 1000 == 0:
        # Periodically display a sample image in the notebook
        # (These are also being sent to TensorBoard every 10 iterations)
        images = sess.run(generator(xin[1,:,:], .3, True, np.shape(xin[i,:,:])[0]))
        d_result = sess.run(discriminator(x_placeholder, y_placeholder, drate1, is_training1, market_place), {shape1 : np.shape(xin[i,:,:])[0], x_placeholder: xin[1,:,:], y_placeholder : yin[1,:,:], market_place : market[1,:], drate1 : 0, is_training1: False})

    if i % 5000 == 0:
        save_path = saver.save(sess, "models/pretrained_gan.ckpt", global_step=i)
        print("saved to %s" % save_path)

#tf.summary.scalar('d_on_generated_eval', d_on_generated)
#tf.summary.scalar('d_on_real_eval', d_on_real)


