import numpy as np
import numpy.linalg as la
import re
import math
import tensorflow as tf

'''
l1-l1 LeSITA operator implementation 1
'''
def proxLeSITA(u, z, mut):
    print('LeSITA activation')

    zerost = np.zeros_like(u)
    output = np.zeros_like(u)
    #---- z>=0 ---#
    mask_11 = tf.logical_and((z >= zerost), (u < -2*mut))
    mask_12 = tf.logical_and((z >= zerost), tf.logical_and(-2*mut <= u, u <= zerost))
    mask_13 = tf.logical_and((z >= zerost), tf.logical_and(zerost < u, u < z))
    mask_14 = tf.logical_and((z >= zerost), tf.logical_and(z <= u, u <= z+2*mut))
    mask_15 = tf.logical_and((z >= zerost), (u > z+2*mut))
    out_11  =  tf.multiply(tf.cast(mask_11 , tf.float32), (u + 2*mut))
    out_12  =  tf.multiply(tf.cast(mask_12 , tf.float32), (zerost))
    out_13  =  tf.multiply(tf.cast(mask_13 , tf.float32), (u))
    out_14  =  tf.multiply(tf.cast(mask_14 , tf.float32), (z))
    out_15  =  tf.multiply(tf.cast(mask_15 , tf.float32), (u - 2*mut))
    #---- z<0 ---#
    mask_21 = tf.logical_and((z < zerost), (u < z-2*mut))
    mask_22 = tf.logical_and((z < zerost), tf.logical_and(z-2*mut <= u, u <= z))
    mask_23 = tf.logical_and((z < zerost), tf.logical_and(z < u, u < zerost))
    mask_24 = tf.logical_and((z < zerost), tf.logical_and(zerost <= u, u <= 2*mut))
    mask_25 = tf.logical_and((z < zerost), (u > 2*mut))
    out_21  = tf.multiply(tf.cast(mask_21, tf.float32), (u + 2*mut))
    out_22  = tf.multiply(tf.cast(mask_22, tf.float32), (z))
    out_23  = tf.multiply(tf.cast(mask_23, tf.float32), (u))
    out_24  = tf.multiply(tf.cast(mask_24, tf.float32), (zerost))
    out_25  = tf.multiply(tf.cast(mask_25, tf.float32), (u-2*mut))

    output = out_11+out_12+out_13+out_14+out_15 + out_21+out_22+out_23+out_24+out_25 

    return output

"""
**LeSITA_iman is a (30% faster) implementation of the LeSITA operator using 
  ReLU functions
**due to numerical considerations the same function (proxLeSITA or LeSITA_iman) 
  should be used for both training and inference
"""
def LeSITA_iman(a,w,mu):
    w1 = abs(w)
    out = tf.multiply(tf.sign(w),(tf.nn.relu(tf.sign(w)*a-2*mu-w1)-tf.nn.relu(tf.sign(w)*a-w1)+tf.nn.relu(tf.sign(w)*a)-tf.nn.relu(-tf.sign(w)*a-2*mu)))
    
    return out

'''
Soft thresholding operator
'''

def ShLU(a, th):
    return tf.sign(a)*tf.maximum(0.0, tf.abs(a)-th)

