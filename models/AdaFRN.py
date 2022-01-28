# -*- coding: utf-8 -*-
# Time-Requency Resolution Attention Net(TFRANet) created by J. Liang in TJU
#==============================================================================
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Input, layers, regularizers
from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Layer, 
    UpSampling2D, MaxPooling2D,
    Conv2D, SeparableConv2D,
    ReLU,
    Add, 
    GlobalAveragePooling2D,
    Dense,
    BatchNormalization, ReLU, Softmax,
    concatenate, Lambda
    )
# from keras.utils.vis_utils import plot_model

class _weighted_add(Layer):
    """ Weighted add in BiFRN. 
        This custom layer contains learable weights 
        and calculate the importance of each branch by propogation.
    """
    
    def __init__(self, epsilon=1e-4, **kwargs):
        super(_weighted_add, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        """ Fast normalized fusion: O = sum((w_i * I_i) / (eps + sum(w_i)))"""
        w = keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(_weighted_add, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _SepConv(num_channels, kernel_size, strides, name):
    """SeparableConv-BN-ReLU"""
    def wrapper(input):
        x = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                    use_bias=False, name='{}_sepconv'.format(name))(input)
        x = BatchNormalization(name='{}_bn'.format(name))(x)
        res = ReLU(name='{}_relu'.format(name))(x)
        
        return res
        
    return wrapper
 
    
def bifpn(num_channels, id=0):
    """ 
        p5_i -------------------------- p5_o -------->
        p4_i ---------- p4_m ---------- p4_o -------->
        p3_i ---------- p3_m ---------- p3_o -------->
        p2_i ---------- p2_m ---------- p2_o -------->
        p1_i -------------------------- p1_o -------->
    ----args----
        inputs: list of tensors, len(inputs) = 5
        num_channels: W_bifpn in the paper, int
    """
    def wrapper(inputs):
        p1_i, p2_i, p3_i, p4_i, p5_i = inputs
        
        # Unify channel num with Conv1x1 only
        p5_i = Conv2D(num_channels, kernel_size=1, strides=1, name='BiFPN{}_P5'.format(id))(p5_i)
        p4_i = Conv2D(num_channels, kernel_size=1, strides=1, name='BiFPN{}_P4'.format(id))(p4_i)
        p3_i = Conv2D(num_channels, kernel_size=1, strides=1, name='BiFPN{}_P3'.format(id))(p3_i)
        p2_i = Conv2D(num_channels, kernel_size=1, strides=1, name='BiFPN{}_P2'.format(id))(p2_i)
        p1_i = Conv2D(num_channels, kernel_size=1, strides=1, name='BiFPN{}_P1'.format(id))(p1_i)
        
        # Connections for p5_i, p4_i to p4_m
        p4_m = _weighted_add(name='BiFPN{}_u5_wadd_4'.format(id))([UpSampling2D()(p5_i), p4_i])
        p4_m = _SepConv(num_channels, kernel_size=3, strides=1, name='BiFPN{}_Conv4m'.format(id))(p4_m)
        # Connections for p4_m, p3_i to p3_m
        p3_m = _weighted_add(name='BiFPN{}_u4m_wadd_3'.format(id))([UpSampling2D()(p4_m), p3_i])
        p3_m = _SepConv(num_channels, kernel_size=3, strides=1, name='BiFPN{}_Conv3m'.format(id))(p3_m)
        # Connections for p3_m, p2_i to p2_m
        p2_m = _weighted_add(name='BiFPN{}_u3m_wadd_2'.format(id))([UpSampling2D()(p3_m), p2_i])
        p2_m = _SepConv(num_channels, kernel_size=3, strides=1, name='BiFPN{}_Conv2m'.format(id))(p2_m)
        
        # Connections for p2_m, p1_i to p1_o
        p1_o = _weighted_add(name='BiFPN{}_u2m_wadd_1'.format(id))([UpSampling2D()(p2_m), p1_i])
        p1_o = _SepConv(num_channels, kernel_size=3, strides=1, name='BiFPN{}_Conv1o'.format(id))(p1_o)

        # Connections for p1_o, p2_i, p2_m to p2_o
        p2_o = _weighted_add(name='BiFPN{}_d1o_wadd_2im'.format(id))([MaxPooling2D(strides=(2, 2))(p1_o), p2_i, p2_m])
        p2_o = _SepConv(num_channels, kernel_size=3, strides=1, name='BiFPN{}_Conv2o'.format(id))(p2_o)
        # Connections for p2_o, p3_i, p3_m to p3_o
        p3_o = _weighted_add(name='BiFPN{}_d2o_wadd_3im'.format(id))([MaxPooling2D(strides=(2, 2))(p2_o), p3_i, p3_m])
        p3_o = _SepConv(num_channels, kernel_size=3, strides=1, name='BiFPN{}_Conv3o'.format(id))(p3_o)
        # Connections for p3_o, p4_i, p4_m to p4_o
        p4_o = _weighted_add(name='BiFPN{}_d3o_wadd_4im'.format(id))([MaxPooling2D(strides=(2, 2))(p3_o), p4_i, p4_m])
        p4_o = _SepConv(num_channels, kernel_size=3, strides=1, name='BiFPN{}_Conv4o'.format(id))(p4_o)
        
        # Connections for p4_o, p5_i to p5_o
        p5_o = _weighted_add(name='BiFPN{}_d4o_wadd_5i'.format(id))([MaxPooling2D(strides=(2, 2))(p4_o), p5_i])
        p5_o = _SepConv(num_channels, kernel_size=3, strides=1, name='BiFPN{}_Conv5o'.format(id))(p5_o)

        return p1_o, p2_o, p3_o, p4_o, p5_o
        
    return wrapper


###############################################################################
def _block_1(num_filters, if_trainable, id, relu_first=True):
    """ block_1 include two separable convolutions"""
    # Define id of block
    index_block = 'block' + str(id)
    
    def wrapper(input):
        # Residual connection
        residual = layers.Conv2D(num_filters, (1, 1),
                                 strides=(2, 2),
                                 padding='same',
                                 use_bias=False, 
                                 trainable=if_trainable,)(input)
        residual = layers.BatchNormalization()(residual)
        
        if relu_first == True:
            x = layers.Activation(activation='relu', name=index_block+'_sepconv1_act')(input)
        else:
            x = input
            
        # Sepconv1
        x = layers.SeparableConv2D(num_filters, (3, 3), 
                                   padding='same',
                                   use_bias=False, 
                                   trainable=if_trainable,
                                   name=index_block+'_sepconv1')(x)
        x = layers.BatchNormalization(name=index_block+'_sepconv1_bn')(x)
        # Sepconv2
        x = layers.Activation(activation='relu', name=index_block+'_sepconv2_act')(x)
        x = layers.SeparableConv2D(num_filters, (3, 3), 
                                   padding='same',
                                   use_bias=False, 
                                   trainable=if_trainable,
                                   name=index_block+'_sepconv2')(x)
        x = layers.BatchNormalization(name=index_block+'_sepconv2_bn')(x)
        # Pool operation
        x = layers.MaxPooling2D((3, 3), padding='same', 
                                strides=(2, 2), name=index_block+'_pool')(x)
        
        # Add res and x
        x = layers.add([x, residual])
        return x
        
    return wrapper
    
    
def _block_2(num_filters, if_trainable, id):
    """ block_2 include three separable convolutions"""    
    # Define id of block
    index_block = 'block' + str(id)
    
    def wrapper(input):
        # Residual connection
        residual = input
        # Sepconv1
        x = layers.Activation('relu', name=index_block + '_sepconv1_act')(input)
        x = layers.SeparableConv2D(num_filters, (3, 3), 
                                   padding='same',
                                   use_bias=False, 
                                   trainable=if_trainable,
                                   name=index_block + '_sepconv1')(x)
        x = layers.BatchNormalization(name=index_block + '_sepconv1_bn')(x)
        # Sepconv2
        x = layers.Activation('relu', name=index_block + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(num_filters, (3, 3), 
                                   padding='same', use_bias=False, 
                                   trainable=if_trainable,
                                   name=index_block + '_sepconv2')(x)
        x = layers.BatchNormalization(name=index_block + '_sepconv2_bn')(x)
        # Sepconv3
        x = layers.Activation('relu', name=index_block + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(num_filters, (3, 3), 
                                   padding='same', use_bias=False, 
                                   trainable=if_trainable,
                                   name=index_block + '_sepconv3')(x)
        x = layers.BatchNormalization(name=index_block + '_sepconv3_bn')(x)
        # Add res and x
        x = layers.add([x, residual])
        return x
        
    return wrapper
    

def _block_3(num1_filters, num2_filters, if_trainable, id):
    """ block_3 include two separable convolutions with bottleneck"""
    # Deine id of block
    index_block = 'block' + str(id)
    
    def wrapper(input):
        # Residual connection
        residual = layers.Conv2D(num2_filters, (1, 1), 
                                 strides=(2, 2), padding='same', 
                                 use_bias=False, 
                                 trainable=if_trainable,)(input)
        residual = layers.BatchNormalization()(residual)
        # Sepconv1
        x = layers.Activation('relu', name=index_block+'_sepconv1_act')(input)
        x = layers.SeparableConv2D(num1_filters, (3, 3), 
                                   padding='same', use_bias=False, 
                                   trainable=if_trainable,
                                   name=index_block+'_sepconv1')(x)
        x = layers.BatchNormalization(name=index_block+'_sepconv1_bn')(x)
        # Sepconv2
        x = layers.Activation('relu', name=index_block+'_sepconv2_act')(x)
        x = layers.SeparableConv2D(num2_filters, (3, 3), 
                                   padding='same', use_bias=False, 
                                   trainable=if_trainable,
                                   name=index_block+'_sepconv2')(x)
        x = layers.BatchNormalization(name=index_block+'_sepconv2_bn')(x)
        # Pool operation
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same',
                                name=index_block+'_pool')(x) 
        
        # Add res and x
        x = layers.add([x, residual])

        return x
        
    return wrapper


def backbone(if_trainable=True, include_top=False):
    """ Build a backbone as Feature Extractor
        ----args----
        if_trainable: freeze the layers, boolean
        include_top: if set to True, add GAP and softmax at the top. 
                     Or add max pool, and output multi-scale feature
        ---return---
        tensors
        ---examples---
        input = keras.Input()
        x = backbone(if_trainable=True, include_top=True)(input) #single scale
        model = keras.models.Model(input, x)
    """
    def wrapper(input):
        # Conv1, strides = 2
        x = layers.Conv2D(32, (3, 3),
                          strides=(2, 2),
                          padding='same',
                          use_bias=False, 
                          name='block1_conv1',
                          trainable=if_trainable,)(input)
        x = layers.BatchNormalization(name='block1_conv1_bn')(x)
        x = layers.Activation('relu', name='block1_conv1_act')(x)
        # Conv2
        x = layers.Conv2D(64, (3, 3),
                          padding='same',
                          use_bias=False, 
                          name='block1_conv2',
                          trainable=if_trainable,)(x)
        x = layers.BatchNormalization(name='block1_conv2_bn')(x)
        x = layers.Activation('relu', name='block1_conv2_act')(x)
        
        # Construction built by block_1(s) with pool
        l_0 = _block_1(num_filters=128, if_trainable=if_trainable, 
                       id=2, relu_first=False)(x)
        l_1 = _block_1(num_filters=256, if_trainable=if_trainable, id=3)(l_0)
        l_2 = _block_1(num_filters=728, if_trainable=if_trainable, id=4)(l_1)
        # Construction built by block_2(s)
        x = _block_2(num_filters=728, 
                    if_trainable=if_trainable, id=5)(l_2)
        # x = _block_2(num_filters=728, 
                    # if_trainable=if_trainable, id=6)(x)
        #
        # Additional block
        # x = _block_2(num_filters=728, 
                    # if_trainable=if_trainable, id=61)(x)
        # x = _block_2(num_filters=728, 
                    # if_trainable=if_trainable, id=62)(x)
        # x = _block_2(num_filters=728, 
                    # if_trainable=if_trainable, id=63)(x)
        # x = _block_2(num_filters=728, 
                    # if_trainable=if_trainable, id=64)(x)
        # x = _block_2(num_filters=728, 
                    # if_trainable=if_trainable, id=65)(x)
        # x = _block_2(num_filters=728, 
                    # if_trainable=if_trainable, id=66)(x)
        ##
        # Construction built by block_3 with pool
        l_3 = _block_3(num1_filters=728, num2_filters=1024, 
                         if_trainable=if_trainable, id=7)(x)
        # Sepconv1
        x = layers.SeparableConv2D(1536, (3, 3), 
                                   padding='same', use_bias=False, 
                                   trainable=if_trainable, 
                                   name='block8_sepconv1')(l_3)
        x = layers.BatchNormalization(name='block8_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block8_sepconv1_act')(x)
        # Sepconv2
        x = layers.SeparableConv2D(2048, (3, 3), 
                                   padding='same', use_bias=False, 
                                   trainable=if_trainable, 
                                   name='block8_sepconv2')(x)
        x = layers.BatchNormalization(name='block8_sepconv2_bn')(x)
        x = layers.Activation('relu', name='block8_sepconv2_act')(x)
        
        if include_top == True:
            x = GlobalAveragePooling2D()(x)
            preds = Dense(10, activation='softmax', name='Prediction')(x)
            
            return preds

        else:
            # Pool operation
            l_4 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same',
                                      name='block8_pool')(x) 
            
            return l_0, l_1, l_2, l_3, l_4 
        
    return wrapper

###############################################################################
def _mul_add(inputs, axis):
    """ reduce_sum(x * y, axis=axis) """
    x, y = inputs
    
    return tf.reduce_sum(tf.multiply(x, y), axis=axis)
    # return tf.reduce_sum(np.multiply(x, y), axis=axis)

    
def fsnet(r=16, L=8, name='FSnet'):
    """ 
    i_4 ---------|-------------------------------|- s_4 -|
    i_3 ---------|-------------------------------|- s_3 -|
    ...       fea_u ----> fea_s ----> fea_z -----|  ...  |-out
    i_0 ---------|-------------------------------|- s_0 -|
    ----Args----
        inputs: lists of tensors
        r: the radio for compute d, the length of fea_z.
        L: the minimum dim of the fea_z, default 8.
    """
    def wrapper(inputs):
        c = inputs[0].get_shape().as_list()[-1]
        d = max(c//r, L)
        
        # feats.shape=(n, len(input), h, w, c)
        for i in range(len(inputs)):
            if i == 0:
                fea_u = inputs[i]
                feats = Lambda(tf.expand_dims, arguments={'axis': 1})(inputs[i])
                
            else:
                vec = UpSampling2D((2**i, 2**i))(inputs[i])
                fea_u = layers.add([fea_u, vec], name=name+'_fuse_add{}'.format(i))
                vec = Lambda(tf.expand_dims, arguments={'axis': 1})(vec)
                feats = concatenate([feats, vec], axis=1)
        
        fea_s = GlobalAveragePooling2D(name=name+'_glp')(fea_u)
        fea_z = Dense(d, use_bias=False, kernel_regularizer=regularizers.l2(0.1), name=name+'_fc_z')(fea_s)
        fea_z = layers.BatchNormalization(name=name+'_fc_z_bn')(fea_z)
        fea_z = layers.Activation('relu', name=name+'_fc_z_relu')(fea_z)
        
        for i in range(len(inputs)):
            vec = Dense(c, use_bias=False, kernel_regularizer=regularizers.l2(0.1), name=name+'_fc_vec{}'.format(i))(fea_z)
            vec = Lambda(tf.expand_dims, arguments={'axis': 1})(vec)
            if i == 0:
                attention_vecs = vec
            else:
                attention_vecs = concatenate([attention_vecs, vec], axis=1)

        attention_vecs = Softmax(axis=1, name=name+'_softmax')(attention_vecs)
        # attention_vecs.shape from (n, len(inputs), c) to (n, len(inputs), 1, 1, c)
        attention_vecs = Lambda(tf.expand_dims, arguments={'axis': 2})(attention_vecs)
        attention_vecs = Lambda(tf.expand_dims, arguments={'axis': 3})(attention_vecs)

        out = Lambda(_mul_add, name=name+'_mul_add',
                     arguments={'axis': 1})([feats, attention_vecs])

        return out
        # return feats
        
    return wrapper
    
    
def AdaFRN(in_shape, num_channels=256, r=16, L=8):
    """ 
    ----args----
    in_shape:
    num_channels: W_bifpn in the paper, int. [64, 88, 112, 160, 224, 288, 384]
    r: the radio for compute d, the length of fea_z.
    L: the minimum dim of the fea_z, default 32.
    """
    backbone_trainable = True #backbone is trainable
    
    input = Input(shape=in_shape)
    
    xs = backbone(backbone_trainable, include_top=False)(input)
    
    xs = bifpn(num_channels)(xs)
    
    x = fsnet(r, L, name='FSnet')(xs)
    
    x = GlobalAveragePooling2D(name='GLP')(x)
    
    x = Dense(10, activation='softmax', name='franet_Prediction')(x)
    
    model = Model(input, x)
    
    return model
    
    
if __name__ == '__main__':
    model = AdaFRN((128, 512, 1))
    model.summary()
    # plot_model(model, to_file="model.png",show_shapes=True)