# coding: utf-8
"""Contains functions for add_dimension and one-hot encoder
"""
import numpy as np
import dcase_util

def add_dimension(train_data, eval_data):
    """add a dim of data to generate two tensor for fitting
    """
    train_data = np.expand_dims(train_data, axis=3)
    eval_data = np.expand_dims(eval_data, axis=3)
    return train_data, eval_data


def one_hot_encoder(labels_list, num_label_class, train_label, eval_label):
    """Encode the labels to one-hot format
    """
    #create a dict mapping from str to num
    dict = {}
    for index, label in enumerate(labels_list):
        dict[label] = index
    #encode train label
    one_hot_train_label = np.zeros(shape=(len(train_label), num_label_class))
    for i, scene in enumerate(train_label):
        j = dict[scene]
        one_hot_train_label[i][j] = 1
    #encode eval label
    one_hot_eval_label = np.zeros(shape=(len(eval_label), num_label_class))
    for i, scene in enumerate(eval_label):
        j = dict[scene]
        one_hot_eval_label[i][j] = 1
    return one_hot_train_label, one_hot_eval_label


def one_hot_decoder(labels_list, one_hot_labels):
    """Deconde one-hot labels into string format"""
    #create a dict mapping from num to str
    dict = {}
    str_labels = []
    for index, label in enumerate(labels_list):
        dict[index] = label
    #decode one-hot labels
    for item in one_hot_labels:
        maxindex  = np.argmax(item)
        str_labels.append(dict[maxindex])
    return str_labels
