# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:31:40 2016

@author: sean
"""


class Params(object):
    vgg_path = 'imagenet-vgg-verydeep-19.mat'
    layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 
              'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
              'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 
              'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
              'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
              'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
              'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
              'relu5_3', 'conv5_4', 'relu5_4')   
    content_layer = 'relu4_2'
    style_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
    learning_rate = 10.0
    epochs = 1000


class Config(Params):
    # origin image path, required
    content_path = './contents/landscape02.jpg'
    # path list of the style images, required
    style_path = ['./styles/shanshui.jpg']
    # output path, required
    output = './outputs/landscape02_shanshui.jpg'
    # initial image path, default None
    initial_image = None
    # weight balance between content and style
    content_weight = 1.0
    style_weight = 10.0
    # weights between multiple style images, the length must be the same as the style images.
    style_blend_weights = None
