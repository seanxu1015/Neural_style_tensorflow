# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:32:54 2016

@author: sean
"""

from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io
import sys
from config import Config
import time


class NeuralArt(object):
    
    def __init__(self, config):
        self.config = config
        self.load_data()
        content_features = self.add_content_features()
        style_features = self.add_style_features()
        self.add_model(content_features, style_features)
        self.loss_op = self.add_loss_op()
        self.train_op = self.add_train_op()
        self.init_op = tf.initialize_all_variables()
        
    def load_data(self):
        self.content_layer = self.config.content_layer
        self.style_layers = self.config.style_layers
        def image_read(path):
            return scipy.misc.imread(path).astype(np.float)
        pretrained_model = scipy.io.loadmat(self.config.vgg_path)
        mean = pretrained_model['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))
        self.mean_pixel = mean_pixel
        content_image = image_read(self.config.content_path) - mean_pixel
        self.content_image = np.expand_dims(content_image, axis=0)
        style_images = [image_read(s) - mean_pixel 
                        for s in self.config.style_path]
        self.style_images = map(lambda x: np.expand_dims(x, axis=0), 
                                style_images)
        self.content_shape = self.content_image.shape
        self.style_shapes = [style.shape for style in self.style_images]
        self.pretrained_weights = pretrained_model['layers'][0]

        if self.config.initial_image:
            initial_image = image_read(self.config.initial_image)
            initial_image = scipy.misc.imresize(initial_image, 
                                                self.content_shape[1:])
            self.initial_image = np.array([initial_image - 
                                           mean_pixel]).astype('float32')
        else:
            self.initial_image = np.random.normal(size=self.content_shape) \
                                 * 0.256
    
    def add_feature_extrator(self, image):
        net = {}
        current = image
        for i, name in enumerate(self.config.layers):
            kind = name[:4]
            if kind == 'conv':
                weights, bias = self.pretrained_weights[i][0][0][0][0]
                weights = tf.constant(np.transpose(weights, (1, 0, 2, 3)))
                bias = tf.constant(bias.reshape(-1))
                current = tf.nn.bias_add(tf.nn.conv2d(current, weights,
                          strides=(1, 1, 1, 1), padding='SAME'), bias)
            elif kind == 'relu':
                current = tf.nn.relu(current)
            elif kind == 'pool':
                current = tf.nn.max_pool(current, ksize=(1, 2, 2, 1),
                          strides=(1, 2, 2, 1), padding='SAME')
            net[name] = current
        return net
    
    def add_content_features(self):
        content_features = {}
        g = tf.Graph()
        layer = self.content_layer
        with g.as_default(), g.device('/cpu:0'), tf.Session():
            image = tf.placeholder('float', shape=self.content_shape)
            net = self.add_feature_extrator(image)
            feed_dict = {image: self.content_image}
            content_features[layer] = net[layer].eval(feed_dict=feed_dict)
        return content_features
        
    def add_style_features(self):
        num_styles = len(self.style_images)
        style_features = [{} for _ in range(num_styles)]
        for i in range(num_styles):
            g = tf.Graph()
            with g.as_default(), g.device('/cpu:0'), tf.Session():
                image = tf.placeholder('float', shape=self.style_shapes[i])
                net = self.add_feature_extrator(image)
                feed_dict = {image: self.style_images[i]}
                for layer in self.style_layers:
                    features = net[layer].eval(feed_dict=feed_dict)
                    features = np.reshape(features, (-1, features.shape[3]))
                    gram = np.matmul(features.T, features) / features.size
                    style_features[i][layer] = gram
        return style_features
    
    def add_model(self, content_features, style_features):
        self.c_fs = {self.content_layer: tf.constant(
                    content_features[self.content_layer], dtype=tf.float32)}
        self.s_fs = []
        for i in range(len(self.style_images)):
            features = {}
            for layer in self.style_layers:
                features[layer] = tf.constant(style_features[i][layer], 
                                  dtype=tf.float32)
            self.s_fs.append(features)
        self.image = tf.Variable(self.initial_image, dtype=tf.float32)
        self.net = self.add_feature_extrator(self.image)
            
    def add_loss_op(self):
        alpha = self.config.content_weight
        beta = self.config.style_weight
        c_l = self.content_layer
        s_ls = self.style_layers
        sbw = self.config.style_blend_weights
        if sbw:
            tbw = sum(sbw)
            sbw = [float(w) / tbw for w in sbw]
        else:
            sbw = [1.0 / len(self.style_images) for _ in self.style_images]
            
        def get_size(tensor):
            _, height, width, number = map(lambda x: x.value, 
                                           tensor.get_shape())           
            size = height * width * number
            return size, number
        size, _ = get_size(self.c_fs[c_l])
        content_loss = tf.nn.l2_loss(self.net[c_l] - self.c_fs[c_l]) / size
        style_loss = 0        
        for i in range(len(self.style_images)):
            style_losses = []
            for s_l in s_ls:
                layer = self.net[s_l]
                size, number = get_size(layer)
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(feats, feats, transpose_a=True) / size
                s_gram = self.s_fs[i][s_l]
                size = reduce(lambda x, y: x * y, 
                              map(lambda z: z.value, s_gram.get_shape()))
                temp_loss =  tf.nn.l2_loss(gram - s_gram) / size
                style_losses.append(temp_loss)
            style_loss += sbw[i] * reduce(tf.add, style_losses)
        loss = alpha * content_loss + beta * style_loss
        return loss
    
    def add_train_op(self):
        lr = self.config.learning_rate
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(self.loss_op)
        return train_op
        
    def run_epoch(self, session):
        session.run(self.init_op)
        best_loss = float('inf')
        start_time = time.time()
        for epoch in range(self.config.epochs):
            loss, _ = session.run([self.loss_op, self.train_op])
            if best_loss > loss:
                best_loss = loss
                best = self.image.eval(session=session)
            if epoch % 10 == 0:
                time_cost = int(time.time() - start_time)
                sys.stdout.write('{} / {}, time cost: {}, loss: {}\n'.format(epoch, 
                                 self.config.epochs, time_cost, loss))
        best = best.reshape(self.content_shape[1:]) + self.mean_pixel
        time_cost = int(time.time() - start_time)
        sys.stdout.write('Final loss: {}, total time cost: {}\n'.format(best_loss, time_cost))
        return best
        
    def stylizing(self):
        session = tf.Session()
        stylized_image = self.run_epoch(session)
        stylized_image = np.clip(stylized_image, 0, 255).astype(np.uint8)
        scipy.misc.imsave(self.config.output, stylized_image)

if __name__ == '__main__':
    config = Config()
    art = NeuralArt(config)
    art.stylizing()



    
