import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import pickle
from loadCifar import load_cifar10_test,load_cifar10_train,get_cifar_info

num_classes = 10
learning_rate = 0.0001
size_of_filter1 = 5
num_features_f1 = 64
size_of_filter2 = 5
num_features_f2 = 64
size_of_fc1 = 256
size_of_fc2 = 128
size_of_cropped = 24
epochs = 10000
img_size = 32
n_channels = 3
n_classes = 10
path = '/home/kirito/Python/TensorFlow-Book/ch09_cnn/cifar-10-batches-py'



sess = tf.InteractiveSession(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
_,class_names,_ = get_cifar_info(path)
images_train, cls_train, labels_train = load_cifar10_train(path)
images_test, cls_test, labels_test = load_cifar10_test(path)


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(shape):
    return tf.Variable(tf.constant(0.05, shape=[shape]))

def create_conv_layer_with_biases(input, number_channels, filter_size, num_filters):
    shape = [filter_size, filter_size, number_channels, num_filters]
    w = create_weights(shape)
    b = create_biases(num_filters)
    output = tf.nn.conv2d(input = input, filter=w,strides=[1,1,1,1],padding='SAME')
    output += b
    output = tf.nn.max_pool(value=output, ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'SAME')
    output = tf.nn.relu(output)
    return output, w

def create_flatten_layer(input):
    shape = input.get_shape()
    num_features = shape[1:4].num_elements()
    flatten = tf.reshape(input, [-1, num_features])
    return flatten, num_features

def create_fc_layer(input, num_in, num_out):
    w = create_weights(shape=[num_in,num_out])
    b = create_biases(num_out)
    output = tf.matmul(input, w) + b
    output = tf.nn.relu(output)
    return output

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, n_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_con1, w_conv1 = create_conv_layer_with_biases(input = x,number_channels=n_channels,filter_size=size_of_filter1,num_filters=num_features_f1)
layer_con2, w_conv2 = create_conv_layer_with_biases(input = layer_con1,number_channels=num_features_f1,filter_size=size_of_filter2,num_filters=num_features_f2)
layer_flatten, num_features = create_flatten_layer(layer_con2)
layer_fc1 = create_fc_layer(input = layer_flatten, num_in=num_features,num_out=size_of_filter1)
layer_fc2 = create_fc_layer(input = layer_fc1, num_in=size_of_filter1,num_out=size_of_filter2)
layer_fc3 = create_fc_layer(input = layer_fc2, num_in=size_of_filter2,num_out=n_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

y_result = tf.nn.softmax(layer_fc3)
y_sc = tf.argmax(y_result, dimension=1)

global_step = tf.Variable(initial_value=0,name='global_step', trainable=False)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
correct_prediction = tf.equal(y_sc, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def pre_process_data_for_training(image):
    image = tf.random_crop(image, size=[size_of_cropped, size_of_cropped, n_channels])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)
    return image

def pre_process(images):
    images = tf.map_fn(lambda image:pre_process_data_for_training(image), images)
    return images

distorted_images = pre_process(images=x)

sess.run(tf.global_variables_initializer())
train_batch_size = 64

def random_batch():
    num_images = len(images_train)
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]
    return x_batch, y_batch

total_iterations = 0

def train(num_iterations):
    start_time = time.time()
    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        i_global, _ = sess.run([global_step, optimizer],feed_dict=feed_dict_train)
        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            msg = "Epochs: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))
    
    end_time = time.time()

    time_dif = end_time - start_time
    print("Train tooks: " + str(timedelta(seconds=int(round(time_dif)))))


train(epochs)