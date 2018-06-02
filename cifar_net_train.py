# coding: utf-8
# @author : Abhishek R S

import math
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from cifar_net_utils import init, read_config_file, get_data, get_preprocessed_labels, get_train_validation_set
import network_architecture as na

param_config_file_name = os.path.join(os.getcwd(), "cifar_config.json")


# return the output of the softmax layer
def get_softmax_layer(input_tensor, name = 'softmax'):
    prediction = tf.nn.softmax(input_tensor, name = name)
    return prediction


# return the loss function which has to be minimized
def get_loss_function(groundtruth_labels, predicted_logits, name = 'categorical_cross_entropy'):
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = groundtruth_labels, logits = predicted_logits, name = name))
    return loss_function


# return the optimizer which has to be used to minimize the loss function
def get_optimizer(learning_rate, loss_function):
    adam_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_function)
    return adam_optimizer


# return the placeholder
def get_placeholders(img_placeholder_shape, training = False, lbl_placeholder_shape = None):
    img_pl = tf.placeholder(tf.float32, shape = img_placeholder_shape)
    # set the image placeholder

    if training:
        # set the label placeholder in the training phase
        label_pl = tf.placeholder(tf.float32, shape = lbl_placeholder_shape)
        return (img_pl, label_pl)

    return img_pl


# save the trained model
def save_model(session, model_directory, model_file, epoch):
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), os.path.join(model_directory, model_file)), global_step = (epoch + 1))


def batch_train():

    print("Reading the config file..................")
    config = read_config_file(param_config_file_name)
    print("Reading the config file completed........")
    print("")

    print("Initializing.............................")
    model_directory = config['model_directory'] + str(config['num_epochs'])
    init(model_directory)
    print("Initializing completed...................")
    print("")

    print("Reading train data.......................")
    all_images, all_labels = get_data(bool(config['TRAINING']), config['DATA_PATH'], config['data_format'])
    print("All data shape : " + str(all_images.shape))
    print("All labels shape : " + str(all_labels.shape))
    print("Reading train data completed.............")
    print("")

    print("Preprocessing the data...................")
    all_labels = get_preprocessed_labels(all_labels, bool(config['TRAINING']))
    train_images, train_labels, valid_images, valid_labels = get_train_validation_set(all_images, all_labels) 
    print("Preprocessing of the data completed......")
    print("")


    print("Building the network.....................")
     
    if config['data_format'] == 'channels_last': 
        IMAGE_PLACEHOLDER_SHAPE = [None] + config['TARGET_IMAGE_SIZE'] + [config['NUM_CHANNELS']]
    else:
        IMAGE_PLACEHOLDER_SHAPE = [None] + [config['NUM_CHANNELS']] + config['TARGET_IMAGE_SIZE']
    
    LABEL_PLACEHOLDER_SHAPE = [None] + [config['NUM_CLASSES']]
    img_pl, lbl_pl = get_placeholders(img_placeholder_shape = IMAGE_PLACEHOLDER_SHAPE, training = bool(config['TRAINING']), lbl_placeholder_shape = LABEL_PLACEHOLDER_SHAPE)
    
    '''
    net_arch = na.Network_Architecture(img_pl, config['kernel_size'], config['num_kernels'], config['strides'], config['data_format'], config['padding'], config['pool_size'], bool(config['TRAINING']), config['dense_layer_neurons'], config['NUM_CLASSES'], config['dropout_rate'])
    net_arch.vgg_encoder()
    vgg_out = net_arch.pool2
    net_arch.dense_network(vgg_out)
    logits = net_arch.logits
    '''

    net_arch = na.Network_Architecture(img_pl, config['kernel_size'], config['num_kernels'], config['strides'], config['data_format'], config['padding'], config['pool_size'], bool(config['TRAINING']), config['dense_layer_neurons'], config['NUM_CLASSES'], config['dropout_rate'], config['reduction_strides'])
    net_arch.residual_encoder()
    logits = net_arch.logits
    

    loss = get_loss_function(lbl_pl, logits)
    optimizer = get_optimizer(config['learning_rate'], loss)
 
    print("Building the network completed...........")
    print("")
    
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_batches = int(math.ceil(train_images.shape[0] / float(batch_size)))
    
    print("Train Images shape : " + str(train_images.shape))
    print("Train Labels shape : " + str(train_labels.shape))
    print("Validation Images shape : " + str(valid_images.shape))
    print("Validation Labels shape : " + str(valid_labels.shape))
    print("Number of epochs to train : " + str(num_epochs))
    print("Batch size : " + str(batch_size))
    print("Number of batches : " + str(num_batches))
    print("")

    print("Training the Network.....................")
    ss = tf.Session()
    ss.run(tf.global_variables_initializer())

    train_loss_per_epoch = list()
    valid_loss_per_epoch = list()
    #print(logits)
    
    #'''
    for epoch in range(num_epochs):
        ti = time.time()
        temp_loss_per_epoch = 0
        train_images, train_labels = shuffle(train_images, train_labels) 
        for batch_id in range(num_batches):
            batch_images = train_images[batch_id * batch_size : (batch_id + 1) * batch_size]
            batch_labels = train_labels[batch_id * batch_size : (batch_id + 1) * batch_size]
            
            _, loss_per_batch = ss.run([optimizer, loss], feed_dict = {img_pl : batch_images, lbl_pl : batch_labels})
            temp_loss_per_epoch += (batch_labels.shape[0] * loss_per_batch)
        ti = time.time() - ti
        loss_validation_set = ss.run(loss, feed_dict = {img_pl : valid_images, lbl_pl : valid_labels})
        train_loss_per_epoch.append(temp_loss_per_epoch)
        valid_loss_per_epoch.append(loss_validation_set)

        print("Epoch : " + str(epoch+1) + "/" + str(num_epochs) + ", time taken : " + str(ti) + " sec.")
        print("Avg. training loss : " + str(temp_loss_per_epoch / train_images.shape[0]))
        print("Avg. validation loss : " + str(loss_validation_set))
        print("")
    
    print("Training the Network Completed...........")
    print("")
    
    print("Saving the model.........................")
    save_model(ss, model_directory, config['model_file'], epoch)
    train_loss_per_epoch = np.array(train_loss_per_epoch)
    valid_loss_per_epoch = np.array(valid_loss_per_epoch)
    
    train_loss_per_epoch = np.true_divide(train_loss_per_epoch, train_images.shape[0])
   
    losses_dict = dict()
    losses_dict['train_loss'] = train_loss_per_epoch
    losses_dict['valid_loss'] = valid_loss_per_epoch

    np.save(os.path.join(os.getcwd(), os.path.join(model_directory, config['model_metrics'])), (losses_dict))
    print("Saving the model Completed...............")
    print("")
    #'''

    ss.close()

def main():
    batch_train()

if __name__ == '__main__':
    main()

