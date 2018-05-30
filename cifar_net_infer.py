# coding: utf-8
# @author : Abhishek R S

import os
import time
import numpy as np
import tensorflow as tf

from cifar_net_utils import read_config_file, get_data, get_preprocessed_labels, get_accuracy_score, get_confusion_matrix
from cifar_net_train import get_placeholders, get_softmax_layer
import network_architecture as na

param_config_file_name = os.path.join(os.getcwd(), "cifar_config.json")


# load the vgg based architecture
def load_model_vgg(img_pl, config):
    net_arch = na.Network_Architecture(img_pl, config['kernel_size'], config['num_kernels'], config['strides'], config['data_format'], config['padding'], config['pool_size'], not(config['TRAINING']), config['dense_layer_neurons'], config['NUM_CLASSES'], config['dropout_rate'])
    net_arch.vgg_encoder()
    vgg_out = net_arch.pool2
    net_arch.dense_network(vgg_out)
    logits = net_arch.logits

    return logits

# load the resnet based architecture
def load_model_res(img_pl, config):
    net_arch = na.Network_Architecture(img_pl, config['kernel_size'], config['num_kernels'], config['strides'], config['data_format'], config['padding'], config['pool_size'], not(config['TRAINING']), config['dense_layer_neurons'], config['NUM_CLASSES'], config['dropout_rate'], config['reduction_strides'])
    net_arch.residual_encoder()
    res_out = net_arch.elu3_4
    net_arch.dense_network(res_out)
    logits = net_arch.logits

    return logits

# run inference on test set
def infer():
    print("Reading the Config File..................")
    config = read_config_file(param_config_file_name)
    model_directory = config['model_directory'] + str(config['num_epochs'])
    print("Reading the Config File Completed........")
    print("")

    print("Reading Test Images.....................")
    all_images, all_labels = get_data(not(config['TRAINING']), config['DATA_PATH'], config['data_format'])
    print("Reading Test Images Completed...........")
    print("")

    print("Preprocessing the data...................")
    all_labels, all_original_classes = get_preprocessed_labels(all_labels, not(config['TRAINING']))
    all_original_classes = [int(x) for x in all_original_classes]
    print("Preprocessing of the data Completed......")
    print("")

 
    print("Loading the Network.....................")
    
    if config['data_format'] == 'channels_last':
        IMAGE_PLACEHOLDER_SHAPE = [None] + config['TARGET_IMAGE_SIZE'] + [config['NUM_CHANNELS']]
    else:
        IMAGE_PLACEHOLDER_SHAPE = [None] + [config['NUM_CHANNELS']] + config['TARGET_IMAGE_SIZE']
 
    img_pl = get_placeholders(img_placeholder_shape = IMAGE_PLACEHOLDER_SHAPE, training = not(config['TRAINING']))
    network_logits = load_model_res(img_pl, config)
    probs_prediction = get_softmax_layer(input_tensor = network_logits)
    print("Loading the Network Completed...........")
    print("")

    print("Images shape : " + str(all_images.shape))
    print("Labels shape : " + str(all_labels.shape))
    print("")    

    ss = tf.Session()
    ss.run(tf.global_variables_initializer())

    # load the model parameters
    tf.train.Saver().restore(ss, os.path.join(os.getcwd(), os.path.join(model_directory, config['model_file'])) + '-' + str(config['num_epochs']))

    print("")
    print("Inference Started.......................")
    ti = time.time()
    probs_predicted = ss.run(probs_prediction, feed_dict = {img_pl : all_images})
    ti = time.time() - ti
    print("Inference Completed.....................")
    print("Time Taken for Inference : " +str(ti))
    print("")

    logits_predicted_tensor = tf.convert_to_tensor(probs_predicted)
    output_labels = tf.argmax(logits_predicted_tensor, axis = 1)
    all_labels_predicted = ss.run(output_labels)
    all_labels_predicted = np.array(all_labels_predicted)

 
    print("Accuracy Score of the model : " + str(get_accuracy_score(all_labels, all_labels_predicted)))
    print("")
    print("Confusion Matrix for the prediction : ")
    print(get_confusion_matrix(all_labels, all_labels_predicted))
    print("")
    print("Original Labels : " + str(list(all_original_classes)))
    print("")
    ss.close()


def main():
    infer()

if __name__ == '__main__':
    main()
