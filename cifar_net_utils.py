# coding: utf-8
# @author : Abhishek R S

import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from cifar_data_loader import cifar_data_loader

# read the json file and return the content
def read_config_file(json_file_name):
    # open and read the json file
    config = json.load(open(json_file_name))

    # return the content
    return config


# create the model directory if not present
def init(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# return training or test data
def get_data(is_training, directory, data_format):
    data, labels = None, None
    
    if not is_training:
        test_data = "test_batch"        
        data, labels = cifar_data_loader(os.path.join(os.getcwd(), os.path.join(directory, test_data)), data_format)
    else:
        train_data = "data_batch_"
         
        for i in range(1, 6):
            temp_data, temp_labels = cifar_data_loader(os.path.join(os.getcwd(), os.path.join(directory, train_data + str(i))), data_format)
            
            if i == 1:
                data = temp_data
                labels = temp_labels
            else:
                data = np.vstack((data, temp_data))
                labels = np.vstack((labels, temp_labels)) 

    data = data / 255.0 
    return (data, labels)

# return labels in one-hot encoding manner
def get_preprocessed_labels(all_labels, training = False):
    lbl_encoder = LabelEncoder()
    all_labels = lbl_encoder.fit_transform(all_labels)
    
    if training == False:
        return all_labels, lbl_encoder.classes_

    all_labels = all_labels.reshape(-1, 1)
    lbl_onehot_encoder = OneHotEncoder(categorical_features = [0])
    all_labels = lbl_onehot_encoder.fit_transform(all_labels).toarray()

    return all_labels


# split into train and validation set
def get_train_validation_set(all_images, all_labels, validation_size = 0.01):
    train_images, valid_images, train_labels, valid_labels = train_test_split(all_images, all_labels, test_size = validation_size)
    
    return (train_images, train_labels, valid_images, valid_labels)


# return the accuracy score of the predictions by the model
def get_accuracy_score(labels_groundtruth, labels_predicted):
    return accuracy_score(labels_groundtruth, labels_predicted)


# return the confusion matrix of the predictions by the model
def get_confusion_matrix(labels_groundtruth, labels_predicted):
    return confusion_matrix(labels_groundtruth, labels_predicted)
