import os
import pickle
import numpy as np

# number of channels = 3, since images are rgb
num_channels = 3

# image dimension is 32 x 32
img_dimension = 32

# cifar data loader
def cifar_data_loader(pickled_data, data_format = "channels_first"):
    data_file = open(pickled_data, "rb")
    data_dict = pickle.load(data_file, encoding = "bytes")
    data = data_dict[b"data"].reshape(data_dict[b"data"].shape[0], num_channels, img_dimension, img_dimension)
    
    if data_format == "channels_last":
        data = np.transpose(data, [0, 2, 3, 1])
   
    labels = np.array(data_dict[b'labels']).reshape(-1, 1)

    return (data, labels) 

def main():
    data, labels = cifar_data_loader(os.path.join(os.getcwd(), os.path.join("cifar-10-batches-py", "test_batch")))
    print("cifar-10 data read successful")
    print("data shape : " + str(data.shape))
    print("labels shape : " + str(labels.shape))

if __name__ == "__main__":
    main()
