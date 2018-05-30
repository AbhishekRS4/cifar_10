import tensorflow as tf

class Network_Architecture:
    def __init__(self, img_pl, kernel_size, num_kernels, strides, data_format, padding, pool_size, training, neurons, num_classes, dropout_rate = 0.5, reduction_strides = None):
        self.img_pl = img_pl
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.strides = strides
        self.data_format = data_format
        self.padding = padding 
        self.pool_size = pool_size
        self.training = training
        self.neurons = neurons
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.reduction_strides = reduction_strides


    # build a vgg based encoder
    def vgg_encoder(self):
        # encoder 1
        self.conv1_1 = self._get_conv2d_layer(self.img_pl, self.num_kernels[0], self.kernel_size, self.strides, self.padding, self.data_format, "conv1_1")
        self.elu1_1 = self._get_elu_activation(self.conv1_1, "elu1_1") 
        self.conv1_2 = self._get_conv2d_layer(self.elu1_1, self.num_kernels[0], self.kernel_size, self.strides, self.padding, self.data_format, "conv1_2") 
        self.elu1_2 = self._get_elu_activation(self.conv1_2, "elu1_2")
        self.pool1 = self._get_maxpool2d_layer(self.elu1_2, self.pool_size, self.pool_size, self.padding, self.data_format, "pool1") 

        # encoder 2
        self.conv2_1 = self._get_conv2d_layer(self.pool1, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, "conv2_1") 
        self.elu2_1 = self._get_elu_activation(self.conv2_1, "elu2_1") 
        self.conv2_2 = self._get_conv2d_layer(self.elu2_1, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, "conv2_2") 
        self.elu2_2 = self._get_elu_activation(self.conv2_2, "elu2_2") 
        self.pool2 = self._get_maxpool2d_layer(self.elu2_2, self.pool_size, self.pool_size, self.padding, self.data_format, "pool2") 


    # build a residual encoder
    def residual_encoder(self):
        # encoder 1
        self.elu1_0 = self._strided_convolution_block(self.img_pl, self.num_kernels[0], self.kernel_size, self.strides, self.padding, self.data_format, 1)
         
        # encoder 2
        self.elu2_0 = self._strided_convolution_block(self.elu1_0, self.num_kernels[1], self.kernel_size, self.reduction_strides, self.padding, self.data_format, 2)
        self.fuse2_1 = self._residual_block(self.elu2_0, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, 2, 1)
        self.elu2_2 = self._get_elu_activation(self.fuse2_1, "elu2_2")
        self.fuse2_2 = self._residual_block(self.elu2_2, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, 2, 3)
        self.elu2_4 = self._get_elu_activation(self.fuse2_2, "elu2_4")

        # encoder 3
        self.elu3_0 = self._strided_convolution_block(self.elu2_4, self.num_kernels[2], self.kernel_size, self.reduction_strides, self.padding, self.data_format, 3)
        self.fuse3_1 = self._residual_block(self.elu3_0, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, 3, 1)
        self.elu3_2 = self._get_elu_activation(self.fuse3_1, "elu3_2")
        self.fuse3_2 = self._residual_block(self.elu3_2, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, 3, 3)
        self.elu3_4 = self._get_elu_activation(self.fuse3_2, "elu3_4")

    
    # build a dense network
    def dense_network(self, dense_input):
        self.flatten = self._get_flattened_features(dense_input, "flatten")
        self.dense1 = self._get_dense_layer(self.flatten, self.neurons[0], "dense1")
        self.dropout = self._get_dropout_layer(self.dense1, self.dropout_rate, self.training, "dropout")
        self.elu_dense = self._get_elu_activation(self.dropout, "elu_dense")

        self.logits = self._get_dense_layer(self.elu_dense, self.num_classes, "logits")


    # build a strided convolution block
    def _strided_convolution_block(self, input_layer, num_kernels, kernel_size, strides, padding, data_format, num):
        _conv1_0 = self._get_conv2d_layer(input_layer, num_kernels, kernel_size, strides, padding, data_format, "conv" + str(num) + "_0")
        _elu1_0 = self._get_elu_activation(_conv1_0, "elu" + str(num) +"_0")
        return _elu1_0

    # build a residual block
    def _residual_block(self, input_layer, num_kernels, kernel_size, strides, padding, data_format, num_1, num_2):
        _conv1 = self._get_conv2d_layer(input_layer, num_kernels, kernel_size, strides, padding, data_format, "conv" + str(num_1) + "_" + str(num_2))
        _elu1 = self._get_elu_activation(_conv1, "elu" + str(num_1) + "_" + str(num_2))
        _conv2 = self._get_conv2d_layer(_elu1, num_kernels, kernel_size, strides, padding, data_format, "conv" + str(num_1) + "_" + str(num_2 + 1))
        _fuse = tf.add(input_layer, _conv2, "fuse" + str(num_1) + "_" + str(num_2))
        return _fuse 


    # return ELU activation function
    def _get_elu_activation(self, input_tensor, name = "elu"):
        return tf.nn.elu(input_tensor, name = name)
    

    # return MaxPool2D layer
    def _get_maxpool2d_layer(self, input_tensor, pool_size, strides, padding, data_format, name = "pool"):
        return tf.layers.max_pooling2d(inputs = input_tensor, pool_size = pool_size, strides = strides, padding = padding, data_format = data_format, name = name)

    # return Convolution2D layer
    def _get_conv2d_layer(self, input_tensor, num_filters, kernel_size, strides, padding, data_format, name = "conv"):
        return tf.layers.conv2d(inputs = input_tensor, filters = num_filters, kernel_size = kernel_size, strides = strides, padding = padding, data_format = data_format, name = name)


    # return the dense layer
    def _get_dense_layer(self, input_tensor, num_neurons, name = "dense"):
        return tf.layers.dense(input_tensor, units = num_neurons, name = name)


    # return the flattened features
    def _get_flattened_features(self, input_tensor, name = "flatten"):
        return tf.layers.flatten(input_tensor, name = name)


    # return the dropout layer
    def _get_dropout_layer(self, input_tensor, rate = 0.5, training = False, name = "dropout"):
        return tf.layers.dropout(inputs = input_tensor, rate = rate, training = training, name = name)

