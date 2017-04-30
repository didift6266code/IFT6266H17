import theano
import lasagne
import numpy as np

class Conv_discr_out_layer(object):
    
    def __init__(self, batch_size=4, num_filters=1, filter_channels=3, filter_size=2,\
                 stride=1,pad=0):
        
        """Parameter Initialization for Batch Norm"""
        self.batch_size = batch_size
        #self.numpy_rng = numpy_rng
        self.filter_channels = filter_channels
        self.num_filters=num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        
        self.W = theano.shared( np.asarray(np.random.normal(0.0, 0.02, (num_filters,filter_channels,filter_size,filter_size)), dtype='float32'), borrow=True, name='W_conv')
                
        
    def conv_discr_out(self, X, testF=False): 
                
        conv_dis_net = lasagne.layers.Conv2DLayer(X, num_filters = self.num_filters, filter_size = self.filter_size,\
                                                  stride=self.stride, pad=self.pad,\
                            untie_biases=False, W = self.W , b=None,\
                            nonlinearity=lasagne.nonlinearities.sigmoid, flip_filters=True,\
                            convolution=theano.tensor.nnet.conv2d)
        
        
        return conv_dis_net
