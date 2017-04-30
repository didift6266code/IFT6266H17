import theano
import lasagne
import numpy as np

class Conv_discr_layer(object):
    
    def __init__(self, batch_size=4, num_filters=2, filter_channels=3, filter_size=3,\
                 stride=1,pad=1):
        
        """Parameter Initialization for Batch Norm"""
        self.batch_size = batch_size
        
        self.filter_channels = filter_channels
        self.num_filters=num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        
        self.W = theano.shared( np.asarray(np.random.normal(0.0, 0.02, (num_filters,filter_channels,filter_size,filter_size)), dtype='float32'), borrow=True, name='W_conv')
        
    def conv_discr(self, X, testF=False): 
                
        conv_dis_net = lasagne.layers.Conv2DLayer(X, num_filters = self.num_filters, filter_size = 3, stride=1, pad=1,\
                            untie_biases=False, W =  self.W , b=None,\
                            nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.2), flip_filters=True,\
                            convolution=theano.tensor.nnet.conv2d)

        
        conv_dis_net = lasagne.layers.batch_norm(conv_dis_net)
        
        
        
        return conv_dis_net
    
    
