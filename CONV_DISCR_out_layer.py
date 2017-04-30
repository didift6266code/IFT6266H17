import theano
import lasagne
import numpy as np

class Conv_discr_out_layer(object):
    
    def __init__(self, batch_size=4, num_filters=1, filter_channels=3, filter_size=4,\
                 stride=1,pad=0):
        
        """Parameter Initialization for Batch Norm"""
        self.batch_size = batch_size
        self.filter_channels = filter_channels
        self.num_filters=num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        
        
        self.gamma       = theano.shared(np.ones((num_filters,), dtype=theano.config.floatX), borrow=True, name='gamma') 
        self.beta        = theano.shared(np.zeros((num_filters,), dtype=theano.config.floatX), borrow=True, name='beta')
        self.stat_mean   = theano.shared(np.zeros((num_filters,), dtype=theano.config.floatX), borrow=True, name='running_avg')
        self.stat_std    = theano.shared(np.zeros((num_filters,), dtype=theano.config.floatX), borrow=True, name='running_std')
        self.inv_std     = theano.shared(np.ones((num_filters,), dtype=theano.config.floatX), borrow=True, name='running_inv_std')
        
        
        self.W = theano.shared( np.asarray(np.random.normal(0.0, 0.02, (num_filters,filter_channels,filter_size,filter_size)), dtype='float32'), borrow=True, name='W_conv')


        self.params = [self.W]

        
        self.params += [self.gamma, self.beta]
        
        
    def conv_discr_out(self, X, testF=False): 
                
        conv_dis_net = lasagne.layers.Conv2DLayer(X, num_filters = self.num_filters, filter_size = 4, stride=1, pad=0,\
                            untie_biases=False, W = self.W , b=None,\
                            nonlinearity=lasagne.nonlinearities.sigmoid, flip_filters=True,\
                            convolution=theano.tensor.nnet.conv2d)
        
        
        return conv_dis_net
