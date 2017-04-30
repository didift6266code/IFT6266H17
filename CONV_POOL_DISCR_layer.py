import theano
import lasagne
import numpy as np

class Conv_pool_discr_layer(object):
    
    def __init__(self, batch_size=4, num_filters=2, filter_channels=3, filter_size=3,\
                 stride=1,pad=1):
        
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
        
        
    def conv_pool_discr(self, X, activation_type='relu', testF=False):        
                          
        conv_dis_pool_net = lasagne.layers.Conv2DLayer(X, num_filters = self.num_filters, filter_size = 3, stride=1, pad=1,\
                            untie_biases=False, W = self.W, b=lasagne.init.Constant(0.),\
                            nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.2), flip_filters=True,\
                            convolution=theano.tensor.nnet.conv2d)

        
        conv_dis_pool_net = lasagne.layers.batch_norm(conv_dis_pool_net)
        
        conv_dis_pool_net = lasagne.layers.MaxPool2DLayer(conv_dis_pool_net, pool_size = 2, stride=2, pad=0, ignore_border=True)

        
        return conv_dis_pool_net

    
        
        
    def conv_pool_discr_first(self, X, activation_type='relu', testF=False):        
                          
        conv_dis_pool_net = lasagne.layers.Conv2DLayer(X, num_filters = self.num_filters, filter_size = 3, stride=1, pad=1,\
                            untie_biases=False, W = self.W, b=lasagne.init.Constant(0.),\
                            nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.2), flip_filters=True,\
                            convolution=theano.tensor.nnet.conv2d)

        
        conv_dis_pool_net = lasagne.layers.MaxPool2DLayer(conv_dis_pool_net, pool_size = 2, stride=2, pad=0, ignore_border=True)

        
        return conv_dis_pool_net
    