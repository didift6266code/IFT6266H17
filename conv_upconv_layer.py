import theano
import theano.tensor as T
import lasagne
import numpy as np

class Conv_Upconv_layer(object):
    
    def __init__(self, batch_size=100, num_filters=100, filter_channels=3, filter_size=4,\
                 stride=2,pad=1):
        
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
        
    def conv(self, X, activation_type='relu', testF=False):  
        if activation_type=='tanh':
            linearite=lasagne.nonlinearities.tanh
        else:
            linearite=lasagne.nonlinearities.rectify
            
        conv_net_layer = lasagne.layers.Conv2DLayer(X, num_filters = self.num_filters, filter_size = self.filter_size,\
                                     stride=self.stride,pad=self.pad, untie_biases=False,\
                                     W=self.W, b=lasagne.init.Constant(0.), nonlinearity=linearite,\
                                     flip_filters=True, convolution=theano.tensor.nnet.conv2d)                            
        
        conv_net_layer_tide = conv_net_layer
        
        conv_net_layer = lasagne.layers.batch_norm(conv_net_layer)
        
        return conv_net_layer,conv_net_layer_tide

    
    def upconv_concat(self, X , X_tide, activation_type='relu', testF=False):
        
        if activation_type=='tanh':
            linearite=lasagne.nonlinearities.tanh
        else:
            linearite=lasagne.nonlinearities.rectify
        
        
        deconv_net_layer = lasagne.layers.TransposedConv2DLayer(X, X_tide.input_shape[1],
                            X_tide.filter_size, stride=X_tide.stride, crop=X_tide.pad,
                            W = lasagne.init.GlorotUniform(), nonlinearity=linearite,flip_filters=not X_tide.flip_filters)

        deconv_net_layer=lasagne.layers.batch_norm(deconv_net_layer)
        
        return deconv_net_layer    

    
    def upconv(self, X , X_tide, activation_type='relu', testF=False):
        
        if activation_type=='tanh':
            linearite=lasagne.nonlinearities.tanh
        else:
            linearite=lasagne.nonlinearities.rectify
        
        deconv_net_layer = lasagne.layers.TransposedConv2DLayer(X, X_tide.input_shape[1],
                            X_tide.filter_size, stride=X_tide.stride, crop=X_tide.pad,
                            W=X_tide.W , nonlinearity=linearite, flip_filters=not X_tide.flip_filters)

        
        deconv_net_layer=lasagne.layers.batch_norm(deconv_net_layer)
        
        return deconv_net_layer

    
    def upconv_out(self, X , X_tide, activation_type='tanh', testF=False):
        
        if activation_type=='tanh':
            linearite=lasagne.nonlinearities.tanh
        else:
            linearite=lasagne.nonlinearities.sigmoid
        
        deconv_net_layer = lasagne.layers.TransposedConv2DLayer(X, X_tide.input_shape[1],
                            X_tide.filter_size, stride=X_tide.stride, crop=X_tide.pad,
                            W=X_tide.W , nonlinearity=linearite, flip_filters=not X_tide.flip_filters)

        
        return deconv_net_layer
