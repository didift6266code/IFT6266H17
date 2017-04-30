import theano
import theano.tensor as T
import lasagne
import numpy as np

TINY    = 1e-6
#rng = RandomStreams(seed=np.random.randint(1 << 30))
class Conv_Upconv_layer(object):
    
    def __init__(self, batch_size=100, num_filters=100, filter_channels=3, filter_size=4,\
                 stride=2,pad=1):
        
        """Parameter Initialization for Batch Norm"""
        self.batch_size = batch_size
        #self.numpy_rng = numpy_rng
        self.filter_channels = filter_channels
        self.num_filters=num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        

        
    def conv(self, X, activation_type='relu', testF=False):  
        if activation_type=='tanh':
            linearite=lasagne.nonlinearities.tanh
        else:
            linearite=lasagne.nonlinearities.rectify
            
        conv_net_layer = lasagne.layers.Conv2DLayer(X, num_filters = self.num_filters, filter_size = self.filter_size,\
                                     stride=self.stride,pad=self.pad, untie_biases=False,\
                                     W= lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=linearite,\
                                     flip_filters=True, convolution=theano.tensor.nnet.conv2d)                            
        
        conv_net_layer_tide = conv_net_layer
        
        conv_net_layer = lasagne.layers.batch_norm(conv_net_layer)
        
        return conv_net_layer,conv_net_layer_tide

    
    def upconv(self, X , channels, filter_size=4,stride=2, pad=1, activation_type='relu', testF=False):
        
        if activation_type=='tanh':
            linearite=lasagne.nonlinearities.tanh
        else:
            linearite=lasagne.nonlinearities.rectify
        
        
        deconv_net_layer = lasagne.layers.TransposedConv2DLayer(X, channels,
                            filter_size, stride=stride, crop = pad,
                            W = lasagne.init.GlorotUniform(), nonlinearity=linearite)
                
        deconv_net_layer=lasagne.layers.batch_norm(deconv_net_layer)
        
        return deconv_net_layer    

    
    
    def upconv_out(self, X , channels, filter_size=4,stride=2, pad=1, activation_type='tanh', testF=False):
        
        if activation_type=='tanh':
            linearite=lasagne.nonlinearities.tanh
        else:
            linearite=lasagne.nonlinearities.sigmoid
        
        
        deconv_net_layer = lasagne.layers.TransposedConv2DLayer(X, channels,
                            filter_size, stride=stride, crop = pad,
                            W = lasagne.init.GlorotUniform(), nonlinearity=linearite)
                
        #deconv_net_layer=lasagne.layers.batch_norm(deconv_net_layer)
        
        return deconv_net_layer    
