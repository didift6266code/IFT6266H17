'''
@article{Im2015,
    title={Generating Images with Recurrent Adversarial Networks },
    author={Im, Daniel Jiwoong and Kim, Chris Dongjoo and Jiang, Hui and Memisevic, Roland},
    journal={http://arxiv.org/abs/1602.05110},
    year={2016}
}

'''
import numpy as np
import theano
import theano.tensor as T
import lasagne
from theano.tensor.shared_randomstreams import RandomStreams
import theano.sandbox.rng_mrg as RNG_MRG
rng = np.random.RandomState()

def activation_fn_th(X,activation_type='sigmoid', leak_thrd=0.2):
    '''collection of useful activation functions'''

    if activation_type == 'softmax':
        return lasagne.nonlinearities.softmax(X)
    elif activation_type == 'sigmoid':
        return lasagne.nonlinearities.sigmoid(X)
    elif activation_type == 'tanh':
        return lasagne.nonlinearities.tanh(X)
    elif activation_type == 'softplus':
        return lasagne.nonlinearities.softplus(X)
    elif activation_type == 'relu':
        return lasagne.nonlinearities.rectify(X)#(X + abs(X)) / 2.0
    elif activation_type == 'linear':
        return X
    elif activation_type =='leaky':
        return lasagne.nonlinearities.LeakyRectify(leakiness=0.2)
        '''
        f1 = 0.5 * (1 + leak_thrd)
        f2 = 0.5 * (1 - leak_thrd)
        return f1 * X + f2 * abs(X)
        '''
    
def intX(X):
    return np.asarray(X, dtype=np.int32)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)

def sharedNs(shape, n, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape)*n, dtype=dtype, name=name)  

def init_conv_weights(filter_shape, numpy_rng, rng_dist='normal'):
    """
    initializes the convnet weights.
    """

    #if 'uniform' in rng_dist:
    #    return np.asarray(
    #        numpy_rng.uniform(low=W_low, high=W_high, size=filter_shape),
    #            dtype=theano.config.floatX) 
    #elif rng_dist == 'normal':
    return  numpy_rng.normal(loc=0.0, scale=0.02, size=filter_shape).astype(theano.config.floatX)



''' improving learning rate'''
def get_epsilon_inc(epsilon, n, i):
    """
    n: total num of epoch
    i: current epoch num
    """
    return epsilon / ( 1 - i/float(n))

'''decaying learning rate'''
def get_epsilon(epsilon, n, i):
    """
    n: total num of epoch
    i: current epoch num
    """
    return epsilon / ( 1 + i/float(n))

def get_epsilon_decay(i, num_epoch, constant=4): 
    c = np.log(num_epoch/2)/ np.log(constant)
    return 10.**(1-(i-1)/(float(c)))
